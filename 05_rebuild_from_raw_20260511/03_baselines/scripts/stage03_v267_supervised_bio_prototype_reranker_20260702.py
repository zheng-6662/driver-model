#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v267 supervised bio prototype reranker。

v266 证明 vehicle-matched prototype 候选库本身有 headroom：
test bad_top10 candidate oracle k40 = 0.6166，接近 full oracle 0.6125。
但 v266 的可部署规则只是简单距离/投票，bio 只能小幅改善，仍远差于 fixed wait-latest。

本轮进一步做一个更强但仍守边界的检验：

    在 train split 上构造 query-prototype pair，标签为“把 prototype 的 oracle delay
    映射到 query 事件时的真实 tail RMSE”。训练一个 pairwise/listwise reranker，
    在 val 上选择 K/模型，test 只报告。

边界：
- prototype 仍只来自 train split；
- val/test 驾驶员历史不参与检索、不参与标签；
- query 输入只使用 0ms 车辆上下文与 observation_s 之前的 bio260_sp64；
- 生理只参与候选内部重排，不直接生成轨迹；
- candidate oracle 只作为 headroom，不作为可部署策略。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V266_SCRIPT = BASELINES / "scripts" / "stage03_v266_vehicle_matched_bio_residual_prototype_20260702.py"
OUT = BASELINES / "v267_supervised_bio_prototype_reranker_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v267_supervised_bio_prototype_reranker_20260702_pack.zip"

SEED = 26702
K_VALUES = [3, 5, 10, 20, 40]
FIXED_WAIT_LATEST_BADTOP10 = 0.695048

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_v266_module():
    """复用 v266 已通过 guardrail 的候选构造、检索和汇总函数。"""
    if not V266_SCRIPT.exists():
        raise FileNotFoundError(f"缺少 v266 脚本：{V266_SCRIPT}")
    spec = importlib.util.spec_from_file_location("v266_vehicle_matched_proto", V266_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 v266 脚本：{V266_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V266 = load_v266_module()


def ensure_dirs() -> None:
    for folder in (TABLES, FIGURES, REPORTS, LOGS):
        folder.mkdir(parents=True, exist_ok=True)


def clean_out_dir() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    ensure_dirs()
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_fill_scale(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只用 train pairs 拟合填充和标准化参数。"""
    with np.errstate(all="ignore"):
        med = np.nanmedian(x[train_mask], axis=0)
    med = np.asarray(med, dtype=float)
    med[~np.isfinite(med)] = 0.0
    filled = np.where(np.isfinite(x), x, med[None, :])
    mean = np.nanmean(filled[train_mask], axis=0)
    std = np.nanstd(filled[train_mask], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    z = (filled - mean[None, :]) / std[None, :]
    return z.astype(np.float32), med.astype(float), mean.astype(float), std.astype(float)


def rmse_at_delay(lookup: Dict[str, Dict[int, float]], event_uid: str, delay: int) -> Tuple[int, float]:
    return V266.rmse_at_delay(lookup, event_uid, int(delay))


def build_pair_dataset(
    events: pd.DataFrame,
    neighbors: pd.DataFrame,
    lookup: Dict[str, Dict[int, float]],
    veh_z: np.ndarray,
    bio_z: np.ndarray,
    max_k: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, List[str]]]:
    """构造 query-prototype pair 表和三组特征矩阵。"""
    event_index = {uid: i for i, uid in enumerate(events["event_uid"].astype(str).tolist())}
    scalar_rows: List[Dict[str, object]] = []
    vehicle_rows: List[np.ndarray] = []
    bio_rows: List[np.ndarray] = []
    base_rows: List[np.ndarray] = []

    scalar_feature_names = [
        "neighbor_rank_norm",
        "prototype_delay_norm",
        "vehicle_distance",
        "bio_distance",
        "prototype_wait_gain",
        "prototype_oracle_gap",
        "prototype_keep0_tail_rmse",
        "prototype_latest_tail_rmse",
        "prototype_oracle_tail_rmse",
    ]
    veh_feature_names = (
        [f"query_vehicle_z_{i:02d}" for i in range(veh_z.shape[1])]
        + [f"prototype_vehicle_z_{i:02d}" for i in range(veh_z.shape[1])]
        + [f"absdiff_vehicle_z_{i:02d}" for i in range(veh_z.shape[1])]
    )
    bio_feature_names = (
        [f"query_bio_z_{i:02d}" for i in range(bio_z.shape[1])]
        + [f"prototype_bio_z_{i:02d}" for i in range(bio_z.shape[1])]
        + [f"absdiff_bio_z_{i:02d}" for i in range(bio_z.shape[1])]
    )

    for _, nrow in neighbors.iterrows():
        rank = int(nrow["neighbor_rank_vehicle"])
        if rank > max_k:
            continue
        q_uid = str(nrow["event_uid"])
        p_uid = str(nrow["prototype_event_uid"])
        qi = event_index[q_uid]
        pi = event_index[p_uid]
        delay = int(nrow["prototype_oracle_delay_ms"])
        mapped_delay, target_rmse = rmse_at_delay(lookup, q_uid, delay)
        q_event = events.iloc[qi]
        p_event = events.iloc[pi]

        prototype_keep0 = float(p_event["keep0_tail_rmse_v241"])
        prototype_latest = float(p_event["latest_tail_rmse_v241"])
        prototype_oracle = float(p_event["oracle_tail_rmse_v241"])
        scalar = np.array(
            [
                rank / float(max_k),
                delay / 1000.0,
                float(nrow["vehicle_distance"]),
                float(nrow["bio_distance"]),
                prototype_keep0 - prototype_latest,
                prototype_latest - prototype_oracle,
                prototype_keep0,
                prototype_latest,
                prototype_oracle,
            ],
            dtype=np.float32,
        )
        qv = veh_z[qi]
        pv = veh_z[pi]
        qb = bio_z[qi]
        pb = bio_z[pi]
        vehicle_block = np.concatenate([qv, pv, np.abs(qv - pv)]).astype(np.float32)
        bio_block = np.concatenate([qb, pb, np.abs(qb - pb)]).astype(np.float32)

        scalar_rows.append(
            {
                "event_uid": q_uid,
                "split": str(q_event["split"]),
                "subject": str(q_event["subject"]),
                "prototype_event_uid": p_uid,
                "prototype_subject": str(p_event["subject"]),
                "neighbor_rank_vehicle": rank,
                "prototype_oracle_delay_ms": delay,
                "mapped_delay_ms": int(mapped_delay),
                "target_tail_rmse_v241": float(target_rmse),
                "keep0_tail_rmse_v241": float(q_event["keep0_tail_rmse_v241"]),
                "latest_tail_rmse_v241": float(q_event["latest_tail_rmse_v241"]),
                "oracle_tail_rmse_v241": float(q_event["oracle_tail_rmse_v241"]),
                "bad_top10": bool(q_event["bad_top10"]),
                "very_bad_top5": bool(q_event["very_bad_top5"]),
                "normal": bool(q_event["normal"]),
                "observe_later_like": bool(q_event["observe_later_like"]),
                "strong_steer": bool(q_event["strong_steer"]),
                "reverse": bool(q_event["reverse"]),
                "early_best_after_400": bool(q_event["early_best_after_400"]),
                "vehicle_distance": float(nrow["vehicle_distance"]),
                "bio_distance": float(nrow["bio_distance"]),
                "same_subject_as_prototype": bool(nrow["same_subject_as_prototype"]),
            }
        )
        base_rows.append(scalar)
        vehicle_rows.append(vehicle_block)
        bio_rows.append(bio_block)

    pair_meta = pd.DataFrame(scalar_rows)
    base = np.vstack(base_rows).astype(np.float32)
    vehicle = np.vstack(vehicle_rows).astype(np.float32)
    bio = np.vstack(bio_rows).astype(np.float32)
    matrices = {
        "base": base,
        "vehicle": np.concatenate([base, vehicle], axis=1).astype(np.float32),
        "bio": np.concatenate([base, bio], axis=1).astype(np.float32),
        "vehicle_bio": np.concatenate([base, vehicle, bio], axis=1).astype(np.float32),
    }
    names = {
        "base": scalar_feature_names,
        "vehicle": scalar_feature_names + veh_feature_names,
        "bio": scalar_feature_names + bio_feature_names,
        "vehicle_bio": scalar_feature_names + veh_feature_names + bio_feature_names,
    }
    return pair_meta, matrices, names


def train_pair_model(
    pair_meta: pd.DataFrame,
    x: np.ndarray,
    feature_names: List[str],
    model_name: str,
    bad_weight: bool,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """训练 pairwise RMSE 预测器；预测值越小，候选越优先。"""
    train_mask = pair_meta["split"].astype(str).eq("train").to_numpy()
    y = pd.to_numeric(pair_meta["target_tail_rmse_v241"], errors="coerce").to_numpy(dtype=float)
    good = train_mask & np.isfinite(y)
    xz, med, mean, std = fit_fill_scale(x, train_mask)
    sample_weight = None
    if bad_weight:
        sample_weight = 1.0 + 4.0 * pair_meta["bad_top10"].astype(bool).to_numpy(dtype=float)
        sample_weight = sample_weight[good]

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=420,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.06,
        random_state=SEED,
    )
    model.fit(xz[good], y[good], sample_weight=sample_weight)
    pred = model.predict(xz).astype(float)
    audit = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_names,
            "fill_median": med,
            "scale_mean": mean,
            "scale_std": std,
        }
    )
    return pred, audit


def add_pair_predictions(pair_meta: pd.DataFrame, matrices: Dict[str, np.ndarray], names: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        ("pair_base_hgb", "base", False, "base"),
        ("pair_vehicle_hgb", "vehicle", False, "vehicle_only"),
        ("pair_bio_hgb", "bio", False, "vehicle_bio"),
        ("pair_vehicle_bio_hgb", "vehicle_bio", False, "vehicle_bio"),
        ("pair_vehicle_bio_badweighted_hgb", "vehicle_bio", True, "vehicle_bio"),
    ]
    out = pair_meta.copy()
    audits: List[pd.DataFrame] = []
    blocks: List[Dict[str, object]] = []
    for model_name, block, bad_weight, family in specs:
        pred, audit = train_pair_model(out, matrices[block], names[block], model_name, bad_weight)
        out[f"pred_{model_name}"] = pred
        audits.append(audit)
        blocks.append(
            {
                "model": model_name,
                "family": family,
                "feature_block": block,
                "bad_weight": bool(bad_weight),
                "feature_n": int(matrices[block].shape[1]),
            }
        )
    return out, pd.concat(audits, ignore_index=True), pd.DataFrame(blocks)


def baseline_rows(events: pd.DataFrame, lookup: Dict[str, Dict[int, float]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for _, event in events.iterrows():
        uid = str(event["event_uid"])
        for strategy, family, deployable, delay in [
            ("policy_keep_0ms_anchor", "baseline", True, 0),
            ("policy_wait_to_latest_anchor", "baseline", True, max(lookup[uid].keys())),
        ]:
            mapped_delay, rmse = rmse_at_delay(lookup, uid, int(delay))
            rows.append(selected_row(event, strategy, family, deployable, np.nan, mapped_delay, rmse, 1))
        oracle_delay, oracle_rmse = min(lookup[uid].items(), key=lambda kv: kv[1])
        rows.append(selected_row(event, "oracle_best_anchor_upper_bound", "oracle", False, np.nan, oracle_delay, oracle_rmse, len(lookup[uid])))
    return rows


def selected_row(
    event: pd.Series,
    strategy: str,
    family: str,
    deployable: bool,
    k: float,
    delay: int,
    rmse: float,
    candidate_n: int,
) -> Dict[str, object]:
    keep0 = float(event["keep0_tail_rmse_v241"])
    latest = float(event["latest_tail_rmse_v241"])
    oracle = float(event["oracle_tail_rmse_v241"])
    return {
        "strategy": strategy,
        "strategy_family": family,
        "deployable": bool(deployable),
        "k": k,
        "bio_lambda": np.nan,
        "event_uid": str(event["event_uid"]),
        "split": str(event["split"]),
        "subject": str(event["subject"]),
        "recording": str(event["recording"]),
        "selected_delay_ms": int(delay),
        "selected_tail_rmse_v241": float(rmse),
        "keep0_tail_rmse_v241": keep0,
        "latest_tail_rmse_v241": latest,
        "oracle_tail_rmse_v241": oracle,
        "delta_selected_minus_keep0": float(rmse - keep0),
        "delta_selected_minus_latest": float(rmse - latest),
        "bad_top10": bool(event["bad_top10"]),
        "very_bad_top5": bool(event["very_bad_top5"]),
        "normal": bool(event["normal"]),
        "observe_later_like": bool(event["observe_later_like"]),
        "strong_steer": bool(event["strong_steer"]),
        "reverse": bool(event["reverse"]),
        "early_best_after_400": bool(event["early_best_after_400"]),
        "prototype_unique_delay_n": int(candidate_n),
    }


def build_selected(events: pd.DataFrame, pair_pred: pd.DataFrame, lookup: Dict[str, Dict[int, float]]) -> pd.DataFrame:
    event_map = {str(row["event_uid"]): row for _, row in events.iterrows()}
    rows = baseline_rows(events, lookup)
    pred_cols = [col for col in pair_pred.columns if col.startswith("pred_pair_")]
    family_by_pred = {
        "pred_pair_base_hgb": "base",
        "pred_pair_vehicle_hgb": "vehicle_only",
        "pred_pair_bio_hgb": "vehicle_bio",
        "pred_pair_vehicle_bio_hgb": "vehicle_bio",
        "pred_pair_vehicle_bio_badweighted_hgb": "vehicle_bio",
    }
    for k in K_VALUES:
        sub_k = pair_pred[pair_pred["neighbor_rank_vehicle"].astype(int) <= k].copy()
        for uid, g in sub_k.groupby("event_uid", sort=False):
            event = event_map[str(uid)]
            # candidate oracle 是当前 topK 候选集合的理论上限。
            best = g.loc[pd.to_numeric(g["target_tail_rmse_v241"], errors="coerce").idxmin()]
            rows.append(
                selected_row(
                    event,
                    f"pair_candidate_oracle_k{k}",
                    "candidate_oracle",
                    False,
                    float(k),
                    int(best["mapped_delay_ms"]),
                    float(best["target_tail_rmse_v241"]),
                    int(g["mapped_delay_ms"].nunique()),
                )
            )
            for pred_col in pred_cols:
                chosen = g.loc[pd.to_numeric(g[pred_col], errors="coerce").idxmin()]
                strategy = f"{pred_col.replace('pred_', '')}_k{k}"
                rows.append(
                    selected_row(
                        event,
                        strategy,
                        family_by_pred[pred_col],
                        True,
                        float(k),
                        int(chosen["mapped_delay_ms"]),
                        float(chosen["target_tail_rmse_v241"]),
                        int(g["mapped_delay_ms"].nunique()),
                    )
                )
    return pd.DataFrame(rows)


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    return V266.summarize_selected(selected)


def choose_val_strategies(summary: pd.DataFrame) -> pd.DataFrame:
    """按 val bad_top10 选择可部署 pair reranker；test 不参与选择。"""
    val = summary[
        summary["split"].eq("val")
        & summary["event_group"].eq("bad_top10")
        & summary["deployable"].astype(bool)
    ].copy()
    choices = [
        ("val_best_pair_vehicle", ["vehicle_only"]),
        ("val_best_pair_vehicle_bio", ["vehicle_bio"]),
        ("val_best_pair_any", ["base", "vehicle_only", "vehicle_bio"]),
    ]
    rows: List[Dict[str, object]] = []
    for label, families in choices:
        sub = val[val["strategy_family"].isin(families)].copy()
        if sub.empty:
            continue
        # val 上优先 bad_top10 RMSE，其次 all-test 无法看，因此这里只用 val 的平均 delay 做保守 tie-break。
        sub = sub.sort_values(["selected_tail_rmse_mean", "selected_delay_ms_mean", "strategy"], ascending=[True, True, True])
        best = sub.iloc[0]
        rows.append(
            {
                "chosen_label": label,
                "chosen_strategy": str(best["strategy"]),
                "chosen_family": str(best["strategy_family"]),
                "val_bad_top10_rmse": float(best["selected_tail_rmse_mean"]),
                "val_bad_top10_delay_ms_mean": float(best["selected_delay_ms_mean"]),
            }
        )
    chosen = pd.DataFrame(rows)
    if chosen.empty:
        return chosen
    expanded: List[Dict[str, object]] = []
    for _, choice in chosen.iterrows():
        sub = summary[summary["strategy"].eq(str(choice["chosen_strategy"]))]
        for _, row in sub.iterrows():
            rec = choice.to_dict()
            for col in [
                "split",
                "event_group",
                "n",
                "selected_tail_rmse_mean",
                "delta_selected_minus_keep0_mean",
                "delta_selected_minus_latest_mean",
                "selected_delay_ms_mean",
                "selected_latest_rate",
                "improve_rate_vs_keep0",
            ]:
                rec[col] = row[col]
            expanded.append(rec)
    return pd.DataFrame(expanded)


def compact_focus(summary: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    out = test_bad[
        test_bad["strategy"].isin(
            ["policy_keep_0ms_anchor", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"]
        )
    ].copy()
    for fam in ["candidate_oracle", "vehicle_only", "vehicle_bio", "base"]:
        sub = test_bad[test_bad["strategy_family"].eq(fam)].copy()
        if not sub.empty:
            sub = sub.sort_values(["selected_tail_rmse_mean", "k", "strategy"], ascending=[True, True, True]).head(3)
            out = pd.concat([out, sub], ignore_index=True)
    for label in ["val_best_pair_vehicle", "val_best_pair_vehicle_bio", "val_best_pair_any"]:
        sub = chosen[
            chosen["chosen_label"].eq(label)
            & chosen["split"].eq("test")
            & chosen["event_group"].eq("bad_top10")
        ]
        if len(sub):
            row = sub.iloc[0]
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(
                        [
                            {
                                "strategy": f"{label}: {row['chosen_strategy']}",
                                "strategy_family": row["chosen_family"],
                                "deployable": True,
                                "k": np.nan,
                                "n": row["n"],
                                "selected_tail_rmse_mean": row["selected_tail_rmse_mean"],
                                "keep0_tail_rmse_mean": np.nan,
                                "latest_tail_rmse_mean": np.nan,
                                "oracle_tail_rmse_mean": np.nan,
                                "delta_selected_minus_keep0_mean": row["delta_selected_minus_keep0_mean"],
                                "delta_selected_minus_latest_mean": row["delta_selected_minus_latest_mean"],
                                "improve_rate_vs_keep0": row["improve_rate_vs_keep0"],
                                "selected_delay_ms_mean": row["selected_delay_ms_mean"],
                                "selected_latest_rate": row["selected_latest_rate"],
                                "prototype_unique_delay_n_mean": np.nan,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return out.drop_duplicates(subset=["strategy"], keep="first")


def plot_main(summary: pd.DataFrame, chosen: pd.DataFrame) -> Path:
    path = FIGURES / "v267_test_badtop10_pair_reranker.png"
    focus = compact_focus(summary, chosen)
    keep = [
        "policy_keep_0ms_anchor",
        "policy_wait_to_latest_anchor",
        "oracle_best_anchor_upper_bound",
    ]
    rows: List[Tuple[str, float]] = []
    for strategy in keep:
        sub = focus[focus["strategy"].eq(strategy)]
        if len(sub):
            rows.append((strategy, float(sub["selected_tail_rmse_mean"].iloc[0])))
    for fam_label in ["candidate_oracle", "vehicle_only", "vehicle_bio"]:
        sub = focus[focus["strategy_family"].eq(fam_label)]
        sub = sub[~sub["strategy"].astype(str).str.startswith("val_best")]
        if len(sub):
            row = sub.sort_values("selected_tail_rmse_mean").iloc[0]
            rows.append((f"test-best {fam_label}\n{row['strategy']}", float(row["selected_tail_rmse_mean"])))
    for label in ["val_best_pair_vehicle", "val_best_pair_vehicle_bio", "val_best_pair_any"]:
        sub = focus[focus["strategy"].astype(str).str.startswith(label)]
        if len(sub):
            rows.append((str(sub["strategy"].iloc[0]), float(sub["selected_tail_rmse_mean"].iloc[0])))
    if not rows:
        return path
    labels, vals = zip(*rows)
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=["#9CA3AF", "#E15759", "#B07AA1", "#76B7B2", "#4C78A8", "#F28E2B", "#59A14F", "#EDC948"][: len(vals)])
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed wait-latest 0.6950")
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace("_", "\n") for lab in labels], fontsize=8)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v267: 监督式 query-prototype reranker 是否把 headroom 转成收益")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_val_test_generalization(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v267_val_test_badtop10_generalization.png"
    sub = summary[
        summary["event_group"].eq("bad_top10")
        & summary["deployable"].astype(bool)
        & summary["strategy_family"].isin(["vehicle_only", "vehicle_bio"])
    ].copy()
    if sub.empty:
        return path
    val = sub[sub["split"].eq("val")][["strategy", "selected_tail_rmse_mean"]].rename(columns={"selected_tail_rmse_mean": "val_rmse"})
    test = sub[sub["split"].eq("test")][["strategy", "selected_tail_rmse_mean", "strategy_family"]].rename(columns={"selected_tail_rmse_mean": "test_rmse"})
    joined = val.merge(test, on="strategy", how="inner")
    if joined.empty:
        return path
    fig, ax = plt.subplots(figsize=(7, 5.4))
    colors = joined["strategy_family"].map({"vehicle_only": "#4C78A8", "vehicle_bio": "#F28E2B"}).fillna("#9CA3AF")
    ax.scatter(joined["val_rmse"], joined["test_rmse"], c=colors, alpha=0.75)
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.0)
    ax.set_xlabel("val bad_top10 RMSE")
    ax.set_ylabel("test bad_top10 RMSE")
    ax.set_title("v267: val 选择是否能泛化到 test")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [("v266_script", V266_SCRIPT)]:
        rows.append({"label": label, "path": str(path), "exists": bool(path.exists()), "sha256": file_sha256(path) if path.exists() else ""})
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    write_csv(pd.DataFrame(rows), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def write_report(
    summary: pd.DataFrame,
    chosen: pd.DataFrame,
    feature_block_audit: pd.DataFrame,
    pair_audit: pd.DataFrame,
    figures: List[Path],
) -> None:
    focus = compact_focus(summary, chosen)
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    candidate = test_bad[test_bad["strategy_family"].eq("candidate_oracle")].sort_values("selected_tail_rmse_mean").iloc[0]
    vehicle = chosen[
        chosen["chosen_label"].eq("val_best_pair_vehicle")
        & chosen["split"].eq("test")
        & chosen["event_group"].eq("bad_top10")
    ]
    bio = chosen[
        chosen["chosen_label"].eq("val_best_pair_vehicle_bio")
        & chosen["split"].eq("test")
        & chosen["event_group"].eq("bad_top10")
    ]

    lines: List[str] = []
    lines.append("# v267 supervised bio prototype reranker")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v266 说明 vehicle-matched prototype 候选库有 headroom，但简单距离/投票规则选不准。")
    lines.append("- v267 用 train query-prototype pair 监督训练 reranker，检验更强的可部署候选选择器能否把 headroom 转成 test bad_top10 收益。")
    lines.append("")
    lines.append("## 方法边界")
    lines.append("")
    lines.append("- prototype 只来自 train split。")
    lines.append("- query 只使用 0ms 车辆上下文和 observation_s 之前的 bio260_sp64。")
    lines.append("- 标签是 train query 在 prototype oracle delay 下的真实 tail RMSE；val/test 只用于选择/报告，不参与训练。")
    lines.append("- 生理只参与候选内部 reranking，不直接生成轨迹。")
    lines.append("")
    lines.append("## 特征块")
    lines.append("")
    lines.append(feature_block_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## Pair 构造审计")
    lines.append("")
    lines.append(pair_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## Test bad_top10 关键对照")
    lines.append("")
    cols = [
        "strategy",
        "strategy_family",
        "deployable",
        "n",
        "selected_tail_rmse_mean",
        "delta_selected_minus_keep0_mean",
        "delta_selected_minus_latest_mean",
        "selected_delay_ms_mean",
        "selected_latest_rate",
    ]
    lines.append(focus[[c for c in cols if c in focus.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Val 选择的可部署策略")
    lines.append("")
    if chosen.empty:
        lines.append("- 未能选出可部署策略。")
    else:
        display = chosen[chosen["event_group"].isin(["bad_top10", "all"])][
            [
                "chosen_label",
                "chosen_strategy",
                "chosen_family",
                "split",
                "event_group",
                "n",
                "selected_tail_rmse_mean",
                "delta_selected_minus_latest_mean",
                "selected_delay_ms_mean",
            ]
        ]
        lines.append(display.to_markdown(index=False))
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    lines.append(f"- candidate oracle 最好为 `{float(candidate['selected_tail_rmse_mean']):.4f}`，仍证明候选库 headroom 存在。")
    if len(vehicle) and len(bio):
        veh = float(vehicle["selected_tail_rmse_mean"].iloc[0])
        bio_rmse = float(bio["selected_tail_rmse_mean"].iloc[0])
        lines.append(f"- val-best pair vehicle 在 test bad_top10 为 `{veh:.4f}`。")
        lines.append(f"- val-best pair vehicle+bio 在 test bad_top10 为 `{bio_rmse:.4f}`。")
        if bio_rmse < veh:
            lines.append(f"- 生理监督式 reranker 比 vehicle-only 低 `{veh - bio_rmse:.4f}`。")
        else:
            lines.append(f"- 生理监督式 reranker 比 vehicle-only 高 `{bio_rmse - veh:.4f}`，没有增量。")
        if bio_rmse < FIXED_WAIT_LATEST_BADTOP10:
            lines.append("- vehicle+bio 已低于 fixed wait-latest，可进入更严格的曲线级实现。")
        else:
            lines.append("- vehicle+bio 仍高于 fixed wait-latest，不能算差样本本质改善。")
    lines.append("- 若 val 选择的策略在 test 上不稳定，说明当前 pairwise 监督信号存在 split 泛化问题。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v267_supervised_bio_prototype_reranker_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v267] supervised bio prototype reranker", flush=True)
    clean_out_dir()
    np.random.seed(SEED)

    cand, events, merge_audit, veh_cols, bio_cols = V266.load_candidate_and_events()
    lookup = V266.candidate_rmse_lookup(cand)
    train_event_mask = events["split"].astype(str).eq("train").to_numpy()
    veh_z, _, _, _ = V266.fit_fill_scale(events[veh_cols].to_numpy(dtype=float), train_event_mask)
    bio_z, _, _, _ = V266.fit_fill_scale(events[bio_cols].to_numpy(dtype=float), train_event_mask)
    neighbors = V266.build_neighbor_table(events, veh_z, bio_z, train_event_mask, max_k=max(K_VALUES))
    pair_meta, matrices, names = build_pair_dataset(events, neighbors, lookup, veh_z, bio_z, max_k=max(K_VALUES))
    pair_pred, fill_audit, feature_block_audit = add_pair_predictions(pair_meta, matrices, names)
    selected = build_selected(events, pair_pred, lookup)
    summary = summarize_selected(selected)
    chosen = choose_val_strategies(summary)
    figures = [plot_main(summary, chosen), plot_val_test_generalization(summary)]

    pair_audit = pd.DataFrame(
        [
            {
                "event_n": int(len(events)),
                "pair_n": int(len(pair_pred)),
                "train_pair_n": int(pair_pred["split"].astype(str).eq("train").sum()),
                "val_pair_n": int(pair_pred["split"].astype(str).eq("val").sum()),
                "test_pair_n": int(pair_pred["split"].astype(str).eq("test").sum()),
                "vehicle_feature_n": int(len(veh_cols)),
                "bio260_sp64_feature_n": int(len(bio_cols)),
                "max_k": int(max(K_VALUES)),
                "bio260_uses_post_observation_max": float(merge_audit["bio260_uses_post_observation_max"].iloc[0]),
            }
        ]
    )

    # 输出紧凑 pair 表，完整特征矩阵不落盘，避免产物过大。
    pred_cols = [c for c in pair_pred.columns if c.startswith("pred_pair_")]
    compact_pair_cols = [
        "event_uid",
        "split",
        "subject",
        "prototype_event_uid",
        "prototype_subject",
        "neighbor_rank_vehicle",
        "prototype_oracle_delay_ms",
        "mapped_delay_ms",
        "target_tail_rmse_v241",
        "vehicle_distance",
        "bio_distance",
        "bad_top10",
    ] + pred_cols
    write_csv(pair_pred[compact_pair_cols], TABLES / "v267_pair_predictions_compact.csv")
    write_csv(selected, TABLES / "v267_selected_pair_reranker_by_strategy.csv")
    write_csv(summary, TABLES / "v267_pair_reranker_summary.csv")
    write_csv(chosen, TABLES / "v267_val_chosen_pair_strategy_summary.csv")
    write_csv(feature_block_audit, TABLES / "v267_feature_block_audit.csv")
    write_csv(fill_audit, TABLES / "v267_feature_fill_audit.csv")
    write_csv(pair_audit, TABLES / "v267_pair_construction_audit.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary, chosen, feature_block_audit, pair_audit, figures)
    write_file_inventory()
    zip_ok = make_zip()

    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    candidate = test_bad[test_bad["strategy_family"].eq("candidate_oracle")].sort_values("selected_tail_rmse_mean").iloc[0]
    bio = chosen[
        chosen["chosen_label"].eq("val_best_pair_vehicle_bio")
        & chosen["split"].eq("test")
        & chosen["event_group"].eq("bad_top10")
    ]
    bio_rmse = float(bio["selected_tail_rmse_mean"].iloc[0]) if len(bio) else float("nan")
    guardrail = {
        "pass": bool(zip_ok and float(merge_audit["bio260_uses_post_observation_max"].iloc[0]) == 0.0),
        "zip_testzip": bool(zip_ok),
        "prototype_source_split": "train_only",
        "pair_training_split": "train_only",
        "val_tuned_only": True,
        "test_subject_history_used": False,
        "oracle_deployable": False,
        "bio_role": "supervised_rerank_within_vehicle_matched_prototypes_only",
        "event_n": int(len(events)),
        "pair_n": int(len(pair_pred)),
        "best_candidate_oracle_test_badtop10": float(candidate["selected_tail_rmse_mean"]),
        "val_best_vehicle_bio_test_badtop10": bio_rmse,
        "fixed_wait_latest_badtop10": float(FIXED_WAIT_LATEST_BADTOP10),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v267 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = compact_focus(summary, chosen)
    print(f"[v267] report={REPORTS / 'v267_supervised_bio_prototype_reranker_cn.md'}", flush=True)
    print(f"[v267] zip={ZIP_PATH}", flush=True)
    print(
        focus[
            [
                "strategy",
                "strategy_family",
                "selected_tail_rmse_mean",
                "delta_selected_minus_latest_mean",
                "selected_delay_ms_mean",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
