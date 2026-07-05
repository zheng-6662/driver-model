#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v292 source-physio pairwise candidate ranker.

本轮目标：
- v291 证明：把 ECG/RESP/EDA 合并成 event-level 监督 selector 后，仍无法稳定判断
  什么时候覆盖 latest；
- v292 改成更贴近核心问题的 pairwise 任务：
  对每个 query 的 40 个 vehicle-similar train prototype 候选，比较 query 与 prototype
  的源生理状态是否匹配，学习哪个 prototype 的未来轨迹更接近 query；
- 这一步直接检验“车辆锚点前很像，但后续行为不同”的场景里，生理能否作为 tie-breaker。

边界：
- 候选池固定使用 v278 的 listrank_vehicle 候选，每个 query 40 个 prototype；
- 已核查 v278 prototype 全部来自 train split，val/test query 不会引用 val/test prototype；
- 源生理特征来自 v288/v289/v290，均为 observation_s 前 causal 特征；
- 生理特征筛选沿用 v291 train-only screen；
- pairwise 模型只用 train query 训练，阈值只用 val query 选择，test 只报告。
"""

from __future__ import annotations

import hashlib
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

OUT = BASELINES / "v292_source_physio_pairwise_candidate_ranker_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v292_source_physio_pairwise_candidate_ranker_20260702_pack.zip"

V278_CANDIDATES = (
    BASELINES
    / "v278_listwise_candidate_rank_loss_20260702"
    / "tables"
    / "v278_candidate_listrank_predictions_compact.csv"
)
V291_EVENT_TABLE = (
    BASELINES
    / "v291_multisignal_physio_supervised_probe_20260702"
    / "tables"
    / "v291_multisignal_event_table.csv"
)
V291_SCREEN = (
    BASELINES
    / "v291_multisignal_physio_supervised_probe_20260702"
    / "tables"
    / "v291_train_only_bio_feature_screen.csv"
)
V291_GUARDRAIL = (
    BASELINES
    / "v291_multisignal_physio_supervised_probe_20260702"
    / "logs"
    / "guardrail_check.json"
)

SEED = 29202
FIXED_WAIT_LATEST_BADTOP10 = 0.6950484153471495
# pairwise 表会把 query/prototype/diff 特征展开到 46680 行；
# 这里保守使用 top45，避免高维 DataFrame 在 Windows/Pandas 下触发内存碎片崩溃。
TOP_ALL_N = 45
TOP_LOWID_N = 45

GROUP_FLAGS = {
    "all": None,
    "bad_top10": "bad_top10",
    "vehicle_ambiguous": "vehicle_ambiguous",
    "bad_top10_vehicle_ambiguous": "bad_top10_vehicle_ambiguous",
    "eda_usable": "bio290_eda_event_usable",
}

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


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


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y"]).astype(int)


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    missing = [p for p in [V278_CANDIDATES, V291_EVENT_TABLE, V291_SCREEN] if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少输入：" + "; ".join(str(p) for p in missing))
    cand = pd.read_csv(V278_CANDIDATES, encoding="utf-8-sig", low_memory=False)
    cand = cand[cand["feature_set"].astype(str).eq("listrank_vehicle")].copy()
    events = pd.read_csv(V291_EVENT_TABLE, encoding="utf-8-sig", low_memory=False)
    screen = pd.read_csv(V291_SCREEN, encoding="utf-8-sig", low_memory=False)
    guard = json.loads(V291_GUARDRAIL.read_text(encoding="utf-8")) if V291_GUARDRAIL.exists() else {}
    return cand, events, screen, guard


def choose_bio_features(screen: pd.DataFrame, events: pd.DataFrame) -> Dict[str, List[str]]:
    """沿用 v291 的 train-only 筛选结果，构造 all/low-identity 两套源生理特征。"""

    usable = screen[screen["feature"].isin(events.columns)].copy()
    usable["screen_score_all"] = pd.to_numeric(usable.get("screen_score_all", usable["behavior_corr_max"]), errors="coerce")
    usable["screen_score_lowid"] = pd.to_numeric(usable.get("screen_score_lowid", usable["behavior_corr_max"]), errors="coerce")
    usable["finite_rate_train"] = pd.to_numeric(usable["finite_rate_train"], errors="coerce")
    all_top = (
        usable.sort_values(["screen_score_all", "finite_rate_train"], ascending=[False, False])["feature"]
        .head(TOP_ALL_N)
        .tolist()
    )
    lowid = usable[usable["low_identity_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if len(lowid) < 30:
        lowid = usable.copy()
    lowid_top = (
        lowid.sort_values(["screen_score_lowid", "finite_rate_train"], ascending=[False, False])["feature"]
        .head(TOP_LOWID_N)
        .tolist()
    )
    return {"bio_all_top": all_top, "bio_lowid_top": lowid_top}


def add_candidate_vehicle_features(cand: pd.DataFrame) -> pd.DataFrame:
    """只用推理时可见的 vehicle candidate 分数和延迟构造 pair-level 车辆特征。"""

    out = cand.copy()
    for col in ["mapped_delay_ms", "pred_rank_score", "target_tail_rmse_v241", "latest_tail_rmse_v241", "rank_target_z"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["target_gain_vs_latest"] = out["latest_tail_rmse_v241"] - out["target_tail_rmse_v241"]
    out["vehicle_score_rank_desc"] = out.groupby("event_uid")["pred_rank_score"].rank(ascending=False, method="first")
    out["vehicle_score_rank_pct"] = out.groupby("event_uid")["pred_rank_score"].rank(pct=True, ascending=True)
    out["vehicle_score_best"] = out.groupby("event_uid")["pred_rank_score"].transform("max")
    out["vehicle_score_margin_to_best"] = out["vehicle_score_best"] - out["pred_rank_score"]
    out["vehicle_score_mean"] = out.groupby("event_uid")["pred_rank_score"].transform("mean")
    out["vehicle_score_std"] = out.groupby("event_uid")["pred_rank_score"].transform("std").replace(0, np.nan)
    out["vehicle_score_z_within_event"] = (out["pred_rank_score"] - out["vehicle_score_mean"]) / out["vehicle_score_std"]
    out["candidate_delay_s"] = out["mapped_delay_ms"] / 1000.0
    out["candidate_is_latest_delay"] = out["mapped_delay_ms"].eq(1000).astype(int)
    return out


def make_pair_table(cand: pd.DataFrame, events: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """把 query 与 prototype 的生理特征拼成 pairwise 表。"""

    cand = add_candidate_vehicle_features(cand)
    flags = ["bad_top10", "vehicle_ambiguous", "bad_top10_vehicle_ambiguous", "bio290_eda_event_usable"]
    meta_cols = ["event_uid", "split", "subject", "recording"] + [c for c in flags if c in events.columns]
    for c in flags:
        if c in events.columns:
            events[c] = safe_bool_series(events[c])

    all_bio_cols = sorted(set(sum(feature_sets.values(), [])))
    query_cols = meta_cols + all_bio_cols
    proto_cols = ["event_uid", "split", "subject"] + all_bio_cols

    query = events[query_cols].drop_duplicates("event_uid").copy()
    proto = events[proto_cols].drop_duplicates("event_uid").copy()
    proto = proto.rename(columns={"event_uid": "prototype_event_uid", "split": "prototype_split", "subject": "prototype_subject"})

    pairs = cand.merge(query, on="event_uid", how="left", validate="many_to_one", suffixes=("", "_query"))
    pairs = pairs.merge(
        proto,
        on="prototype_event_uid",
        how="left",
        validate="many_to_one",
        suffixes=("_query", "_proto"),
    )

    audit = {
        "candidate_rows": int(len(pairs)),
        "event_n": int(pairs["event_uid"].nunique()),
        "prototype_missing_n": int(pairs["prototype_split"].isna().sum()),
        "prototype_train_only": bool(pairs["prototype_split"].dropna().astype(str).eq("train").all()),
        "query_split_event_counts": pairs.groupby("split")["event_uid"].nunique().to_dict(),
        "same_subject_pair_rate_by_split": pairs.assign(
            same_subject=pairs["subject"].astype(str).eq(pairs["prototype_subject"].astype(str))
        )
        .groupby("split")["same_subject"]
        .mean()
        .to_dict(),
    }

    pair_feature_cols: Dict[str, pd.Series] = {}
    for f in all_bio_cols:
        # query/prototype 都有同名源生理列，第二次 merge 后通常变成 f_query / f_proto。
        # 少数情况下若 pandas 未加后缀，则回退到原列名。
        q_col = f"{f}_query" if f"{f}_query" in pairs.columns else f
        p_col = f"{f}_proto" if f"{f}_proto" in pairs.columns else f
        q = pd.to_numeric(pairs[q_col], errors="coerce")
        p = pd.to_numeric(pairs[p_col], errors="coerce")
        pair_feature_cols[f"pair_absdiff__{f}"] = (q - p).abs()
        pair_feature_cols[f"pair_signeddiff__{f}"] = q - p
        pair_feature_cols[f"query__{f}"] = q
        pair_feature_cols[f"proto__{f}"] = p

    drop_cols = [c for c in pairs.columns if c.endswith("_query") or c.endswith("_proto")]
    pair_feature_df = pd.DataFrame(pair_feature_cols, index=pairs.index)
    pairs = pd.concat([pairs.drop(columns=drop_cols, errors="ignore"), pair_feature_df], axis=1).copy()
    return pairs, audit


def vehicle_cols() -> List[str]:
    return [
        "mapped_delay_ms",
        "candidate_delay_s",
        "candidate_is_latest_delay",
        "pred_rank_score",
        "vehicle_score_rank_desc",
        "vehicle_score_rank_pct",
        "vehicle_score_margin_to_best",
        "vehicle_score_z_within_event",
    ]


def build_feature_blocks(feature_sets: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """构造车辆、源生理 pair 匹配、车辆+源生理 pair 匹配的对照块。"""

    blocks: Dict[str, List[str]] = {"vehicle_candidate_score": vehicle_cols()}
    for name, feats in feature_sets.items():
        pair_cols = []
        for f in feats:
            pair_cols.extend([f"pair_absdiff__{f}", f"pair_signeddiff__{f}", f"query__{f}", f"proto__{f}"])
        blocks[f"{name}_pair_only"] = pair_cols
        blocks[f"vehicle_plus_{name}_pair"] = vehicle_cols() + pair_cols
    return blocks


def models() -> Dict[str, Pipeline]:
    return {
        "ridge_a10": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "hgb_d3": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=140,
                        learning_rate=0.045,
                        max_leaf_nodes=15,
                        l2_regularization=0.4,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "extra_trees_d5": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=220,
                        max_depth=5,
                        min_samples_leaf=8,
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def fit_predict_pair_gain(pairs: pd.DataFrame, cols: List[str], model: Pipeline) -> np.ndarray:
    train = pairs["split"].astype(str).eq("train")
    X = pairs[cols].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(pairs["target_gain_vs_latest"], errors="coerce")
    model.fit(X.loc[train], y.loc[train])
    return np.asarray(model.predict(X), dtype=float)


def top_candidate_per_event(pairs: pd.DataFrame, pred_gain: np.ndarray, tag: str) -> pd.DataFrame:
    cols = [
        "event_uid",
        "split",
        "subject",
        "recording",
        "prototype_event_uid",
        "prototype_subject",
        "mapped_delay_ms",
        "target_tail_rmse_v241",
        "latest_tail_rmse_v241",
        "target_gain_vs_latest",
        "pred_rank_score",
        "vehicle_score_rank_desc",
        "bad_top10",
        "vehicle_ambiguous",
        "bad_top10_vehicle_ambiguous",
        "bio290_eda_event_usable",
    ]
    top = pairs[[c for c in cols if c in pairs.columns]].copy()
    top["selector_tag"] = tag
    top["pred_gain_vs_latest"] = pred_gain
    top = top.sort_values(
        ["event_uid", "pred_gain_vs_latest", "target_tail_rmse_v241", "vehicle_score_rank_desc"],
        ascending=[True, False, True, True],
    ).drop_duplicates("event_uid")
    top = top.rename(
        columns={
            "target_tail_rmse_v241": "candidate_rmse",
            "latest_tail_rmse_v241": "latest_rmse",
            "mapped_delay_ms": "selected_delay_ms_before_threshold",
        }
    )
    return top


def summarize_selected(selected: pd.DataFrame, split: str, group: str, flag: str | None) -> Dict[str, object]:
    sub = selected[selected["split"].astype(str).eq(split)].copy()
    if flag is not None:
        if flag not in sub.columns:
            sub = sub.iloc[0:0]
        else:
            sub = sub[safe_bool_series(sub[flag]).astype(bool)]
    if sub.empty:
        return {
            "n": 0,
            "latest_rmse_mean": math.nan,
            "selected_rmse_mean": math.nan,
            "candidate_rmse_mean": math.nan,
            "delta_vs_latest_mean": math.nan,
            "override_rate": math.nan,
            "override_n": 0,
            "candidate_beats_latest_rate": math.nan,
        }
    return {
        "n": int(len(sub)),
        "latest_rmse_mean": float(sub["latest_rmse"].mean()),
        "selected_rmse_mean": float(sub["selected_rmse"].mean()),
        "candidate_rmse_mean": float(sub["candidate_rmse"].mean()),
        "delta_vs_latest_mean": float((sub["selected_rmse"] - sub["latest_rmse"]).mean()),
        "override_rate": float(sub["override_latest"].astype(bool).mean()),
        "override_n": int(sub["override_latest"].astype(bool).sum()),
        "candidate_beats_latest_rate": float((sub["candidate_rmse"] < sub["latest_rmse"]).mean()),
    }


def threshold_search(top: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """阈值只由 val 的 pred_gain 构造，inf 表示永不覆盖 latest。"""

    val = pd.to_numeric(top.loc[top["split"].astype(str).eq("val"), "pred_gain_vs_latest"], errors="coerce")
    positive = val[np.isfinite(val) & (val > 0)]
    thresholds = [float("inf")]
    if len(positive):
        thresholds += [float(x) for x in np.unique(np.nanquantile(positive, np.linspace(0.05, 0.95, 19)))]
        thresholds.append(0.0)
    thresholds = sorted(set(thresholds), key=lambda x: (math.isinf(x), x))

    all_selected: List[pd.DataFrame] = []
    rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        selected = top.copy()
        selected["threshold"] = threshold
        selected["override_latest"] = pd.to_numeric(selected["pred_gain_vs_latest"], errors="coerce") >= threshold
        selected.loc[np.isinf(threshold), "override_latest"] = False
        selected["selected_rmse"] = np.where(selected["override_latest"], selected["candidate_rmse"], selected["latest_rmse"])
        selected["selected_delay_ms"] = np.where(selected["override_latest"], selected["selected_delay_ms_before_threshold"], 1000)
        all_selected.append(selected)

        row: Dict[str, object] = {"selector_tag": str(top["selector_tag"].iloc[0]), "threshold": threshold}
        for split in ["train", "val", "test"]:
            for group, flag in GROUP_FLAGS.items():
                m = summarize_selected(selected, split, group, flag)
                prefix = f"{split}_{group}"
                for key, value in m.items():
                    row[f"{prefix}_{key}"] = value
        row["active_val_bad_top10"] = int(row.get("val_bad_top10_override_n", 0) or 0) > 0
        row["noharm_val"] = (
            float(row.get("val_bad_top10_delta_vs_latest_mean", math.inf)) <= 0.0
            and float(row.get("val_all_delta_vs_latest_mean", math.inf)) <= 0.003
            and float(row.get("val_vehicle_ambiguous_delta_vs_latest_mean", math.inf)) <= 0.005
        )
        row["selection_score"] = float(row.get("val_bad_top10_delta_vs_latest_mean", 0.0)) + 5.0 * max(
            0.0, float(row.get("val_all_delta_vs_latest_mean", 0.0))
        )
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(all_selected, ignore_index=True)


def run_pair_rankers(pairs: pd.DataFrame, blocks: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: List[pd.DataFrame] = []
    selected_events: List[pd.DataFrame] = []
    feature_rows: List[Dict[str, object]] = []
    for block_name, cols0 in blocks.items():
        cols = [c for c in cols0 if c in pairs.columns]
        if not cols:
            continue
        for model_name, model in models().items():
            tag = f"{block_name}__{model_name}"
            print(f"[v292] ranker {tag} feature_n={len(cols)}")
            pred = fit_predict_pair_gain(pairs, cols, model)
            top = top_candidate_per_event(pairs, pred, tag)
            summary, per_event = threshold_search(top)
            summary["feature_block"] = block_name
            summary["model_name"] = model_name
            summary["feature_n"] = len(cols)
            per_event["feature_block"] = block_name
            per_event["model_name"] = model_name
            summaries.append(summary)
            selected_events.append(per_event)
            feature_rows.append(
                {
                    "feature_block": block_name,
                    "model_name": model_name,
                    "selector_tag": tag,
                    "feature_n": len(cols),
                    "features": json.dumps(cols, ensure_ascii=False),
                }
            )
    return pd.concat(summaries, ignore_index=True), pd.concat(selected_events, ignore_index=True), pd.DataFrame(feature_rows)


def oracle_pair_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    top_oracle = pairs.sort_values(["event_uid", "target_tail_rmse_v241"]).drop_duplicates("event_uid").copy()
    top_vehicle = pairs.sort_values(["event_uid", "pred_rank_score", "target_tail_rmse_v241"], ascending=[True, False, True]).drop_duplicates("event_uid").copy()
    for name, top in [("vehicle_score_top1_no_threshold", top_vehicle), ("oracle_best_candidate", top_oracle)]:
        selected = top.rename(columns={"target_tail_rmse_v241": "candidate_rmse", "latest_tail_rmse_v241": "latest_rmse"}).copy()
        selected["override_latest"] = True
        selected["selected_rmse"] = selected["candidate_rmse"]
        for split in ["train", "val", "test"]:
            for group, flag in GROUP_FLAGS.items():
                m = summarize_selected(selected, split, group, flag)
                rows.append({"policy": name, "split": split, "event_group": group, **m})
    return pd.DataFrame(rows)


def choose_deployable(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    active = summary[summary["active_val_bad_top10"].astype(bool) & summary["noharm_val"].astype(bool)].copy()
    if len(active):
        chosen = active.sort_values(["selection_score", "val_bad_top10_delta_vs_latest_mean"]).iloc[0]
        rows.append({"chosen_type": "best_val_noharm_active", **chosen.to_dict()})
    else:
        fallback = summary[summary["threshold"].map(math.isinf)].copy()
        if len(fallback):
            chosen = fallback.sort_values("val_bad_top10_delta_vs_latest_mean").iloc[0]
            rows.append({"chosen_type": "fallback_no_override", **chosen.to_dict()})
    diag = summary[pd.to_numeric(summary["test_bad_top10_override_n"], errors="coerce").fillna(0) > 0].copy()
    if len(diag):
        chosen = diag.sort_values(["test_bad_top10_delta_vs_latest_mean", "val_bad_top10_delta_vs_latest_mean"]).iloc[0]
        rows.append({"chosen_type": "test_best_diagnostic_not_deployable", **chosen.to_dict()})
    return pd.DataFrame(rows)


def route_decision(chosen: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    dep = chosen[chosen["chosen_type"].astype(str).eq("best_val_noharm_active")].copy() if len(chosen) else pd.DataFrame()
    if len(dep):
        row = dep.iloc[0]
        dep_bad = float(row.get("test_bad_top10_delta_vs_latest_mean", math.inf))
        dep_amb = float(row.get("test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean", math.inf))
        dep_override = float(row.get("test_bad_top10_override_rate", 0.0))
    else:
        dep_bad = math.inf
        dep_amb = math.inf
        dep_override = 0.0
    oracle_bad = oracle[
        oracle["policy"].eq("oracle_best_candidate")
        & oracle["split"].eq("test")
        & oracle["event_group"].eq("bad_top10")
    ]
    oracle_delta = float(oracle_bad["delta_vs_latest_mean"].iloc[0]) if len(oracle_bad) else math.nan
    rows = [
        {
            "check": "deployable_pairwise_selector_beats_latest_bad_top10",
            "requirement": "val no-harm active pairwise selector 在 test bad_top10 上低于 latest",
            "pass": bool(dep_bad < -1e-9 and dep_override > 0),
            "evidence": dep_bad if np.isfinite(dep_bad) else None,
            "deployable": True,
        },
        {
            "check": "deployable_pairwise_selector_beats_latest_bad_ambiguous",
            "requirement": "同一 pairwise selector 在 test bad_top10_vehicle_ambiguous 上低于 latest",
            "pass": bool(dep_amb < -1e-9 and dep_override > 0),
            "evidence": dep_amb if np.isfinite(dep_amb) else None,
            "deployable": True,
        },
        {
            "check": "candidate_pool_oracle_has_headroom",
            "requirement": "vehicle top40 candidate pool 在 test bad_top10 上有至少 0.05 RMSE oracle 空间",
            "pass": bool(np.isfinite(oracle_delta) and oracle_delta <= -0.05),
            "evidence": oracle_delta,
            "deployable": False,
        },
    ]
    decision = pd.DataFrame(rows)
    decision["route_viable_now"] = bool(
        decision.loc[
            decision["check"].isin(
                [
                    "deployable_pairwise_selector_beats_latest_bad_top10",
                    "deployable_pairwise_selector_beats_latest_bad_ambiguous",
                ]
            ),
            "pass",
        ].all()
    )
    return decision


def plot_oracle(oracle: pd.DataFrame) -> Path:
    data = oracle[oracle["split"].eq("test") & oracle["event_group"].isin(["bad_top10", "bad_top10_vehicle_ambiguous"])].copy()
    data["label"] = data["event_group"] + " | " + data["policy"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(data["label"], data["delta_vs_latest_mean"], color="#4E79A7")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("delta vs latest RMSE")
    ax.set_title("v292 vehicle-top40 candidate-pool headroom")
    fig.tight_layout()
    path = FIGURES / "v292_candidate_pool_oracle_delta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_chosen(chosen: pd.DataFrame) -> Path:
    data = chosen.copy() if len(chosen) else pd.DataFrame({"chosen_type": ["none"], "test_bad_top10_delta_vs_latest_mean": [math.nan]})
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh(data["chosen_type"].astype(str), pd.to_numeric(data["test_bad_top10_delta_vs_latest_mean"], errors="coerce"), color="#F28E2B")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("test bad_top10 delta vs latest RMSE")
    ax.set_title("v292 validation-chosen pairwise selector")
    fig.tight_layout()
    path = FIGURES / "v292_chosen_selector_badtop10_delta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_top_diagnostic(summary: pd.DataFrame) -> Path:
    data = summary[pd.to_numeric(summary["test_bad_top10_override_n"], errors="coerce").fillna(0) > 0].copy()
    data = data.sort_values("test_bad_top10_delta_vs_latest_mean").head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = data["selector_tag"].astype(str).str.replace("__", "\n", regex=False)
    ax.barh(labels, data["test_bad_top10_delta_vs_latest_mean"], color="#59A14F")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("test bad_top10 delta vs latest RMSE")
    ax.set_title("v292 diagnostic top pairwise selectors")
    fig.tight_layout()
    path = FIGURES / "v292_diagnostic_top_selectors.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, cols: List[str], n: int | None = None) -> str:
    if df is None or df.empty:
        return "- 无记录。"
    view = df[[c for c in cols if c in df.columns]].copy()
    if n is not None:
        view = view.head(n)
    return view.to_markdown(index=False)


def write_report(
    decision: pd.DataFrame,
    chosen: pd.DataFrame,
    summary: pd.DataFrame,
    oracle: pd.DataFrame,
    feature_audit: pd.DataFrame,
    input_audit: Dict[str, object],
    guardrail: Dict[str, object],
) -> Path:
    path = REPORTS / "v292_source_physio_pairwise_candidate_ranker_cn.md"
    lines: List[str] = []
    lines.append("# v292 source-physio pairwise candidate ranker")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- v291 的 event-level 多信号监督 selector 没有过关。")
    lines.append("- v292 改成 pairwise candidate ranking：在每个 query 的 40 个 vehicle-similar train prototype 之间，用源生理匹配程度做 tie-breaker。")
    lines.append("- 这一步直接检验“车辆锚点前相似但未来分歧”的核心假设。")
    lines.append("")
    lines.append("## route decision")
    lines.append("")
    lines.append(markdown_table(decision, ["check", "requirement", "pass", "evidence", "deployable", "route_viable_now"]))
    lines.append("")
    lines.append("## 候选池 oracle / vehicle top1 边界")
    lines.append("")
    view_oracle = oracle[oracle["split"].eq("test") & oracle["event_group"].isin(["bad_top10", "bad_top10_vehicle_ambiguous", "all"])].copy()
    lines.append(markdown_table(view_oracle, ["policy", "event_group", "n", "latest_rmse_mean", "selected_rmse_mean", "delta_vs_latest_mean", "candidate_beats_latest_rate"], 80))
    lines.append("")
    lines.append("## validation 选择出的 selector")
    lines.append("")
    lines.append(
        markdown_table(
            chosen,
            [
                "chosen_type",
                "selector_tag",
                "threshold",
                "feature_block",
                "model_name",
                "feature_n",
                "val_bad_top10_delta_vs_latest_mean",
                "val_all_delta_vs_latest_mean",
                "test_bad_top10_delta_vs_latest_mean",
                "test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean",
                "test_bad_top10_override_rate",
            ],
        )
    )
    lines.append("")
    lines.append("## test diagnostic top selector")
    lines.append("")
    diag = summary[pd.to_numeric(summary["test_bad_top10_override_n"], errors="coerce").fillna(0) > 0].copy()
    diag = diag.sort_values(["test_bad_top10_delta_vs_latest_mean", "val_bad_top10_delta_vs_latest_mean"]).head(24)
    lines.append(
        markdown_table(
            diag,
            [
                "selector_tag",
                "threshold",
                "feature_block",
                "model_name",
                "feature_n",
                "val_bad_top10_delta_vs_latest_mean",
                "val_all_delta_vs_latest_mean",
                "test_bad_top10_delta_vs_latest_mean",
                "test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean",
                "test_bad_top10_override_rate",
                "noharm_val",
            ],
        )
    )
    lines.append("")
    lines.append("## feature block")
    lines.append("")
    lines.append(markdown_table(feature_audit, ["feature_block", "model_name", "feature_n"], 30))
    lines.append("")
    lines.append("## input audit")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(input_audit, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    if bool(decision["route_viable_now"].iloc[0]):
        lines.append("- v292 找到了 validation no-harm 后仍能改善 test 差样本的 pairwise 源生理候选排序路线，需要继续做样本级复核。")
    else:
        lines.append("- v292 没有找到可部署 pairwise 源生理候选排序路线。")
        lines.append("- 如果候选池 oracle 很好但 selector 过不了，说明问题不在候选池没有好未来，而在源生理无法稳定识别哪个 prototype 才是对的。")
        lines.append("- 这一步比 event-level selector 更贴近“车辆相似但未来分歧”的假设，因此失败会进一步削弱继续堆生理匹配模型的理由。")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_input_hashes() -> None:
    rows = []
    for name, path in {
        "v278_candidates": V278_CANDIDATES,
        "v291_event_table": V291_EVENT_TABLE,
        "v291_screen": V291_SCREEN,
        "v291_guardrail": V291_GUARDRAIL,
    }.items():
        rows.append({"name": name, "path": str(path), "exists": path.exists(), "sha256": file_sha256(path) if path.exists() else None})
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in OUT.rglob("*"):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows).sort_values("relative_path"), LOGS / "file_inventory.csv")


def make_zip() -> bool:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in OUT.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(OUT.parent))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        return zf.testzip() is None


def main() -> None:
    print("[v292] 目的：在 vehicle-similar train prototype 候选中测试源生理 pairwise tie-breaker。")
    clean_out_dir()
    cand, events, screen, prev_guard = load_inputs()
    feature_sets = choose_bio_features(screen, events)
    pairs, input_audit = make_pair_table(cand, events, feature_sets)
    blocks = build_feature_blocks(feature_sets)
    oracle = oracle_pair_summary(pairs)
    summary, selected, feature_audit = run_pair_rankers(pairs, blocks)
    chosen = choose_deployable(summary)
    decision = route_decision(chosen, oracle)

    write_csv(pairs, TABLES / "v292_pairwise_candidate_table.csv")
    write_csv(pd.DataFrame([{"feature_set": k, "feature_n": len(v), "features": json.dumps(v, ensure_ascii=False)} for k, v in feature_sets.items()]), TABLES / "v292_bio_feature_sets.csv")
    write_csv(feature_audit, TABLES / "v292_feature_block_audit.csv")
    write_csv(oracle, TABLES / "v292_candidate_pool_oracle_summary.csv")
    write_csv(summary, TABLES / "v292_pairwise_threshold_summary.csv")
    write_csv(selected, TABLES / "v292_pairwise_selected_per_event_thresholds.csv")
    write_csv(chosen, TABLES / "v292_pairwise_chosen_by_val.csv")
    write_csv(decision, TABLES / "v292_route_decision.csv")

    plot_oracle(oracle)
    plot_chosen(chosen)
    plot_top_diagnostic(summary)

    guardrail = {
        "pass": True,
        "event_n": int(pairs["event_uid"].nunique()),
        "candidate_rows": int(len(pairs)),
        "train_event_n": int(pairs[pairs["split"].astype(str).eq("train")]["event_uid"].nunique()),
        "val_event_n": int(pairs[pairs["split"].astype(str).eq("val")]["event_uid"].nunique()),
        "test_event_n": int(pairs[pairs["split"].astype(str).eq("test")]["event_uid"].nunique()),
        "prototype_train_only": bool(input_audit["prototype_train_only"]),
        "bio_all_feature_n": int(len(feature_sets["bio_all_top"])),
        "bio_lowid_feature_n": int(len(feature_sets["bio_lowid_top"])),
        "selector_config_n": int(summary["selector_tag"].nunique()),
        "route_viable_now": bool(decision["route_viable_now"].iloc[0]),
        "candidate_pool_test_badtop10_oracle_delta": float(
            oracle[
                oracle["policy"].eq("oracle_best_candidate")
                & oracle["split"].eq("test")
                & oracle["event_group"].eq("bad_top10")
            ]["delta_vs_latest_mean"].iloc[0]
        ),
        "vehicle_score_top1_test_badtop10_delta": float(
            oracle[
                oracle["policy"].eq("vehicle_score_top1_no_threshold")
                & oracle["split"].eq("test")
                & oracle["event_group"].eq("bad_top10")
            ]["delta_vs_latest_mean"].iloc[0]
        ),
        "best_val_noharm_active_exists": bool(chosen["chosen_type"].astype(str).eq("best_val_noharm_active").any()) if len(chosen) else False,
        "best_deployable_test_badtop10_delta": None,
        "best_test_diagnostic_badtop10_delta": None,
        "test_used_for_feature_screen_or_threshold": False,
        "v291_route_viable_now": bool(prev_guard.get("route_viable_now", False)),
    }
    if len(chosen):
        dep = chosen[chosen["chosen_type"].astype(str).eq("best_val_noharm_active")]
        if len(dep):
            guardrail["best_deployable_test_badtop10_delta"] = float(dep["test_bad_top10_delta_vs_latest_mean"].iloc[0])
        diag = chosen[chosen["chosen_type"].astype(str).eq("test_best_diagnostic_not_deployable")]
        if len(diag):
            guardrail["best_test_diagnostic_badtop10_delta"] = float(diag["test_bad_top10_delta_vs_latest_mean"].iloc[0])

    write_report(decision, chosen, summary, oracle, feature_audit, input_audit, guardrail)
    write_input_hashes()
    guardrail["zip_testzip"] = False
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()
    guardrail["zip_testzip"] = bool(make_zip())
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()
    guardrail["zip_testzip"] = bool(make_zip())
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    write_file_inventory()

    print(f"[v292] report={REPORTS / 'v292_source_physio_pairwise_candidate_ranker_cn.md'}")
    print(f"[v292] zip={ZIP_PATH}")
    print(json.dumps(guardrail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
