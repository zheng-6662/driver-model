#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v266 vehicle-matched bio residual prototype。

本轮承接 GPTPro phase02 的路线 2，但先做成一个严格的最小可证伪实验：

    在车辆历史非常相似的事件里，train 事件的“最佳残差/最佳锚点模式”是否能作为
    test 事件的少量候选？如果这个候选库本身都没有超过 fixed wait-latest 的 headroom，
    就没有必要继续训练更复杂的 bio reranker。

关键边界：
- prototype 只来自 train split，不使用 val/test 驾驶员历史；
- query 事件只用 0ms 时刻已经可见的车辆上下文和不晚于 0ms 的 bio260 floor 特征；
- 生理不直接生成轨迹，也不直接参与全局 anchor selector；
- 生理只在“车辆相似 topK prototype 候选”内部做重排序；
- oracle 只作为 headroom，不作为可部署策略；
- 超参数 K/lambda 只根据 val bad_top10 选择，test 只做一次性报告。
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


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V261_SCRIPT = BASELINES / "scripts" / "stage03_v261_bio260_anchor_selector_20260702.py"
V262_FEATURE_SELECTION = (
    BASELINES
    / "v262_subject_invariant_bio260_selector_20260702"
    / "tables"
    / "v262_feature_selection_audit.csv"
)
GPTPRO_RESPONSE = REBUILD / "gptpro_reviews" / "20260702_phase02_response.md"

OUT = BASELINES / "v266_vehicle_matched_bio_residual_prototype_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v266_vehicle_matched_bio_residual_prototype_20260702_pack.zip"

SEED = 26602
K_VALUES = [3, 5, 10, 20, 40]
BIO_LAMBDAS = [0.05, 0.10, 0.20, 0.50, 1.00, 2.00]
FIXED_WAIT_LATEST_BADTOP10 = 0.695048

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_v261_module():
    """复用 v261 已经验证过的候选锚点与 bio260 合并逻辑，避免另造数据入口。"""
    if not V261_SCRIPT.exists():
        raise FileNotFoundError(f"缺少 v261 脚本：{V261_SCRIPT}")
    spec = importlib.util.spec_from_file_location("v261_bio260_anchor_selector", V261_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 v261 脚本：{V261_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V261 = load_v261_module()
V258 = V261.V258


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


def finite_nanmedian(x: np.ndarray, axis: int = 0) -> np.ndarray:
    with np.errstate(all="ignore"):
        med = np.nanmedian(x, axis=axis)
    med = np.asarray(med, dtype=float)
    med[~np.isfinite(med)] = 0.0
    return med


def fit_fill_scale(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只用 train 拟合缺失填充值与标准化参数，然后应用到所有 split。"""
    train_x = x[train_mask]
    med = finite_nanmedian(train_x, axis=0)
    filled = np.where(np.isfinite(x), x, med[None, :])
    mean = np.nanmean(filled[train_mask], axis=0)
    std = np.nanstd(filled[train_mask], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    z = (filled - mean[None, :]) / std[None, :]
    return z.astype(np.float32), med.astype(float), mean.astype(float), std.astype(float)


def load_sp64_bio_columns(df: pd.DataFrame) -> List[str]:
    """优先使用 v262 选出的 subject-invariant sp64 bio 特征，降低 subject/recording 混淆。"""
    if not V262_FEATURE_SELECTION.exists():
        return []
    fs = pd.read_csv(V262_FEATURE_SELECTION, encoding="utf-8-sig", low_memory=False)
    cols = fs[
        fs["row_type"].astype(str).eq("feature")
        & fs["in_sp64"].astype(str).str.lower().eq("true")
    ]["column"].dropna().astype(str).tolist()
    out = [col for col in cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if "bio260_floor_status_ok" in df.columns:
        out.append("bio260_floor_status_ok")
    return list(dict.fromkeys(out))


def event_vehicle_columns(vehicle_cols: Iterable[str]) -> List[str]:
    """prototype 检索只看 0ms 已有车辆上下文，不把候选 delay 当成检索特征。"""
    banned = {
        "candidate_delay_ms",
        "candidate_delay_s",
        "phase_0_delay_s",
        "phase_1_delay_norm_0_to_1",
    }
    return [col for col in vehicle_cols if col not in banned]


def load_candidate_and_events() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """读取 v261 候选表，并压成每个 event 的 0ms 查询上下文与 oracle delay。"""
    cand, merge_audit = V261.load_augmented_table()
    vehicle_cols, _bio_cols_all = V261.feature_columns(cand)
    veh_cols = event_vehicle_columns(vehicle_cols)
    bio_cols = load_sp64_bio_columns(cand)
    if not veh_cols:
        raise RuntimeError("未找到车辆上下文特征，无法做 vehicle-matched prototype。")
    if not bio_cols:
        raise RuntimeError("未找到 sp64 bio260 特征，无法做 bio reranking。")

    cand = cand.copy()
    cand["candidate_delay_ms"] = pd.to_numeric(cand["candidate_delay_ms"], errors="coerce").astype(int)
    cand["candidate_tail_rmse_v241"] = pd.to_numeric(cand["candidate_tail_rmse_v241"], errors="coerce")
    cand = cand[np.isfinite(cand["candidate_tail_rmse_v241"])].copy()

    event_rows: List[Dict[str, object]] = []
    for event_uid, g in cand.groupby("event_uid", sort=False):
        g = g.sort_values("candidate_delay_ms").copy()
        keep0 = g.iloc[0]
        latest = g.iloc[-1]
        oracle = g.loc[g["candidate_tail_rmse_v241"].idxmin()]
        row: Dict[str, object] = {
            "event_uid": str(event_uid),
            "split": str(keep0["split"]),
            "subject": str(keep0.get("subject", "")),
            "recording": str(keep0.get("recording", "")),
            "keep0_delay_ms": int(keep0["candidate_delay_ms"]),
            "latest_delay_ms": int(latest["candidate_delay_ms"]),
            "oracle_delay_ms": int(oracle["candidate_delay_ms"]),
            "keep0_tail_rmse_v241": float(keep0["candidate_tail_rmse_v241"]),
            "latest_tail_rmse_v241": float(latest["candidate_tail_rmse_v241"]),
            "oracle_tail_rmse_v241": float(oracle["candidate_tail_rmse_v241"]),
            "bad_top10": bool(keep0.get("bad_top10_split_v241", False)),
            "very_bad_top5": bool(keep0.get("very_bad_top5_split_v241", False)),
            "normal": bool(keep0.get("normal_curve_current0", False)),
            "observe_later_like": bool(keep0.get("observe_later_like_current0", False)),
            "strong_steer": bool(keep0.get("strong_steer_current0", False)),
            "reverse": bool(keep0.get("reverse_current0", False)),
        }
        for col in veh_cols + bio_cols:
            row[col] = keep0.get(col, np.nan)
        event_rows.append(row)

    events = pd.DataFrame(event_rows)
    events["early_best_after_400"] = pd.to_numeric(events["oracle_delay_ms"], errors="coerce") >= 400
    return cand, events, merge_audit, veh_cols, bio_cols


def candidate_rmse_lookup(cand: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    lookup: Dict[str, Dict[int, float]] = {}
    for event_uid, g in cand.groupby("event_uid", sort=False):
        sub: Dict[int, float] = {}
        for _, row in g.iterrows():
            sub[int(row["candidate_delay_ms"])] = float(row["candidate_tail_rmse_v241"])
        lookup[str(event_uid)] = sub
    return lookup


def nearest_available_delay(delay: int, available: Iterable[int]) -> int:
    arr = np.asarray(list(available), dtype=int)
    if len(arr) == 0:
        raise ValueError("候选 delay 为空。")
    return int(arr[np.argmin(np.abs(arr - int(delay)))])


def rmse_at_delay(lookup: Dict[str, Dict[int, float]], event_uid: str, delay: int) -> Tuple[int, float]:
    sub = lookup[event_uid]
    chosen_delay = int(delay) if int(delay) in sub else nearest_available_delay(int(delay), sub.keys())
    return chosen_delay, float(sub[chosen_delay])


def pairwise_mean_sq_distance(query: np.ndarray, train: np.ndarray) -> np.ndarray:
    """按特征数归一的欧氏距离，便于 vehicle 与 bio distance 做 lambda 组合。"""
    if train.shape[1] == 0:
        return np.zeros(train.shape[0], dtype=np.float32)
    diff = train - query[None, :]
    return np.mean(diff * diff, axis=1)


def build_neighbor_table(
    events: pd.DataFrame,
    veh_z: np.ndarray,
    bio_z: np.ndarray,
    train_mask: np.ndarray,
    max_k: int,
) -> pd.DataFrame:
    """为每个 event 找 train prototype 邻居；val/test 永远只看 train prototype。"""
    train_indices = np.flatnonzero(train_mask)
    rows: List[Dict[str, object]] = []
    event_uid = events["event_uid"].astype(str).to_numpy()
    split = events["split"].astype(str).to_numpy()
    subject = events["subject"].astype(str).to_numpy()
    train_veh = veh_z[train_indices]
    train_bio = bio_z[train_indices]
    train_delay = pd.to_numeric(events.iloc[train_indices]["oracle_delay_ms"], errors="coerce").to_numpy(dtype=int)
    train_uid = event_uid[train_indices]
    train_subject = subject[train_indices]

    for qi in range(len(events)):
        d_vehicle = pairwise_mean_sq_distance(veh_z[qi], train_veh)
        d_bio = pairwise_mean_sq_distance(bio_z[qi], train_bio)
        if train_mask[qi]:
            # train split 只用于诊断；为了避免同一事件复制自身 oracle，排除同 event。
            same = train_uid == event_uid[qi]
            d_vehicle = d_vehicle.copy()
            d_vehicle[same] = np.inf
        order = np.argsort(d_vehicle, kind="mergesort")
        order = order[np.isfinite(d_vehicle[order])][:max_k]
        for rank, pos in enumerate(order, start=1):
            ti = int(train_indices[pos])
            rows.append(
                {
                    "event_uid": event_uid[qi],
                    "split": split[qi],
                    "subject": subject[qi],
                    "neighbor_rank_vehicle": int(rank),
                    "prototype_event_uid": event_uid[ti],
                    "prototype_subject": subject[ti],
                    "prototype_oracle_delay_ms": int(train_delay[pos]),
                    "vehicle_distance": float(d_vehicle[pos]),
                    "bio_distance": float(d_bio[pos]),
                    "same_subject_as_prototype": bool(subject[qi] == train_subject[pos]),
                }
            )
    return pd.DataFrame(rows)


def weighted_vote_delay(nei: pd.DataFrame, distance_col: str, k: int) -> int:
    sub = nei.nsmallest(k, "neighbor_rank_vehicle").copy()
    if sub.empty:
        return 0
    dist = pd.to_numeric(sub[distance_col], errors="coerce").to_numpy(dtype=float)
    dist = np.where(np.isfinite(dist), dist, np.nanmax(dist[np.isfinite(dist)]) if np.isfinite(dist).any() else 0.0)
    weight = 1.0 / (1e-6 + dist)
    sub = sub.assign(_weight=weight)
    vote = sub.groupby("prototype_oracle_delay_ms", as_index=False)["_weight"].sum()
    vote = vote.sort_values(["_weight", "prototype_oracle_delay_ms"], ascending=[False, False])
    return int(vote.iloc[0]["prototype_oracle_delay_ms"])


def select_strategy_for_event(
    strategy: str,
    event_uid: str,
    nei: pd.DataFrame,
    lookup: Dict[str, Dict[int, float]],
    k: int | None = None,
    lam: float | None = None,
) -> Tuple[int, float, int]:
    """返回某个策略在单个 query event 上选择的 delay、RMSE、候选 delay 数量。"""
    if strategy == "policy_keep_0ms_anchor":
        delay, rmse = rmse_at_delay(lookup, event_uid, 0)
        return delay, rmse, 1
    if strategy == "policy_wait_to_latest_anchor":
        latest_delay = max(lookup[event_uid].keys())
        delay, rmse = rmse_at_delay(lookup, event_uid, latest_delay)
        return delay, rmse, 1
    if strategy == "oracle_best_anchor_upper_bound":
        delay, rmse = min(lookup[event_uid].items(), key=lambda kv: kv[1])
        return int(delay), float(rmse), len(lookup[event_uid])

    if k is None:
        raise ValueError(f"策略 {strategy} 需要 k。")
    sub = nei.nsmallest(k, "neighbor_rank_vehicle").copy()
    if sub.empty:
        delay, rmse = rmse_at_delay(lookup, event_uid, 0)
        return delay, rmse, 1

    unique_delay_n = int(sub["prototype_oracle_delay_ms"].nunique())
    if strategy.startswith("prototype_candidate_oracle"):
        best_delay = None
        best_rmse = math.inf
        for delay in sub["prototype_oracle_delay_ms"].astype(int).tolist():
            mapped_delay, rmse = rmse_at_delay(lookup, event_uid, int(delay))
            if rmse < best_rmse:
                best_delay = mapped_delay
                best_rmse = rmse
        return int(best_delay), float(best_rmse), unique_delay_n

    if strategy == "prototype_vehicle_nearest":
        delay = int(sub.iloc[0]["prototype_oracle_delay_ms"])
        mapped_delay, rmse = rmse_at_delay(lookup, event_uid, delay)
        return mapped_delay, rmse, unique_delay_n

    if strategy.startswith("prototype_vehicle_vote"):
        delay = weighted_vote_delay(sub, "vehicle_distance", k)
        mapped_delay, rmse = rmse_at_delay(lookup, event_uid, delay)
        return mapped_delay, rmse, unique_delay_n

    if strategy.startswith("prototype_bio_closest"):
        chosen = sub.sort_values(["bio_distance", "vehicle_distance"], ascending=[True, True]).iloc[0]
        mapped_delay, rmse = rmse_at_delay(lookup, event_uid, int(chosen["prototype_oracle_delay_ms"]))
        return mapped_delay, rmse, unique_delay_n

    if strategy.startswith("prototype_vehicle_bio"):
        if lam is None:
            raise ValueError(f"策略 {strategy} 需要 lambda。")
        score = pd.to_numeric(sub["vehicle_distance"], errors="coerce").to_numpy(dtype=float) + float(lam) * pd.to_numeric(
            sub["bio_distance"], errors="coerce"
        ).to_numpy(dtype=float)
        chosen = sub.iloc[int(np.nanargmin(score))]
        mapped_delay, rmse = rmse_at_delay(lookup, event_uid, int(chosen["prototype_oracle_delay_ms"]))
        return mapped_delay, rmse, unique_delay_n

    raise ValueError(f"未知策略：{strategy}")


def build_selected(
    events: pd.DataFrame,
    neighbors: pd.DataFrame,
    lookup: Dict[str, Dict[int, float]],
) -> pd.DataFrame:
    """生成所有 baseline、headroom oracle 和可部署 prototype 策略的逐事件选择结果。"""
    neighbor_groups = {uid: g for uid, g in neighbors.groupby("event_uid", sort=False)}
    rows: List[Dict[str, object]] = []

    strategy_specs: List[Tuple[str, int | None, float | None, bool, str]] = [
        ("policy_keep_0ms_anchor", None, None, True, "baseline"),
        ("policy_wait_to_latest_anchor", None, None, True, "baseline"),
        ("oracle_best_anchor_upper_bound", None, None, False, "oracle"),
        ("prototype_vehicle_nearest", 1, None, True, "vehicle_only"),
    ]
    for k in K_VALUES:
        strategy_specs.append((f"prototype_candidate_oracle_k{k}", k, None, False, "candidate_oracle"))
        strategy_specs.append((f"prototype_vehicle_vote_k{k}", k, None, True, "vehicle_only"))
        strategy_specs.append((f"prototype_bio_closest_k{k}", k, None, True, "vehicle_bio"))
        for lam in BIO_LAMBDAS:
            strategy_specs.append((f"prototype_vehicle_bio_k{k}_lam{lam:.2f}", k, lam, True, "vehicle_bio"))

    for _, event in events.iterrows():
        uid = str(event["event_uid"])
        nei = neighbor_groups.get(uid, pd.DataFrame())
        for strategy, k, lam, deployable, family in strategy_specs:
            delay, rmse, unique_delay_n = select_strategy_for_event(strategy, uid, nei, lookup, k=k, lam=lam)
            keep0 = float(event["keep0_tail_rmse_v241"])
            latest = float(event["latest_tail_rmse_v241"])
            oracle = float(event["oracle_tail_rmse_v241"])
            rows.append(
                {
                    "strategy": strategy,
                    "strategy_family": family,
                    "deployable": bool(deployable),
                    "k": int(k) if k is not None else np.nan,
                    "bio_lambda": float(lam) if lam is not None else np.nan,
                    "event_uid": uid,
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
                    "prototype_unique_delay_n": int(unique_delay_n),
                }
            )
    return pd.DataFrame(rows)


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    """沿用 v258/v261 的分桶方式，同时保留 strategy family 与可部署标记。"""
    rows: List[Dict[str, object]] = []
    buckets = [
        ("all", np.ones(len(selected), dtype=bool)),
        ("bad_top10", selected["bad_top10"].astype(bool).to_numpy()),
        ("very_bad_top5", selected["very_bad_top5"].astype(bool).to_numpy()),
        ("normal", selected["normal"].astype(bool).to_numpy()),
        ("observe_later_like", selected["observe_later_like"].astype(bool).to_numpy()),
        ("strong_steer", selected["strong_steer"].astype(bool).to_numpy()),
        ("early_best_after_400", selected["early_best_after_400"].astype(bool).to_numpy()),
    ]
    for split_name in ["train", "val", "test"]:
        split_mask = selected["split"].astype(str).eq(split_name).to_numpy()
        for bucket, bucket_mask in buckets:
            mask = split_mask & bucket_mask
            if int(mask.sum()) == 0:
                continue
            sub = selected[mask]
            for strategy, g in sub.groupby("strategy", sort=False):
                rows.append(
                    {
                        "split": split_name,
                        "event_group": bucket,
                        "strategy": strategy,
                        "strategy_family": str(g["strategy_family"].iloc[0]),
                        "deployable": bool(g["deployable"].iloc[0]),
                        "k": g["k"].iloc[0],
                        "bio_lambda": g["bio_lambda"].iloc[0],
                        "n": int(len(g)),
                        "selected_tail_rmse_mean": float(g["selected_tail_rmse_v241"].mean()),
                        "keep0_tail_rmse_mean": float(g["keep0_tail_rmse_v241"].mean()),
                        "latest_tail_rmse_mean": float(g["latest_tail_rmse_v241"].mean()),
                        "oracle_tail_rmse_mean": float(g["oracle_tail_rmse_v241"].mean()),
                        "delta_selected_minus_keep0_mean": float(g["delta_selected_minus_keep0"].mean()),
                        "delta_selected_minus_latest_mean": float(g["delta_selected_minus_latest"].mean()),
                        "improve_rate_vs_keep0": float((g["delta_selected_minus_keep0"] < 0).mean()),
                        "selected_delay_ms_mean": float(g["selected_delay_ms"].mean()),
                        "selected_latest_rate": float((g["selected_delay_ms"] >= 1000).mean()),
                        "prototype_unique_delay_n_mean": float(g["prototype_unique_delay_n"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def choose_val_strategies(summary: pd.DataFrame) -> pd.DataFrame:
    """只用 val bad_top10 选择 best vehicle-only 与 best vehicle+bio 策略。"""
    val = summary[
        summary["split"].eq("val")
        & summary["event_group"].eq("bad_top10")
        & summary["deployable"].astype(bool)
    ].copy()
    rows: List[Dict[str, object]] = []
    for label, families in [
        ("val_best_vehicle_only", ["vehicle_only"]),
        ("val_best_vehicle_bio", ["vehicle_bio"]),
    ]:
        sub = val[val["strategy_family"].isin(families)].copy()
        if sub.empty:
            continue
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
    for _, row in chosen.iterrows():
        strategy = str(row["chosen_strategy"])
        for _, s in summary[summary["strategy"].eq(strategy)].iterrows():
            rec = row.to_dict()
            for col in [
                "split",
                "event_group",
                "selected_tail_rmse_mean",
                "delta_selected_minus_keep0_mean",
                "delta_selected_minus_latest_mean",
                "selected_delay_ms_mean",
                "selected_latest_rate",
                "improve_rate_vs_keep0",
                "n",
            ]:
                rec[col] = s[col]
            expanded.append(rec)
    return pd.DataFrame(expanded)


def plot_badtop10_summary(summary: pd.DataFrame, chosen: pd.DataFrame) -> Path:
    path = FIGURES / "v266_test_badtop10_main_comparison.png"
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    if test_bad.empty:
        return path

    rows = []
    base_order = [
        "policy_keep_0ms_anchor",
        "policy_wait_to_latest_anchor",
        "oracle_best_anchor_upper_bound",
    ]
    for strategy in base_order:
        r = test_bad[test_bad["strategy"].eq(strategy)]
        if len(r):
            rows.append((strategy, float(r["selected_tail_rmse_mean"].iloc[0])))

    oracle = test_bad[test_bad["strategy_family"].eq("candidate_oracle")].copy()
    if not oracle.empty:
        oracle = oracle.sort_values(["selected_tail_rmse_mean", "k"], ascending=[True, True]).iloc[0]
        rows.append((f"best_candidate_oracle\\n{oracle['strategy']}", float(oracle["selected_tail_rmse_mean"])))

    for label in ["val_best_vehicle_only", "val_best_vehicle_bio"]:
        sub = chosen[
            chosen["chosen_label"].eq(label)
            & chosen["split"].eq("test")
            & chosen["event_group"].eq("bad_top10")
        ]
        if len(sub):
            rows.append((f"{label}\\n{sub['chosen_strategy'].iloc[0]}", float(sub["selected_tail_rmse_mean"].iloc[0])))

    if not rows:
        return path
    names, vals = zip(*rows)
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    x = np.arange(len(vals))
    colors = ["#9CA3AF", "#E15759", "#B07AA1", "#76B7B2", "#4C78A8", "#F28E2B"][: len(vals)]
    ax.bar(x, vals, color=colors)
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed wait-latest 0.6950")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in names], fontsize=8)
    ax.set_ylabel("test bad_top10 tail RMSE")
    ax.set_title("v266: vehicle-matched prototype headroom 与 bio 重排序")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_headroom_by_k(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v266_candidate_oracle_headroom_by_k.png"
    sub = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy_family"].eq("candidate_oracle")
    ].copy()
    if sub.empty:
        return path
    sub = sub.sort_values("k")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(sub["k"], sub["selected_tail_rmse_mean"], marker="o", color="#4C78A8", label="candidate oracle from vehicle neighbors")
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed wait-latest 0.6950")
    ax.set_xlabel("vehicle-matched prototype K")
    ax.set_ylabel("test bad_top10 candidate oracle tail RMSE")
    ax.set_title("v266: 相似车辆 prototype 候选库是否有 headroom")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v261_script", V261_SCRIPT),
        ("v262_feature_selection", V262_FEATURE_SELECTION),
        ("gptpro_phase02_response", GPTPRO_RESPONSE),
    ]:
        rows.append(
            {
                "label": label,
                "path": str(path),
                "exists": bool(path.exists()),
                "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
            }
        )
    write_csv(pd.DataFrame(rows), LOGS / "input_file_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
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


def compact_focus_table(summary: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    """报告里只放最关键的 test bad_top10 行，完整结果保存在 CSV。"""
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    keep = ["policy_keep_0ms_anchor", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"]
    out = test_bad[test_bad["strategy"].isin(keep)].copy()
    for fam in ["candidate_oracle", "vehicle_only", "vehicle_bio"]:
        sub = test_bad[test_bad["strategy_family"].eq(fam)].copy()
        if not sub.empty:
            sub = sub.sort_values(["selected_tail_rmse_mean", "k", "bio_lambda"], ascending=[True, True, True]).head(3)
            out = pd.concat([out, sub], ignore_index=True)
    for label in ["val_best_vehicle_only", "val_best_vehicle_bio"]:
        sub = chosen[
            chosen["chosen_label"].eq(label)
            & chosen["split"].eq("test")
            & chosen["event_group"].eq("bad_top10")
        ]
        if len(sub):
            row = sub.iloc[0].to_dict()
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
                                "bio_lambda": np.nan,
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


def write_report(
    summary: pd.DataFrame,
    chosen: pd.DataFrame,
    feature_audit: pd.DataFrame,
    merge_audit: pd.DataFrame,
    figures: List[Path],
) -> None:
    focus = compact_focus_table(summary, chosen)
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    candidate = test_bad[test_bad["strategy_family"].eq("candidate_oracle")].copy()
    best_candidate = candidate.sort_values(["selected_tail_rmse_mean", "k"], ascending=[True, True]).iloc[0]

    best_vehicle = chosen[
        chosen["chosen_label"].eq("val_best_vehicle_only")
        & chosen["split"].eq("test")
        & chosen["event_group"].eq("bad_top10")
    ]
    best_bio = chosen[
        chosen["chosen_label"].eq("val_best_vehicle_bio")
        & chosen["split"].eq("test")
        & chosen["event_group"].eq("bad_top10")
    ]

    lines: List[str] = []
    lines.append("# v266 vehicle-matched bio residual prototype")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- GPTPro phase02 的路线 1（wait-benefit / CATE-style）已经基本由 v265 覆盖，结果没有形成 bio 增量。")
    lines.append("- v266 因此验证路线 2：在车辆历史相似的局部区域，train 事件的最佳残差/锚点模式是否能给 query 事件提供少量候选，bio260 是否能在这些候选内部重排序。")
    lines.append("- 如果 candidate oracle 本身不能低于 fixed wait-latest `0.6950`，说明这个候选库没有足够 headroom；如果有 headroom 但 bio reranker 追不上，则问题在可部署选择信号。")
    lines.append("")
    lines.append("## 方法边界")
    lines.append("")
    lines.append("- prototype 只来自 train split；val/test 驾驶员历史完全不参与检索。")
    lines.append("- query 只用 0ms 车辆上下文与 floor 0ms 的 bio260_sp64 特征。")
    lines.append("- 生理不直接预测轨迹，只在 vehicle topK prototype 内部重排。")
    lines.append("- K/lambda 只由 val bad_top10 选择；test 不调参。")
    lines.append("")
    lines.append("## 特征与覆盖")
    lines.append("")
    lines.append(feature_audit.to_markdown(index=False))
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
        "prototype_unique_delay_n_mean",
    ]
    lines.append(focus[[col for col in cols if col in focus.columns]].to_markdown(index=False))
    lines.append("")
    lines.append("## Val 选择的可部署策略")
    lines.append("")
    if chosen.empty:
        lines.append("- 未能选出可部署策略。")
    else:
        display_cols = [
            "chosen_label",
            "chosen_strategy",
            "chosen_family",
            "split",
            "event_group",
            "n",
            "selected_tail_rmse_mean",
            "delta_selected_minus_keep0_mean",
            "delta_selected_minus_latest_mean",
            "selected_delay_ms_mean",
        ]
        lines.append(chosen[chosen["event_group"].isin(["bad_top10", "all"])][display_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    lines.append(
        f"- vehicle-matched candidate oracle 最好为 `{float(best_candidate['selected_tail_rmse_mean']):.4f}` "
        f"({best_candidate['strategy']})；fixed wait-latest 是 `{FIXED_WAIT_LATEST_BADTOP10:.4f}`。"
    )
    if float(best_candidate["selected_tail_rmse_mean"]) < FIXED_WAIT_LATEST_BADTOP10:
        lines.append("- 这说明相似车辆 prototype 候选库理论上有一点 headroom，值得继续看可部署 reranker。")
    else:
        lines.append("- 这说明相似车辆 prototype 候选库本身没有越过 fixed wait-latest，复杂 bio reranker 没有必要继续。")
    if len(best_vehicle) and len(best_bio):
        veh = float(best_vehicle["selected_tail_rmse_mean"].iloc[0])
        bio = float(best_bio["selected_tail_rmse_mean"].iloc[0])
        lines.append(
            f"- val 选出的 vehicle-only prototype 在 test bad_top10 为 `{veh:.4f}`；"
            f"val 选出的 vehicle+bio prototype 为 `{bio:.4f}`。"
        )
        if bio < veh:
            lines.append(f"- bio 在可部署重排上比 vehicle-only 低 `{veh - bio:.4f}`。")
        else:
            lines.append(f"- bio 在可部署重排上比 vehicle-only 高 `{bio - veh:.4f}`，没有形成稳定增量。")
        if bio < FIXED_WAIT_LATEST_BADTOP10:
            lines.append("- vehicle+bio 已低于 fixed wait-latest，可以进入更严格的曲线级残差原型实验。")
        else:
            lines.append("- vehicle+bio 未低于 fixed wait-latest，不能算差样本本质性改善。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    lines.append("## 输入合并审计")
    lines.append("")
    lines.append(merge_audit.to_markdown(index=False))
    lines.append("")
    (REPORTS / "v266_vehicle_matched_bio_residual_prototype_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v266] vehicle-matched bio residual prototype")
    clean_out_dir()
    np.random.seed(SEED)

    cand, events, merge_audit, veh_cols, bio_cols = load_candidate_and_events()
    lookup = candidate_rmse_lookup(cand)
    train_mask = events["split"].astype(str).eq("train").to_numpy()

    veh_x = events[veh_cols].to_numpy(dtype=float)
    bio_x = events[bio_cols].to_numpy(dtype=float)
    veh_z, veh_med, veh_mean, veh_std = fit_fill_scale(veh_x, train_mask)
    bio_z, bio_med, bio_mean, bio_std = fit_fill_scale(bio_x, train_mask)

    max_k = max(K_VALUES)
    neighbors = build_neighbor_table(events, veh_z, bio_z, train_mask, max_k=max_k)
    selected = build_selected(events, neighbors, lookup)
    summary = summarize_selected(selected)
    chosen = choose_val_strategies(summary)
    figures = [plot_badtop10_summary(summary, chosen), plot_headroom_by_k(summary)]

    feature_audit = pd.DataFrame(
        [
            {
                "event_n": int(len(events)),
                "train_event_n": int((events["split"].astype(str) == "train").sum()),
                "val_event_n": int((events["split"].astype(str) == "val").sum()),
                "test_event_n": int((events["split"].astype(str) == "test").sum()),
                "vehicle_feature_n": int(len(veh_cols)),
                "bio260_sp64_feature_n": int(len(bio_cols)),
                "max_k": int(max_k),
                "bio260_uses_post_observation_max": float(merge_audit["bio260_uses_post_observation_max"].iloc[0]),
            }
        ]
    )
    fill_audit = pd.concat(
        [
            pd.DataFrame({"block": "vehicle", "feature": veh_cols, "fill_median": veh_med, "scale_mean": veh_mean, "scale_std": veh_std}),
            pd.DataFrame({"block": "bio260_sp64", "feature": bio_cols, "fill_median": bio_med, "scale_mean": bio_mean, "scale_std": bio_std}),
        ],
        ignore_index=True,
    )

    write_csv(events, TABLES / "v266_event_context_table.csv")
    write_csv(neighbors, TABLES / "v266_vehicle_matched_neighbors.csv")
    write_csv(selected, TABLES / "v266_selected_prototype_by_strategy.csv")
    write_csv(summary, TABLES / "v266_prototype_strategy_summary.csv")
    write_csv(chosen, TABLES / "v266_val_chosen_strategy_summary.csv")
    write_csv(feature_audit, TABLES / "v266_feature_block_audit.csv")
    write_csv(fill_audit, TABLES / "v266_feature_fill_audit.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary, chosen, feature_audit, merge_audit, figures)
    write_file_inventory()
    zip_ok = make_zip()

    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    candidate = test_bad[test_bad["strategy_family"].eq("candidate_oracle")].copy()
    best_candidate = candidate.sort_values(["selected_tail_rmse_mean", "k"], ascending=[True, True]).iloc[0]
    best_bio = chosen[
        chosen["chosen_label"].eq("val_best_vehicle_bio")
        & chosen["split"].eq("test")
        & chosen["event_group"].eq("bad_top10")
    ]
    best_bio_rmse = float(best_bio["selected_tail_rmse_mean"].iloc[0]) if len(best_bio) else float("nan")
    guardrail = {
        "pass": bool(zip_ok and float(merge_audit["bio260_uses_post_observation_max"].iloc[0]) == 0.0),
        "zip_testzip": bool(zip_ok),
        "prototype_source_split": "train_only",
        "val_tuned_only": True,
        "test_subject_history_used": False,
        "oracle_deployable": False,
        "bio_role": "rerank_within_vehicle_matched_prototypes_only",
        "event_n": int(len(events)),
        "vehicle_feature_n": int(len(veh_cols)),
        "bio260_sp64_feature_n": int(len(bio_cols)),
        "best_candidate_oracle_test_badtop10": float(best_candidate["selected_tail_rmse_mean"]),
        "best_vehicle_bio_val_chosen_test_badtop10": best_bio_rmse,
        "fixed_wait_latest_badtop10": float(FIXED_WAIT_LATEST_BADTOP10),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v266 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = compact_focus_table(summary, chosen)
    print(f"[v266] report={REPORTS / 'v266_vehicle_matched_bio_residual_prototype_cn.md'}")
    print(f"[v266] zip={ZIP_PATH}")
    print(
        focus[
            [
                "strategy",
                "strategy_family",
                "selected_tail_rmse_mean",
                "delta_selected_minus_latest_mean",
                "selected_delay_ms_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
