#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v284 dynamic low-identity physiology route gate.

v283 已经关闭了旧的 bio selector/reranker/reliability-filter 微调路线。
v284 继续推进 active goal，但先不训练轨迹模型，而是构造一套新的生理状态表示并过 route gate：

1. 只使用 v260 从 200Hz 连续层派生的 0ms 事件前 biomarker；
2. 只在 train split 上计算行为相关性与 subject/recording 身份惩罚；
3. 优先保留动态变化、斜率、相位、burst、HR/RESP/ECG/EMG/SCR 变化类特征；
4. 在 v278 的 vehicle top40 候选池里重新计算生理距离排序；
5. 用 validation 选择 feature set，test 只报告。

如果这一步仍不过 gate，说明“重定义但仍基于当前 200Hz biomarker 的生理状态”也不足以
弥补锚点前车辆信息不足，后续需要更底层的信号重处理或把生理降级为 subject-aware 分支。
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
SCRIPTS = BASELINES / "scripts"

OUT = BASELINES / "v284_dynamic_low_identity_physio_route_gate_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v284_dynamic_low_identity_physio_route_gate_20260702_pack.zip"

V284_SCRIPT = SCRIPTS / "stage03_v284_dynamic_low_identity_physio_route_gate_20260702.py"
V260_FEATURES = (
    BASELINES
    / "v260_event_biomarker_physio_rebuild_20260702"
    / "tables"
    / "v260_event_biomarker_features.csv"
)
V260_TARGETS = (
    BASELINES
    / "v260_event_biomarker_physio_rebuild_20260702"
    / "tables"
    / "v260_future_behavior_targets.csv"
)
V278_CANDIDATES = (
    BASELINES
    / "v278_listwise_candidate_rank_loss_20260702"
    / "tables"
    / "v278_candidate_listrank_predictions_compact.csv"
)
V272_DIAG = (
    BASELINES
    / "v272_physio_ambiguity_disambiguation_20260702"
    / "tables"
    / "v272_neighbor_rank_diagnostics_by_event.csv"
)
V283_GUARDRAIL = (
    BASELINES
    / "v283_physio_route_lineage_gap_audit_20260702"
    / "logs"
    / "guardrail_check.json"
)

SEED = 28402
FIXED_WAIT_LATEST_BADTOP10 = 0.695048
MIN_GROUP_N = 8

TARGETS_FOR_SCREEN = [
    "future_cluster4",
    "high_future_abs_q75",
    "high_future_range_q75",
    "strong_steer_existing",
    "bad_top10_v250_diagnostic",
]


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


def read_json_optional(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def eta_squared(values: np.ndarray, labels: Iterable[object]) -> float:
    """单特征 eta²；只在有限值上计算。"""

    x = np.asarray(values, dtype=float)
    y = pd.Series(labels).astype(str).to_numpy()
    mask = np.isfinite(x) & pd.notna(y)
    x = x[mask]
    y = y[mask]
    if x.size < 10 or np.nanstd(x) < 1e-12:
        return 0.0
    grand = float(np.mean(x))
    ss_total = float(np.sum((x - grand) ** 2))
    if ss_total <= 1e-12:
        return 0.0
    ss_between = 0.0
    for label in pd.unique(y):
        sub = x[y == label]
        if sub.size == 0:
            continue
        ss_between += float(sub.size) * float((np.mean(sub) - grand) ** 2)
    return float(max(0.0, min(1.0, ss_between / ss_total)))


def finite_rate(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.isfinite(x).mean())


def rank_corr_distance_rmse(distance: np.ndarray, rmse: np.ndarray) -> float:
    if len(distance) < 3:
        return math.nan
    d = pd.Series(distance).rank(method="average").to_numpy(dtype=float)
    r = pd.Series(rmse).rank(method="average").to_numpy(dtype=float)
    if np.nanstd(d) < 1e-9 or np.nanstd(r) < 1e-9:
        return math.nan
    return float(np.corrcoef(d, r)[0, 1])


def load_event_feature_table() -> pd.DataFrame:
    if not V260_FEATURES.exists():
        raise FileNotFoundError(f"缺少 v260 生理特征表: {V260_FEATURES}")
    if not V260_TARGETS.exists():
        raise FileNotFoundError(f"缺少 v260 行为目标表: {V260_TARGETS}")
    features = pd.read_csv(V260_FEATURES, low_memory=False)
    targets = pd.read_csv(V260_TARGETS, low_memory=False)
    features["delay_ms"] = pd.to_numeric(features["delay_ms"], errors="coerce")
    delay0 = features[features["delay_ms"].eq(0)].drop_duplicates("event_uid", keep="first").copy()
    merged = delay0.merge(targets, on="row_index", how="left", suffixes=("", "_target"))
    if merged["event_uid"].duplicated().any():
        raise RuntimeError("delay0 生理特征表存在重复 event_uid")
    return merged.reset_index(drop=True)


def load_candidate_table() -> pd.DataFrame:
    if not V278_CANDIDATES.exists():
        raise FileNotFoundError(f"缺少 v278 top40 候选表: {V278_CANDIDATES}")
    cols = [
        "event_uid",
        "split",
        "subject",
        "prototype_event_uid",
        "mapped_delay_ms",
        "target_tail_rmse_v241",
        "latest_tail_rmse_v241",
        "target_gain_vs_latest",
        "feature_set",
    ]
    cand = pd.read_csv(V278_CANDIDATES, usecols=cols, low_memory=False)
    cand = cand[cand["feature_set"].astype(str).eq("listrank_vehicle")].copy()
    cand = cand.drop_duplicates(["event_uid", "prototype_event_uid", "mapped_delay_ms"]).reset_index(drop=True)
    for col in ["mapped_delay_ms", "target_tail_rmse_v241", "latest_tail_rmse_v241", "target_gain_vs_latest"]:
        cand[col] = pd.to_numeric(cand[col], errors="coerce")
    return cand


def candidate_feature_columns(events: pd.DataFrame) -> Tuple[List[str], List[str]]:
    meta = {
        "row_index",
        "event_uid",
        "subject",
        "recording",
        "split",
        "delay_ms",
        "observation_s",
        "bio260_status",
        "future_peak_sign",
        "future_cluster4",
        "future_cluster6",
    }
    meta.update(TARGETS_FOR_SCREEN)
    excluded_substrings = [
        "_rows",
        "_duration_s",
        "sample_hz",
        "recording_duration",
        "uses_post_observation",
        "baseline_rows",
        "baseline_duration",
        "hrv_existing",
    ]
    all_cols = []
    dynamic_cols = []
    dynamic_keywords = [
        "delta_",
        "_z_slope",
        "_z_last_minus_first",
        "_z_range",
        "burst_rate",
        "burst_episode",
        "bpm_from_peaks",
        "ibi_",
        "rmssd",
        "sdnn",
        "period_",
        "phase_",
        "zero_up",
        "_z_pos_area",
        "_z_abs_area",
        "_z_std",
    ]
    for col in events.columns:
        if col in meta or not col.startswith("bio260_"):
            continue
        if any(s in col for s in excluded_substrings):
            continue
        if not pd.api.types.is_numeric_dtype(events[col]):
            continue
        all_cols.append(col)
        if any(k in col for k in dynamic_keywords):
            dynamic_cols.append(col)
    return all_cols, dynamic_cols


def feature_screening(events: pd.DataFrame, all_cols: List[str], dynamic_cols: List[str]) -> pd.DataFrame:
    train = events[events["split"].astype(str).eq("train")].copy()
    rows: List[Dict[str, object]] = []
    for col in all_cols:
        x = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=float)
        behavior_scores = {}
        for target in TARGETS_FOR_SCREEN:
            if target not in train.columns:
                continue
            behavior_scores[target] = eta_squared(x, train[target])
        identity_subject = eta_squared(x, train["subject"])
        identity_recording = eta_squared(x, train["recording"])
        identity_max = max(identity_subject, identity_recording)
        behavior_max = max(behavior_scores.values()) if behavior_scores else 0.0
        bad_score = behavior_scores.get("bad_top10_v250_diagnostic", 0.0)
        is_dynamic = col in dynamic_cols
        no_abs_amp = not any(k in col for k in ["peak_amp", "_count", "_p95"])
        rows.append(
            {
                "feature": col,
                "finite_rate_train": finite_rate(x),
                "is_dynamic": bool(is_dynamic),
                "no_abs_amp": bool(no_abs_amp),
                "behavior_eta_max": float(behavior_max),
                "bad_top10_eta": float(bad_score),
                "identity_eta_max": float(identity_max),
                "identity_to_behavior_ratio": float(identity_max / max(behavior_max, 1e-6)),
                "behavior_identity_score": float(behavior_max / (identity_max + 0.01)),
                "bad_identity_score": float(bad_score / (identity_max + 0.01)),
                **{f"eta_{k}": float(v) for k, v in behavior_scores.items()},
            }
        )
    screen = pd.DataFrame(rows)
    screen = screen.sort_values(["behavior_identity_score", "behavior_eta_max"], ascending=False).reset_index(drop=True)
    return screen


def choose_feature_sets(screen: pd.DataFrame) -> Dict[str, List[str]]:
    usable = screen[screen["finite_rate_train"].ge(0.80)].copy()
    dyn = usable[usable["is_dynamic"].astype(bool)].copy()
    dyn_noamp = dyn[dyn["no_abs_amp"].astype(bool)].copy()

    def top(df: pd.DataFrame, sort_cols: List[str], n: int) -> List[str]:
        if df.empty:
            return []
        return (
            df.sort_values(sort_cols, ascending=[False] * len(sort_cols))["feature"]
            .drop_duplicates()
            .head(n)
            .astype(str)
            .tolist()
        )

    low_identity_pool = dyn[dyn["identity_eta_max"].le(0.10)].copy()
    strict_pool = dyn_noamp[dyn_noamp["identity_to_behavior_ratio"].le(25.0)].copy()
    if len(low_identity_pool) < 24:
        low_identity_pool = dyn.sort_values("identity_eta_max", ascending=True).head(max(24, min(64, len(dyn))))
    if len(strict_pool) < 16:
        strict_pool = dyn_noamp.sort_values("identity_to_behavior_ratio", ascending=True).head(max(16, min(48, len(dyn_noamp))))

    feature_sets = {
        "dyn_behavior_identity_top64": top(dyn, ["behavior_identity_score", "behavior_eta_max"], 64),
        "dyn_bad_identity_top48": top(dyn, ["bad_identity_score", "bad_top10_eta"], 48),
        "dyn_noamp_multi_top48": top(dyn_noamp, ["behavior_identity_score", "behavior_eta_max"], 48),
        "low_identity_dyn_top48": top(low_identity_pool, ["behavior_identity_score", "behavior_eta_max"], 48),
        "strict_ratio_noamp_top32": top(strict_pool, ["behavior_identity_score", "behavior_eta_max"], 32),
    }
    return {k: v for k, v in feature_sets.items() if len(v) >= 8}


def fit_transform_features(events: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """只用 train split 拟合填充和标准化参数。"""

    x = events[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    train_mask = events["split"].astype(str).eq("train").to_numpy()
    train_x = x[train_mask]
    med = np.nanmedian(train_x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isfinite(x), x, med[None, :])
    mean = np.nanmean(filled[train_mask], axis=0)
    std = np.nanstd(filled[train_mask], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    z = (filled - mean[None, :]) / std[None, :]
    audit = pd.DataFrame({"feature": cols, "train_median": med, "train_mean": mean, "train_std": std})
    zdf = pd.DataFrame(z.astype(np.float32), columns=cols)
    zdf.insert(0, "event_uid", events["event_uid"].astype(str).to_numpy())
    return zdf, audit


def build_event_context(cand: pd.DataFrame) -> pd.DataFrame:
    base = cand.groupby(["event_uid", "split", "subject"], as_index=False).agg(
        latest_tail_rmse_v241=("latest_tail_rmse_v241", "first"),
        candidate_oracle_rmse=("target_tail_rmse_v241", "min"),
        candidate_rmse_std=("target_tail_rmse_v241", "std"),
        unique_delay_n=("mapped_delay_ms", "nunique"),
        candidate_n=("prototype_event_uid", "count"),
    )

    # 为了和 v272-v283 的差样本口径完全一致，优先复用 v272 已固定的
    # bad_top10 / vehicle_ambiguous 标签；只有缺失时才回退到当前表内重算。
    if V272_DIAG.exists():
        diag = pd.read_csv(
            V272_DIAG,
            usecols=["raw_set", "k", "event_uid", "bad_top10", "very_bad_top5", "vehicle_ambiguous"],
            low_memory=False,
        )
        diag = diag[diag["k"].eq(40) & diag["raw_set"].astype(str).eq("subject_summary64")].copy()
        diag = diag.drop_duplicates("event_uid")
        for col in ["bad_top10", "very_bad_top5", "vehicle_ambiguous"]:
            diag[col] = diag[col].astype(str).str.lower().isin(["true", "1", "yes"])
        base = base.merge(
            diag[["event_uid", "bad_top10", "very_bad_top5", "vehicle_ambiguous"]],
            on="event_uid",
            how="left",
        )
    else:
        base["bad_top10"] = np.nan
        base["very_bad_top5"] = np.nan
        base["vehicle_ambiguous"] = np.nan

    missing_label = base["bad_top10"].isna() | base["vehicle_ambiguous"].isna()
    if bool(missing_label.any()):
        base.loc[missing_label, "bad_top10"] = False
        base.loc[missing_label, "very_bad_top5"] = False
        for split, sub_idx in base[missing_label].groupby("split").groups.items():
            sub = base.loc[sub_idx]
            q90 = float(sub["latest_tail_rmse_v241"].quantile(0.90))
            q95 = float(sub["latest_tail_rmse_v241"].quantile(0.95))
            base.loc[sub_idx, "bad_top10"] = base.loc[sub_idx, "latest_tail_rmse_v241"].ge(q90)
            base.loc[sub_idx, "very_bad_top5"] = base.loc[sub_idx, "latest_tail_rmse_v241"].ge(q95)
        base.loc[missing_label, "vehicle_ambiguous"] = (
            base.loc[missing_label, "unique_delay_n"].ge(3)
            & (base.loc[missing_label, "latest_tail_rmse_v241"] - base.loc[missing_label, "candidate_oracle_rmse"]).ge(0.05)
            & base.loc[missing_label, "candidate_rmse_std"].ge(0.05)
        )
    for col in ["bad_top10", "very_bad_top5", "vehicle_ambiguous"]:
        base[col] = base[col].astype(bool)
    base["bad_top10_vehicle_ambiguous"] = base["bad_top10"] & base["vehicle_ambiguous"]
    return base


def evaluate_feature_set(
    name: str,
    cols: List[str],
    events: pd.DataFrame,
    cand: pd.DataFrame,
    context: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zdf, scaler = fit_transform_features(events, cols)
    mat = zdf.set_index("event_uid")[cols]
    missing_events = sorted(set(cand["event_uid"].astype(str)) - set(mat.index))
    missing_proto = sorted(set(cand["prototype_event_uid"].astype(str)) - set(mat.index))
    if missing_events or missing_proto:
        raise RuntimeError(f"{name} 缺少 event/prototype 生理特征: query={len(missing_events)}, proto={len(missing_proto)}")

    rows: List[Dict[str, object]] = []
    for event_uid, sub in cand.groupby("event_uid", sort=False):
        q = mat.loc[str(event_uid)].to_numpy(dtype=float)
        proto = mat.loc[sub["prototype_event_uid"].astype(str).tolist()].to_numpy(dtype=float)
        dist = np.mean((proto - q[None, :]) ** 2, axis=1)
        rmse = sub["target_tail_rmse_v241"].to_numpy(dtype=float)
        order = np.argsort(dist, kind="mergesort")
        top1 = int(order[0])
        top3 = order[: min(3, len(order))]
        top5 = order[: min(5, len(order))]
        best = int(np.nanargmin(rmse))
        rank_of_best = int(np.flatnonzero(order == best)[0]) + 1
        rows.append(
            {
                "feature_set": name,
                "event_uid": str(event_uid),
                "split": str(sub.iloc[0]["split"]),
                "subject": str(sub.iloc[0]["subject"]),
                "latest_rmse": float(sub.iloc[0]["latest_tail_rmse_v241"]),
                "bio_top1_rmse": float(rmse[top1]),
                "bio_top1_delay_ms": int(sub.iloc[top1]["mapped_delay_ms"]),
                "bio_top3_oracle_rmse": float(np.nanmin(rmse[top3])),
                "bio_top5_oracle_rmse": float(np.nanmin(rmse[top5])),
                "candidate_oracle_rmse": float(np.nanmin(rmse)),
                "bio_best_candidate_rank": rank_of_best,
                "bio_best_in_top3": bool(rank_of_best <= 3),
                "bio_best_in_top5": bool(rank_of_best <= 5),
                "bio_distance_rmse_rank_corr": rank_corr_distance_rmse(dist, rmse),
                "candidate_n": int(len(sub)),
            }
        )
    per_event = pd.DataFrame(rows)
    per_event = per_event.merge(
        context[
            [
                "event_uid",
                "bad_top10",
                "very_bad_top5",
                "vehicle_ambiguous",
                "bad_top10_vehicle_ambiguous",
                "unique_delay_n",
                "candidate_rmse_std",
            ]
        ],
        on="event_uid",
        how="left",
    )
    scaler.insert(0, "feature_set", name)
    return per_event, scaler, pd.DataFrame({"feature_set": [name], "feature_n": [len(cols)]})


def expand_groups(per_event: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in per_event.iterrows():
        groups = ["all"]
        if bool(row.get("vehicle_ambiguous", False)):
            groups.append("vehicle_ambiguous")
        if bool(row.get("bad_top10", False)):
            groups.append("bad_top10")
        if bool(row.get("very_bad_top5", False)):
            groups.append("very_bad_top5")
        if bool(row.get("bad_top10_vehicle_ambiguous", False)):
            groups.append("bad_top10_vehicle_ambiguous")
        for group in groups:
            item = row.to_dict()
            item["event_group"] = group
            rows.append(item)
    return pd.DataFrame(rows)


def summarize_groups(expanded: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (feature_set, split, event_group), sub in expanded.groupby(["feature_set", "split", "event_group"], dropna=False):
        latest = sub["latest_rmse"].to_numpy(dtype=float)
        top1 = sub["bio_top1_rmse"].to_numpy(dtype=float)
        top3 = sub["bio_top3_oracle_rmse"].to_numpy(dtype=float)
        top5 = sub["bio_top5_oracle_rmse"].to_numpy(dtype=float)
        candidate_oracle = sub["candidate_oracle_rmse"].to_numpy(dtype=float)
        gap = latest - candidate_oracle
        valid_gap = np.isfinite(gap) & (gap > 1e-9)
        closure_top1 = np.full_like(gap, np.nan, dtype=float)
        closure_top3 = np.full_like(gap, np.nan, dtype=float)
        closure_top1[valid_gap] = (latest[valid_gap] - top1[valid_gap]) / gap[valid_gap]
        closure_top3[valid_gap] = (latest[valid_gap] - top3[valid_gap]) / gap[valid_gap]
        corr = sub["bio_distance_rmse_rank_corr"].to_numpy(dtype=float)
        rows.append(
            {
                "feature_set": feature_set,
                "split": split,
                "event_group": event_group,
                "n": int(sub["event_uid"].nunique()),
                "latest_rmse_mean": float(np.nanmean(latest)),
                "candidate_oracle_rmse_mean": float(np.nanmean(candidate_oracle)),
                "bio_top1_rmse_mean": float(np.nanmean(top1)),
                "bio_top3_oracle_rmse_mean": float(np.nanmean(top3)),
                "bio_top5_oracle_rmse_mean": float(np.nanmean(top5)),
                "bio_top1_minus_latest_mean": float(np.nanmean(top1 - latest)),
                "bio_top3_minus_latest_mean": float(np.nanmean(top3 - latest)),
                "bio_top5_minus_latest_mean": float(np.nanmean(top5 - latest)),
                "bio_top1_beats_latest_rate": float(np.nanmean(top1 < latest)),
                "bio_top3_beats_latest_rate": float(np.nanmean(top3 < latest)),
                "bio_best_rank_mean": float(np.nanmean(sub["bio_best_candidate_rank"].to_numpy(dtype=float))),
                "bio_best_in_top3_rate": float(np.nanmean(sub["bio_best_in_top3"].astype(bool))),
                "bio_best_in_top5_rate": float(np.nanmean(sub["bio_best_in_top5"].astype(bool))),
                "bio_corr_mean": float(np.nanmean(corr)),
                "bio_corr_median": float(np.nanmedian(corr)),
                "bio_corr_positive_rate": float(np.nanmean(corr > 0)),
                "bio_corr_gt_010_rate": float(np.nanmean(corr > 0.10)),
                "bio_top1_gap_closure_mean": float(np.nanmean(closure_top1)),
                "bio_top3_gap_closure_mean": float(np.nanmean(closure_top3)),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_group", "split", "bio_top1_minus_latest_mean"])


def val_chosen_generalization(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    event_groups = ["all", "vehicle_ambiguous", "bad_top10", "bad_top10_vehicle_ambiguous"]
    methods = [
        ("bio_top1", "bio_top1_rmse_mean", "bio_top1_minus_latest_mean", True),
        ("bio_top3_oracle", "bio_top3_oracle_rmse_mean", "bio_top3_minus_latest_mean", False),
        ("bio_top5_oracle", "bio_top5_oracle_rmse_mean", "bio_top5_minus_latest_mean", False),
    ]
    for group in event_groups:
        for method, rmse_col, delta_col, deployable in methods:
            val = summary[summary["split"].eq("val") & summary["event_group"].eq(group) & summary["n"].ge(MIN_GROUP_N)].copy()
            if val.empty:
                continue
            val = val.sort_values([delta_col, "bio_corr_mean"], ascending=[True, False]).reset_index(drop=True)
            chosen = val.iloc[0]
            test = summary[
                summary["split"].eq("test")
                & summary["event_group"].eq(group)
                & summary["feature_set"].astype(str).eq(str(chosen["feature_set"]))
            ]
            if test.empty:
                continue
            t = test.iloc[0]
            rows.append(
                {
                    "event_group": group,
                    "method": method,
                    "deployable": bool(deployable),
                    "val_chosen_feature_set": str(chosen["feature_set"]),
                    "val_n": int(chosen["n"]),
                    "test_n": int(t["n"]),
                    "val_latest_rmse_mean": float(chosen["latest_rmse_mean"]),
                    "val_method_rmse_mean": float(chosen[rmse_col]),
                    "val_delta_vs_latest_mean": float(chosen[delta_col]),
                    "val_corr_mean": float(chosen["bio_corr_mean"]),
                    "test_latest_rmse_mean": float(t["latest_rmse_mean"]),
                    "test_method_rmse_mean": float(t[rmse_col]),
                    "test_delta_vs_latest_mean": float(t[delta_col]),
                    "test_corr_mean": float(t["bio_corr_mean"]),
                    "test_corr_positive_rate": float(t["bio_corr_positive_rate"]),
                    "test_passes_latest": bool(float(t[delta_col]) < -1e-9),
                    "val_and_test_same_direction_gain": bool(
                        float(chosen[delta_col]) < -1e-9 and float(t[delta_col]) < -1e-9
                    ),
                }
            )
    return pd.DataFrame(rows)


def route_gate_decision(summary: pd.DataFrame, val_test: pd.DataFrame) -> pd.DataFrame:
    def get(group: str, method: str) -> pd.Series | None:
        sub = val_test[val_test["event_group"].eq(group) & val_test["method"].eq(method)]
        if sub.empty:
            return None
        return sub.iloc[0]

    top1_bad = get("bad_top10", "bio_top1")
    top1_amb = get("bad_top10_vehicle_ambiguous", "bio_top1")
    top3_amb = get("bad_top10_vehicle_ambiguous", "bio_top3_oracle")
    test_bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")]
    best_corr = float(test_bad["bio_corr_mean"].max()) if not test_bad.empty else math.nan
    best_top1_delta = float(test_bad["bio_top1_minus_latest_mean"].min()) if not test_bad.empty else math.nan

    rows = [
        {
            "check": "deployable_top1_val_chosen_bad_top10",
            "requirement": "validation 选出的新生理 top1 在 test bad_top10 上低于 latest",
            "pass": bool(top1_bad is not None and float(top1_bad["test_delta_vs_latest_mean"]) < -1e-9),
            "evidence": None if top1_bad is None else float(top1_bad["test_delta_vs_latest_mean"]),
            "deployable": True,
        },
        {
            "check": "deployable_top1_val_chosen_bad_ambiguous",
            "requirement": "validation 选出的新生理 top1 在 test bad_top10_vehicle_ambiguous 上低于 latest",
            "pass": bool(top1_amb is not None and float(top1_amb["test_delta_vs_latest_mean"]) < -1e-9),
            "evidence": None if top1_amb is None else float(top1_amb["test_delta_vs_latest_mean"]),
            "deployable": True,
        },
        {
            "check": "oracle_top3_val_test_same_direction_bad_ambiguous",
            "requirement": "非部署 top3 上限在 val/test 歧义差样本上同向改善",
            "pass": bool(top3_amb is not None and bool(top3_amb["val_and_test_same_direction_gain"])),
            "evidence": None
            if top3_amb is None
            else f"val={float(top3_amb['val_delta_vs_latest_mean']):.6f}, test={float(top3_amb['test_delta_vs_latest_mean']):.6f}",
            "deployable": False,
        },
        {
            "check": "test_bad_top10_any_feature_corr_gt_005",
            "requirement": "test bad_top10 至少一个新特征集的生理距离-真实误差排序相关均值 > 0.05",
            "pass": bool(np.isfinite(best_corr) and best_corr > 0.05),
            "evidence": best_corr,
            "deployable": False,
        },
        {
            "check": "test_best_top1_diagnostic_beats_latest",
            "requirement": "即使 test-best 诊断，新生理 top1 至少有一个特征集低于 latest",
            "pass": bool(np.isfinite(best_top1_delta) and best_top1_delta < -1e-9),
            "evidence": best_top1_delta,
            "deployable": False,
        },
    ]
    out = pd.DataFrame(rows)
    out["route_viable_now"] = bool(out["pass"].all())
    return out


def plot_val_test_delta(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v284_badtop10_val_test_delta.png"
    data = summary[summary["event_group"].eq("bad_top10") & summary["split"].isin(["val", "test"])].copy()
    feature_sets = list(data["feature_set"].drop_duplicates())
    x = np.arange(len(feature_sets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (split, col, label) in enumerate(
        [
            ("val", "bio_top1_minus_latest_mean", "val top1"),
            ("test", "bio_top1_minus_latest_mean", "test top1"),
            ("val", "bio_top3_minus_latest_mean", "val top3 oracle"),
            ("test", "bio_top3_minus_latest_mean", "test top3 oracle"),
        ]
    ):
        vals = []
        for fs in feature_sets:
            sub = data[data["feature_set"].eq(fs) & data["split"].eq(split)]
            vals.append(float(sub[col].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=label)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in feature_sets], fontsize=8)
    ax.set_ylabel("RMSE delta vs latest, lower is better")
    ax.set_title("v284: dynamic low-identity physiology on bad_top10")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_corr(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v284_bad_ambiguous_corr.png"
    data = summary[summary["event_group"].eq("bad_top10_vehicle_ambiguous") & summary["split"].isin(["train", "val", "test"])].copy()
    feature_sets = list(data["feature_set"].drop_duplicates())
    x = np.arange(len(feature_sets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, split in enumerate(["train", "val", "test"]):
        vals = []
        for fs in feature_sets:
            sub = data[data["feature_set"].eq(fs) & data["split"].eq(split)]
            vals.append(float(sub["bio_corr_mean"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=split)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(0.05, color="tab:green", linestyle="--", linewidth=1, label="weak useful corr=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("_", "\n") for s in feature_sets], fontsize=8)
    ax.set_ylabel("mean rank corr")
    ax.set_title("v284: physiology distance vs candidate RMSE rank")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_feature_screen(screen: pd.DataFrame) -> Path:
    path = FIGURES / "v284_feature_screen_identity_vs_behavior.png"
    data = screen[screen["is_dynamic"].astype(bool)].copy()
    data = data.sort_values("behavior_identity_score", ascending=False).head(120)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(data["identity_eta_max"], data["behavior_eta_max"], s=18, alpha=0.75)
    ax.set_xlabel("identity eta max")
    ax.set_ylabel("behavior eta max")
    ax.set_title("v284: train-only feature screen")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def table_to_md(df: pd.DataFrame, cols: List[str], n: int | None = None) -> str:
    use = df[[c for c in cols if c in df.columns]].copy()
    if n is not None:
        use = use.head(n)
    if use.empty:
        return "（无记录）"
    return use.to_markdown(index=False)


def write_report(
    screen: pd.DataFrame,
    feature_audit: pd.DataFrame,
    summary: pd.DataFrame,
    val_test: pd.DataFrame,
    decision: pd.DataFrame,
    figures: List[Path],
    guardrail: Dict[str, object],
) -> Path:
    path = REPORTS / "v284_dynamic_low_identity_physio_route_gate_cn.md"
    bad = summary[summary["event_group"].eq("bad_top10") & summary["split"].isin(["val", "test"])].sort_values(
        ["split", "bio_top1_minus_latest_mean"]
    )
    bad_amb = summary[
        summary["event_group"].eq("bad_top10_vehicle_ambiguous")
        & summary["split"].isin(["val", "test"])
    ].sort_values(["split", "bio_top1_minus_latest_mean"])
    lines: List[str] = []
    lines.append("# v284 dynamic low-identity physiology route gate")
    lines.append("")
    lines.append("## 本轮目的")
    lines.append("")
    lines.append("- 承接 v283：不再沿旧 bio selector 微调，而是构造新的低身份、动态生理状态特征。")
    lines.append("- 用 train-only 行为相关性和 subject/recording 身份惩罚筛特征。")
    lines.append("- 在 v278 vehicle top40 候选池中重新计算生理距离排序，先过 route gate，再谈轨迹模型。")
    lines.append("")
    lines.append("## route gate 判定")
    lines.append("")
    lines.append(table_to_md(decision, ["check", "requirement", "pass", "evidence", "deployable", "route_viable_now"]))
    lines.append("")
    lines.append("## feature set 审计")
    lines.append("")
    lines.append(table_to_md(feature_audit, ["feature_set", "feature_n"]))
    lines.append("")
    lines.append("## validation 选择后的 test 泛化")
    lines.append("")
    lines.append(
        table_to_md(
            val_test,
            [
                "event_group",
                "method",
                "deployable",
                "val_chosen_feature_set",
                "val_delta_vs_latest_mean",
                "test_delta_vs_latest_mean",
                "test_corr_mean",
                "test_passes_latest",
                "val_and_test_same_direction_gain",
            ],
        )
    )
    lines.append("")
    lines.append("## bad_top10 分层")
    lines.append("")
    show_cols = [
        "feature_set",
        "split",
        "n",
        "latest_rmse_mean",
        "bio_top1_rmse_mean",
        "bio_top1_minus_latest_mean",
        "bio_top3_oracle_rmse_mean",
        "bio_top3_minus_latest_mean",
        "bio_corr_mean",
        "bio_best_in_top3_rate",
    ]
    lines.append(table_to_md(bad, show_cols))
    lines.append("")
    lines.append("## bad_top10 + vehicle_ambiguous 分层")
    lines.append("")
    lines.append(table_to_md(bad_amb, show_cols))
    lines.append("")
    lines.append("## train-only 特征筛选 top20")
    lines.append("")
    lines.append(
        table_to_md(
            screen.sort_values("behavior_identity_score", ascending=False),
            [
                "feature",
                "finite_rate_train",
                "behavior_eta_max",
                "bad_top10_eta",
                "identity_eta_max",
                "identity_to_behavior_ratio",
                "behavior_identity_score",
                "is_dynamic",
            ],
            n=20,
        )
    )
    lines.append("")
    lines.append("## 关键判读")
    lines.append("")
    route_viable = bool(decision["route_viable_now"].iloc[0]) if len(decision) else False
    if route_viable:
        lines.append("- route gate 通过：新构造的动态低身份生理状态已经具备进入轨迹模型的最低证据。")
    else:
        lines.append("- route gate 未通过：即使重筛动态低身份 biomarker，当前生理状态仍未稳定弥补车辆锚点前信息不足。")
        lines.append("- 这说明下一步若还继续生理 goal，需要更底层的信号重处理或明确转为 subject-aware 个体校准任务；不应直接训练更复杂融合模型。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    lines.append("")
    lines.append("## guardrail")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(guardrail, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_input_hashes() -> None:
    rows = []
    for key, path in {
        "v284_script": V284_SCRIPT,
        "v260_features": V260_FEATURES,
        "v260_targets": V260_TARGETS,
        "v278_candidates": V278_CANDIDATES,
        "v272_diag": V272_DIAG,
        "v283_guardrail": V283_GUARDRAIL,
    }.items():
        if path.exists():
            rows.append({"key": key, "path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    write_csv(pd.DataFrame(rows), LOGS / "input_hashes.csv")


def write_file_inventory() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size})
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


def main() -> None:
    print("[v284] 目的：用动态低身份生理状态重新验证车辆歧义候选 route gate。", flush=True)
    clean_out_dir()
    events = load_event_feature_table()
    cand = load_candidate_table()
    context = build_event_context(cand)
    all_cols, dynamic_cols = candidate_feature_columns(events)
    screen = feature_screening(events, all_cols, dynamic_cols)
    feature_sets = choose_feature_sets(screen)
    if not feature_sets:
        raise RuntimeError("没有可用的动态低身份 feature set")

    per_event_parts = []
    scaler_parts = []
    feature_audit_parts = []
    for name, cols in feature_sets.items():
        print(f"[v284] evaluate feature_set={name} feature_n={len(cols)}", flush=True)
        per_event, scaler, audit = evaluate_feature_set(name, cols, events, cand, context)
        per_event_parts.append(per_event)
        scaler_parts.append(scaler)
        feature_audit_parts.append(audit)

    per_event_all = pd.concat(per_event_parts, ignore_index=True)
    scaler_audit = pd.concat(scaler_parts, ignore_index=True)
    feature_audit = pd.concat(feature_audit_parts, ignore_index=True)
    expanded = expand_groups(per_event_all)
    summary = summarize_groups(expanded)
    val_test = val_chosen_generalization(summary)
    decision = route_gate_decision(summary, val_test)

    write_csv(screen, TABLES / "v284_train_only_feature_screen.csv")
    write_csv(feature_audit, TABLES / "v284_feature_set_audit.csv")
    write_csv(scaler_audit, TABLES / "v284_scaler_audit.csv")
    write_csv(per_event_all, TABLES / "v284_route_gate_per_event.csv")
    write_csv(summary, TABLES / "v284_route_group_summary.csv")
    write_csv(val_test, TABLES / "v284_val_chosen_generalization.csv")
    write_csv(decision, TABLES / "v284_route_gate_decision.csv")

    figures = [
        plot_val_test_delta(summary),
        plot_corr(summary),
        plot_feature_screen(screen),
    ]
    v283_guard = read_json_optional(V283_GUARDRAIL)
    guardrail = {
        "pass": True,
        "zip_testzip": False,
        "event_n": int(events["event_uid"].nunique()),
        "candidate_rows": int(len(cand)),
        "feature_set_n": int(len(feature_sets)),
        "all_candidate_feature_n": int(len(all_cols)),
        "dynamic_candidate_feature_n": int(len(dynamic_cols)),
        "fixed_wait_latest_badtop10": FIXED_WAIT_LATEST_BADTOP10,
        "route_viable_now": bool(decision["route_viable_now"].iloc[0]),
        "deployable_top1_badtop10_pass": bool(
            decision.loc[decision["check"].eq("deployable_top1_val_chosen_bad_top10"), "pass"].iloc[0]
        ),
        "deployable_top1_bad_ambiguous_pass": bool(
            decision.loc[decision["check"].eq("deployable_top1_val_chosen_bad_ambiguous"), "pass"].iloc[0]
        ),
        "test_best_top1_diagnostic_pass": bool(
            decision.loc[decision["check"].eq("test_best_top1_diagnostic_beats_latest"), "pass"].iloc[0]
        ),
        "v283_old_route_closed": bool(v283_guard.get("old_feature_selector_route_closed", False)),
    }
    write_input_hashes()
    write_file_inventory()
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(screen, feature_audit, summary, val_test, decision, figures, guardrail)
    write_file_inventory()
    first_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(first_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    report = write_report(screen, feature_audit, summary, val_test, decision, figures, guardrail)
    write_file_inventory()
    second_zip_ok = make_zip()
    guardrail["zip_testzip"] = bool(second_zip_ok)
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[v284] 完成。", flush=True)
    print(f"[v284] report={report}", flush=True)
    print(f"[v284] zip={ZIP_PATH}", flush=True)
    print(json.dumps(guardrail, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
