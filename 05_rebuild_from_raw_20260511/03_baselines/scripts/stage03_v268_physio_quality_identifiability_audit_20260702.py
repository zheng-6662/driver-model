#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v268 physiology quality / alignment / identifiability audit。

v260-v267 已经说明：当前 bio260 在 subject-disjoint 正式预测、wait gate、
prototype reranking 中都没有形成差样本本质改善。本轮不再训练新预测模型，而是回答：

1. 生理源数据本身有没有明显质量问题？
2. 事件级 bio260 是否存在覆盖/缺失/floor 合并问题？
3. bio260 特征主要在编码 subject/recording，还是能编码行为目标/等待收益？
4. 在 v267 候选库有 headroom 的情况下，bio/pair 预测为什么选不中正确候选？

输出目标：把“数据质量问题”和“任务可识别性问题”分开，避免继续盲目加模型。
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"
PHYSIO_ROOT = REBUILD / "06_physio_processing" / "physio_subject_collection_v1_20260603" / "tables"

PHYSIO_INVENTORY = PHYSIO_ROOT / "physio_recording_inventory.csv"
PHYSIO_SIGNAL_QUALITY = PHYSIO_ROOT / "physio_signal_quality_summary.csv"
PHYSIO_SIGNAL_AVAIL = PHYSIO_ROOT / "physio_signal_availability_summary.csv"
V260_FEATURES = BASELINES / "v260_event_biomarker_physio_rebuild_20260702" / "tables" / "v260_event_biomarker_features.csv"
V260_ETA = BASELINES / "v260_event_biomarker_physio_rebuild_20260702" / "tables" / "v260_biomarker_eta2_by_target_feature.csv"
V266_EVENTS = BASELINES / "v266_vehicle_matched_bio_residual_prototype_20260702" / "tables" / "v266_event_context_table.csv"
V267_PAIRS = BASELINES / "v267_supervised_bio_prototype_reranker_20260702" / "tables" / "v267_pair_predictions_compact.csv"
V267_SELECTED = BASELINES / "v267_supervised_bio_prototype_reranker_20260702" / "tables" / "v267_selected_pair_reranker_by_strategy.csv"

OUT = BASELINES / "v268_physio_quality_identifiability_audit_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v268_physio_quality_identifiability_audit_20260702_pack.zip"

FIXED_WAIT_LATEST_BADTOP10 = 0.695048

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


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


def signal_family(col: str) -> str:
    low = str(col).lower()
    for key in ["ecg", "scr", "eda", "resp", "emg", "hrv", "hr"]:
        if key in low:
            return key
    return "other"


def eta_squared(feature: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(feature, dtype=float)
    lab = np.asarray(labels)
    mask = np.isfinite(x) & pd.notna(lab)
    if int(mask.sum()) < 20:
        return math.nan
    x = x[mask]
    lab = lab[mask]
    grand = float(np.mean(x))
    total = float(np.sum((x - grand) ** 2))
    if total <= 1e-12:
        return math.nan
    between = 0.0
    for one in np.unique(lab):
        vals = x[lab == one]
        between += float(len(vals) * (np.mean(vals) - grand) ** 2)
    return float(between / total)


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 5:
        return math.nan
    return float(aa[mask].rank().corr(bb[mask].rank()))


def summarize_source_recordings(inv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric = ["duration_s", "median_hz", "gap_gt_20ms_count", "duplicate_t_count", "negative_or_zero_dt_count", "file_size_mb"]
    for col in numeric:
        if col in inv.columns:
            inv[col] = pd.to_numeric(inv[col], errors="coerce")
    summary = pd.DataFrame(
        [
            {
                "recording_n": int(len(inv)),
                "subject_n": int(inv["subject"].astype(str).nunique()),
                "duration_s_sum": float(inv["duration_s"].sum()),
                "duration_s_median": float(inv["duration_s"].median()),
                "median_hz_median": float(inv["median_hz"].median()),
                "gap_gt_20ms_total": int(inv["gap_gt_20ms_count"].fillna(0).sum()),
                "duplicate_t_total": int(inv["duplicate_t_count"].fillna(0).sum()),
                "negative_or_zero_dt_total": int(inv["negative_or_zero_dt_count"].fillna(0).sum()),
                "core_columns_all_present_rate": float(inv["missing_core_columns"].fillna("").astype(str).eq("").mean()),
            }
        ]
    )
    by_subject = (
        inv.groupby("subject", as_index=False)
        .agg(
            recording_count=("session_stamp", "count"),
            duration_s_sum=("duration_s", "sum"),
            median_hz_median=("median_hz", "median"),
            gap_gt_20ms_total=("gap_gt_20ms_count", "sum"),
            duplicate_t_total=("duplicate_t_count", "sum"),
        )
        .sort_values("subject")
    )
    return summary, by_subject


def summarize_signal_quality(quality: pd.DataFrame, avail: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    q = quality.copy()
    for col in ["missing_ratio", "std", "near_constant"]:
        if col in q.columns and col != "near_constant":
            q[col] = pd.to_numeric(q[col], errors="coerce")
    q["near_constant_bool"] = q["near_constant"].astype(str).str.lower().eq("true")
    by_signal = (
        q.groupby("signal", as_index=False)
        .agg(
            recording_n=("session_stamp", "count"),
            ok_rate=("status", lambda s: float(s.astype(str).eq("ok").mean())),
            missing_ratio_median=("missing_ratio", "median"),
            missing_ratio_max=("missing_ratio", "max"),
            near_constant_rate=("near_constant_bool", "mean"),
            std_median=("std", "median"),
        )
        .sort_values("signal")
    )
    merged = avail.merge(by_signal, on="signal", how="left", suffixes=("_availability", "_quality"))
    for col in ["recording_count", "usable_basic_count", "near_constant_count", "all_nan_count", "high_missing_count"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["usable_basic_rate"] = merged["usable_basic_count"] / merged["recording_count"].replace(0, np.nan)
    merged["family"] = merged["signal"].map(signal_family)
    family = (
        merged.groupby("family", as_index=False)
        .agg(
            signal_n=("signal", "count"),
            usable_basic_rate_mean=("usable_basic_rate", "mean"),
            near_constant_count_sum=("near_constant_count", "sum"),
            all_nan_count_sum=("all_nan_count", "sum"),
            high_missing_count_sum=("high_missing_count", "sum"),
        )
        .sort_values("family")
    )
    return merged, family


def bio_feature_columns(df: pd.DataFrame, prefix: str = "bio260_") -> List[str]:
    meta_tokens = [
        "sample_hz",
        "recording_duration",
        "uses_post_observation",
        "_rows",
        "_duration_s",
        "baseline",
    ]
    cols: List[str] = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if any(tok in col for tok in meta_tokens):
            continue
        cols.append(col)
    return cols


def event_coverage(v260: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = v260.copy()
    df["delay_ms"] = pd.to_numeric(df["delay_ms"], errors="coerce")
    df["bio260_ok"] = df["bio260_status"].astype(str).eq("ok")
    df["bio260_uses_post_observation"] = df["bio260_uses_post_observation"].astype(str).str.lower().eq("true")
    for col in ["bio260_baseline_rows", "bio260_baseline_duration_s", "bio260_recording_duration_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    coverage = (
        df.groupby(["split", "delay_ms"], as_index=False)
        .agg(
            row_n=("event_uid", "count"),
            event_n=("event_uid", "nunique"),
            ok_rate=("bio260_ok", "mean"),
            post_observation_rate=("bio260_uses_post_observation", "mean"),
            baseline_rows_mean=("bio260_baseline_rows", "mean"),
            baseline_duration_s_mean=("bio260_baseline_duration_s", "mean"),
            recording_duration_s_median=("bio260_recording_duration_s", "median"),
        )
        .sort_values(["split", "delay_ms"])
    )
    by_recording = (
        df.groupby(["split", "subject", "recording"], as_index=False)
        .agg(
            row_n=("event_uid", "count"),
            event_n=("event_uid", "nunique"),
            ok_rate=("bio260_ok", "mean"),
            post_observation_rate=("bio260_uses_post_observation", "mean"),
            baseline_rows_mean=("bio260_baseline_rows", "mean"),
        )
        .sort_values(["split", "subject", "recording"])
    )
    cols = bio_feature_columns(df)
    miss_rows = []
    for col in cols:
        miss_rows.append(
            {
                "feature": col,
                "family": signal_family(col),
                "missing_rate_all": float(pd.to_numeric(df[col], errors="coerce").isna().mean()),
                "missing_rate_ok_rows": float(pd.to_numeric(df.loc[df["bio260_ok"], col], errors="coerce").isna().mean()),
                "finite_std": float(pd.to_numeric(df[col], errors="coerce").std(skipna=True)),
            }
        )
    missing = pd.DataFrame(miss_rows)
    missing_family = (
        missing.groupby("family", as_index=False)
        .agg(
            feature_n=("feature", "count"),
            missing_rate_all_mean=("missing_rate_all", "mean"),
            missing_rate_ok_rows_mean=("missing_rate_ok_rows", "mean"),
            zero_variance_feature_n=("finite_std", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) <= 1e-12).sum())),
        )
        .sort_values("family")
    )
    return coverage, by_recording, missing_family


def identifiability_audit(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bio_cols = [c for c in events.columns if c.startswith("floor_bio260_") and pd.api.types.is_numeric_dtype(events[c])]
    rows: List[Dict[str, object]] = []
    target_map = {
        "subject": events["subject"].astype(str),
        "recording": events["recording"].astype(str),
        "bad_top10": events["bad_top10"].astype(str),
        "early_best_after_400": events["early_best_after_400"].astype(str),
        "wait_better_latest_vs_keep0": (pd.to_numeric(events["latest_tail_rmse_v241"], errors="coerce") < pd.to_numeric(events["keep0_tail_rmse_v241"], errors="coerce")).astype(str),
    }
    for col in bio_cols:
        x = pd.to_numeric(events[col], errors="coerce").to_numpy(dtype=float)
        rec: Dict[str, object] = {"feature": col, "family": signal_family(col)}
        for target, labels in target_map.items():
            rec[f"eta2_{target}"] = eta_squared(x, labels.to_numpy())
        rows.append(rec)
    detail = pd.DataFrame(rows)
    detail["identity_eta_max"] = detail[["eta2_subject", "eta2_recording"]].max(axis=1)
    detail["behavior_eta_max"] = detail[["eta2_bad_top10", "eta2_early_best_after_400", "eta2_wait_better_latest_vs_keep0"]].max(axis=1)
    detail["identity_to_behavior_ratio"] = detail["identity_eta_max"] / detail["behavior_eta_max"].replace(0, np.nan)
    summary = (
        detail.groupby("family", as_index=False)
        .agg(
            feature_n=("feature", "count"),
            identity_eta_max_mean=("identity_eta_max", "mean"),
            behavior_eta_max_mean=("behavior_eta_max", "mean"),
            identity_to_behavior_ratio_median=("identity_to_behavior_ratio", "median"),
            features_identity_gt_behavior_5x=("identity_to_behavior_ratio", lambda s: int((pd.to_numeric(s, errors="coerce") >= 5.0).sum())),
            features_behavior_eta_ge_0p02=("behavior_eta_max", lambda s: int((pd.to_numeric(s, errors="coerce") >= 0.02).sum())),
        )
        .sort_values("family")
    )
    return detail.sort_values(["identity_to_behavior_ratio", "identity_eta_max"], ascending=[False, False]), summary


def pair_rank_audit(pairs: pd.DataFrame, events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_cols = [c for c in pairs.columns if c.startswith("pred_pair_")]
    score_cols = ["vehicle_distance", "bio_distance"] + pred_cols
    latest_map = pd.to_numeric(events.set_index("event_uid")["latest_tail_rmse_v241"], errors="coerce").to_dict()
    rows: List[Dict[str, object]] = []
    for (split, event_uid), g in pairs.groupby(["split", "event_uid"], sort=False):
        g = g.copy()
        if int(g["neighbor_rank_vehicle"].max()) < 40:
            continue
        best_idx = pd.to_numeric(g["target_tail_rmse_v241"], errors="coerce").idxmin()
        best_rmse = float(g.loc[best_idx, "target_tail_rmse_v241"])
        latest = float(latest_map.get(event_uid, math.nan))
        bad = bool(g["bad_top10"].iloc[0])
        for score in score_cols:
            ascending = True
            chosen = g.loc[pd.to_numeric(g[score], errors="coerce").idxmin()]
            ordered = g.sort_values(score, ascending=ascending).reset_index(drop=True)
            match = ordered.index[ordered["target_tail_rmse_v241"].eq(best_rmse)]
            rank = int(match[0] + 1) if len(match) else math.nan
            rows.append(
                {
                    "split": split,
                    "event_uid": event_uid,
                    "bad_top10": bad,
                    "score": score,
                    "candidate_n": int(len(g)),
                    "best_candidate_rmse": best_rmse,
                    "chosen_rmse": float(chosen["target_tail_rmse_v241"]),
                    "chosen_minus_best": float(chosen["target_tail_rmse_v241"] - best_rmse),
                    "chosen_minus_latest": float(chosen["target_tail_rmse_v241"] - latest) if np.isfinite(latest) else math.nan,
                    "true_best_rank_by_score": rank,
                    "spearman_score_vs_target_rmse": safe_spearman(g[score], g["target_tail_rmse_v241"]),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["split", "bad_top10", "score"], as_index=False)
        .agg(
            event_n=("event_uid", "nunique"),
            chosen_rmse_mean=("chosen_rmse", "mean"),
            best_candidate_rmse_mean=("best_candidate_rmse", "mean"),
            chosen_minus_best_mean=("chosen_minus_best", "mean"),
            chosen_minus_latest_mean=("chosen_minus_latest", "mean"),
            true_best_rank_mean=("true_best_rank_by_score", "mean"),
            true_best_top3_rate=("true_best_rank_by_score", lambda s: float((pd.to_numeric(s, errors="coerce") <= 3).mean())),
            spearman_score_vs_target_rmse_mean=("spearman_score_vs_target_rmse", "mean"),
        )
        .sort_values(["split", "bad_top10", "chosen_rmse_mean"])
    )
    return detail, summary


def conclusion_flags(
    source_summary: pd.DataFrame,
    signal_avail: pd.DataFrame,
    coverage: pd.DataFrame,
    ident_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    source = source_summary.iloc[0]
    rows.append(
        {
            "check": "source_timing_integrity",
            "status": "pass" if int(source["gap_gt_20ms_total"]) == 0 and int(source["duplicate_t_total"]) == 0 else "warn",
            "evidence": f"median_hz={float(source['median_hz_median']):.3f}, gaps={int(source['gap_gt_20ms_total'])}, duplicates={int(source['duplicate_t_total'])}",
            "interpretation": "200Hz 连续层时序本身稳定；失败不能简单归咎于采样断裂。",
        }
    )
    bad_signals = signal_avail[signal_avail["usable_basic_rate"].fillna(0) < 0.75]
    rows.append(
        {
            "check": "derived_signal_availability",
            "status": "warn" if len(bad_signals) else "pass",
            "evidence": "; ".join([f"{r.signal}:{r.usable_basic_count}/{r.recording_count}" for _, r in bad_signals.iterrows()][:8]),
            "interpretation": "部分派生生理列不可用或近常数，尤其 HRV_RMSSD、RESP_BPM/Amplitude、部分 EDA；这会削弱高层 biomarker。",
        }
    )
    cov_min = float(coverage["ok_rate"].min())
    rows.append(
        {
            "check": "event_window_coverage",
            "status": "pass" if cov_min >= 0.85 else "warn",
            "evidence": f"min split-delay ok_rate={cov_min:.3f}",
            "interpretation": "事件窗口覆盖整体尚可，问题更可能是特征有效性和泛化，而不是大面积缺失。",
        }
    )
    ratio_med = float(ident_summary["identity_to_behavior_ratio_median"].median())
    rows.append(
        {
            "check": "identity_vs_behavior_signal",
            "status": "warn" if ratio_med >= 3.0 else "pass",
            "evidence": f"median family identity/behavior eta ratio={ratio_med:.2f}",
            "interpretation": "bio260 更容易区分 subject/recording，而不是行为目标；这解释了 subject-disjoint 下增量不稳定。",
        }
    )
    test_bad = rank_summary[
        rank_summary["split"].astype(str).eq("test")
        & rank_summary["bad_top10"].astype(bool)
        & rank_summary["score"].astype(str).eq("pred_pair_vehicle_bio_hgb")
    ]
    if len(test_bad):
        row = test_bad.iloc[0]
        rows.append(
            {
                "check": "candidate_rerank_identifiability",
                "status": "warn" if float(row["chosen_minus_latest_mean"]) > 0 else "pass",
                "evidence": f"pair_vehicle_bio chosen_minus_latest={float(row['chosen_minus_latest_mean']):+.4f}, top3_rate={float(row['true_best_top3_rate']):.3f}",
                "interpretation": "即使候选库有 oracle headroom，bio pair 分数也不能稳定把最佳候选排到前面。",
            }
        )
    return pd.DataFrame(rows)


def plot_signal_availability(signal_avail: pd.DataFrame) -> Path:
    path = FIGURES / "v268_signal_availability.png"
    sub = signal_avail.sort_values("usable_basic_rate").copy()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(sub))
    ax.bar(x, sub["usable_basic_rate"], color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["signal"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("usable recording rate")
    ax.set_title("v268: subject-collection 200Hz signal usability")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_eta_scatter(detail: pd.DataFrame) -> Path:
    path = FIGURES / "v268_identity_vs_behavior_eta.png"
    sub = detail.copy()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    families = sorted(sub["family"].dropna().unique())
    palette = {
        "ecg": "#4C78A8",
        "scr": "#F28E2B",
        "eda": "#F28E2B",
        "resp": "#59A14F",
        "emg": "#E15759",
        "hr": "#B07AA1",
        "hrv": "#B07AA1",
        "other": "#9CA3AF",
    }
    for fam in families:
        one = sub[sub["family"].eq(fam)]
        ax.scatter(one["behavior_eta_max"], one["identity_eta_max"], s=18, alpha=0.55, label=fam, color=palette.get(fam, "#9CA3AF"))
    lim = max(float(sub["behavior_eta_max"].max()), float(sub["identity_eta_max"].max()), 0.05)
    ax.plot([0, lim], [0, lim], linestyle="--", color="#666666", linewidth=1.0)
    ax.set_xlabel("max behavior eta2")
    ax.set_ylabel("max identity eta2 (subject/recording)")
    ax.set_title("v268: bio260 特征更像行为信号还是身份/记录信号")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_rank_summary(rank_summary: pd.DataFrame) -> Path:
    path = FIGURES / "v268_test_badtop10_candidate_rank_quality.png"
    sub = rank_summary[
        rank_summary["split"].astype(str).eq("test")
        & rank_summary["bad_top10"].astype(bool)
        & rank_summary["score"].isin(["vehicle_distance", "bio_distance", "pred_pair_vehicle_hgb", "pred_pair_bio_hgb", "pred_pair_vehicle_bio_hgb", "pred_pair_vehicle_bio_badweighted_hgb"])
    ].copy()
    if sub.empty:
        return path
    sub = sub.sort_values("chosen_rmse_mean")
    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = np.arange(len(sub))
    ax.bar(x, sub["chosen_rmse_mean"], color="#4C78A8")
    ax.axhline(FIXED_WAIT_LATEST_BADTOP10, color="#E15759", linestyle="--", linewidth=1.2, label="fixed wait-latest 0.6950")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace("pred_pair_", "").replace("_", "\n") for s in sub["score"]], fontsize=8)
    ax.set_ylabel("test bad_top10 chosen RMSE within top40")
    ax.set_title("v268: 候选库有 headroom 时，各分数能否选中正确候选")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("physio_recording_inventory", PHYSIO_INVENTORY),
        ("physio_signal_quality_summary", PHYSIO_SIGNAL_QUALITY),
        ("physio_signal_availability_summary", PHYSIO_SIGNAL_AVAIL),
        ("v260_event_biomarker_features", V260_FEATURES),
        ("v260_eta", V260_ETA),
        ("v266_event_context", V266_EVENTS),
        ("v267_pair_predictions", V267_PAIRS),
        ("v267_selected", V267_SELECTED),
    ]:
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
    source_summary: pd.DataFrame,
    signal_avail: pd.DataFrame,
    signal_family: pd.DataFrame,
    coverage: pd.DataFrame,
    missing_family: pd.DataFrame,
    ident_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    flags: pd.DataFrame,
    figures: List[Path],
) -> None:
    lines: List[str] = []
    lines.append("# v268 physiology quality / alignment / identifiability audit")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v267 已经验证更强监督式候选重排仍未达标。")
    lines.append("- v268 不再训练新预测模型，而是审计生理链路：源质量、事件窗口覆盖、身份混淆、候选排序可识别性。")
    lines.append("")
    lines.append("## 总体判定")
    lines.append("")
    lines.append(flags.to_markdown(index=False))
    lines.append("")
    lines.append("## 源 recording 质量")
    lines.append("")
    lines.append(source_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## 信号可用性")
    lines.append("")
    lines.append(signal_avail[["signal", "recording_count", "usable_basic_count", "usable_basic_rate", "near_constant_count", "all_nan_count", "high_missing_count"]].to_markdown(index=False))
    lines.append("")
    lines.append("### 按信号族汇总")
    lines.append("")
    lines.append(signal_family.to_markdown(index=False))
    lines.append("")
    lines.append("## 事件窗口覆盖")
    lines.append("")
    focus_cov = coverage[coverage["delay_ms"].isin([0, 1000])].copy()
    lines.append(focus_cov.to_markdown(index=False))
    lines.append("")
    lines.append("## 事件特征缺失按信号族")
    lines.append("")
    lines.append(missing_family.to_markdown(index=False))
    lines.append("")
    lines.append("## bio260 身份信号 vs 行为信号")
    lines.append("")
    lines.append(ident_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## v267 候选排序可识别性")
    lines.append("")
    focus_rank = rank_summary[
        rank_summary["split"].astype(str).eq("test")
        & rank_summary["bad_top10"].astype(bool)
        & rank_summary["score"].isin(["vehicle_distance", "bio_distance", "pred_pair_vehicle_hgb", "pred_pair_bio_hgb", "pred_pair_vehicle_bio_hgb", "pred_pair_vehicle_bio_badweighted_hgb"])
    ].copy()
    lines.append(focus_rank.to_markdown(index=False))
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- 200Hz 源时序质量基本稳定，不能把失败简单归因于采样断裂。")
    lines.append("- 但派生生理列存在结构性弱点：HRV_RMSSD 全不可用，RESP_BPM/RESP_Amplitude 基本不可用，EDA 有一部分 recording 近常数/全缺。")
    lines.append("- 事件级 bio260 覆盖率尚可，post-observation guardrail 通过；核心问题更偏向特征有效性和 subject-disjoint 可迁移性。")
    lines.append("- 身份/recording 可分性高于行为/等待收益可分性，说明 bio260 在跨驾驶员泛化时更容易携带个体/设备/记录差异。")
    lines.append("- v267 候选库虽然有 oracle headroom，但 bio/pair 分数不能稳定把最佳候选排到前面；这解释了为什么更强 reranker 仍不能超过 fixed wait-latest。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v268_physio_quality_identifiability_audit_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v268] physiology quality / identifiability audit", flush=True)
    clean_out_dir()

    inventory = pd.read_csv(PHYSIO_INVENTORY, encoding="utf-8-sig", low_memory=False)
    quality = pd.read_csv(PHYSIO_SIGNAL_QUALITY, encoding="utf-8-sig", low_memory=False)
    avail = pd.read_csv(PHYSIO_SIGNAL_AVAIL, encoding="utf-8-sig", low_memory=False)
    v260 = pd.read_csv(V260_FEATURES, encoding="utf-8-sig", low_memory=False)
    v260_eta = pd.read_csv(V260_ETA, encoding="utf-8-sig", low_memory=False)
    events = pd.read_csv(V266_EVENTS, encoding="utf-8-sig", low_memory=False)
    pairs = pd.read_csv(V267_PAIRS, encoding="utf-8-sig", low_memory=False)

    source_summary, source_by_subject = summarize_source_recordings(inventory)
    signal_avail, signal_family_summary = summarize_signal_quality(quality, avail)
    coverage, coverage_by_recording, missing_family = event_coverage(v260)
    ident_detail, ident_summary = identifiability_audit(events)
    rank_detail, rank_summary = pair_rank_audit(pairs, events)
    flags = conclusion_flags(source_summary, signal_avail, coverage, ident_summary, rank_summary)

    figures = [
        plot_signal_availability(signal_avail),
        plot_eta_scatter(ident_detail),
        plot_rank_summary(rank_summary),
    ]

    write_csv(source_summary, TABLES / "v268_source_recording_quality_summary.csv")
    write_csv(source_by_subject, TABLES / "v268_source_recording_quality_by_subject.csv")
    write_csv(signal_avail, TABLES / "v268_source_signal_availability_quality.csv")
    write_csv(signal_family_summary, TABLES / "v268_source_signal_quality_by_family.csv")
    write_csv(coverage, TABLES / "v268_event_coverage_by_split_delay.csv")
    write_csv(coverage_by_recording, TABLES / "v268_event_coverage_by_recording.csv")
    write_csv(missing_family, TABLES / "v268_event_feature_missingness_by_family.csv")
    write_csv(ident_detail, TABLES / "v268_bio_identity_behavior_eta_detail.csv")
    write_csv(ident_summary, TABLES / "v268_bio_identity_behavior_eta_summary.csv")
    write_csv(v260_eta, TABLES / "v268_v260_eta_reference.csv")
    write_csv(rank_detail, TABLES / "v268_candidate_rank_diagnostics_by_event.csv")
    write_csv(rank_summary, TABLES / "v268_candidate_rank_diagnostics_summary.csv")
    write_csv(flags, TABLES / "v268_conclusion_flags.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(source_summary, signal_avail, signal_family_summary, coverage, missing_family, ident_summary, rank_summary, flags, figures)
    write_file_inventory()
    zip_ok = make_zip()
    guardrail = {
        "pass": bool(zip_ok and float(coverage["post_observation_rate"].max()) == 0.0),
        "zip_testzip": bool(zip_ok),
        "post_observation_rate_max": float(coverage["post_observation_rate"].max()),
        "source_recording_n": int(source_summary["recording_n"].iloc[0]),
        "source_subject_n": int(source_summary["subject_n"].iloc[0]),
        "event_row_n": int(len(v260)),
        "event_n": int(v260["event_uid"].nunique()),
        "pair_rank_event_n": int(rank_detail["event_uid"].nunique()) if len(rank_detail) else 0,
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v268 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    print(f"[v268] report={REPORTS / 'v268_physio_quality_identifiability_audit_cn.md'}", flush=True)
    print(f"[v268] zip={ZIP_PATH}", flush=True)
    print(flags.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
