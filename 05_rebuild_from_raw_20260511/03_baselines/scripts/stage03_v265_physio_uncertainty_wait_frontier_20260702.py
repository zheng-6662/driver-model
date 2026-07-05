#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v265 physiology uncertainty / wait frontier。

v260-v264 显示：生理直接预测、anchor selector、wait gate、online physiology KNN 都没有
形成本质改善。本轮验证最后一个较合理用途：

    生理是否能作为“不确定性/风险校准信号”，在固定等待预算下更准确地挑出需要多观察的样本？

方法边界：
- 只训练风险/收益分数，不输出新轨迹；
- policy 仍只在 0ms keep 与 1000ms wait-latest 之间选择；
- 所有模型只在 train split 拟合；
- 各等待比例阈值只在 val split 定标；
- test 只报告，不用 test 调阈值。

如果 vehicle+bio 分数在同等 wait_rate 下不能稳定优于 vehicle 分数，则说明当前生理也不能
作为可部署的不确定性校准增量。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(r"F:\data_set_process\data_process")
REBUILD = ROOT / "05_rebuild_from_raw_20260511"
BASELINES = REBUILD / "03_baselines"

V258_SCRIPT = BASELINES / "scripts" / "stage03_v258_physio_augmented_anchor_selector_20260702.py"
V263_EVENTS = (
    BASELINES
    / "v263_bio260_wait_gate_20260702"
    / "tables"
    / "v263_event_wait_gate_predictions.csv"
)
V262_FEATURE_SELECTION = (
    BASELINES
    / "v262_subject_invariant_bio260_selector_20260702"
    / "tables"
    / "v262_feature_selection_audit.csv"
)

OUT = BASELINES / "v265_physio_uncertainty_wait_frontier_20260702"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"
LOGS = OUT / "logs"
ZIP_PATH = BASELINES / "v265_physio_uncertainty_wait_frontier_20260702_pack.zip"

SEED = 26502
WAIT_RATES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


def load_v258_module():
    if not V258_SCRIPT.exists():
        raise FileNotFoundError(f"缺少 v258 脚本：{V258_SCRIPT}")
    spec = importlib.util.spec_from_file_location("v258_anchor_selector", V258_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 v258 脚本：{V258_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.TABLES = TABLES
    module.FIGURES = FIGURES
    module.REPORTS = REPORTS
    module.LOGS = LOGS
    module.ZIP_PATH = ZIP_PATH
    module.SEED = SEED
    return module


V258 = load_v258_module()


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


def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    vehicle_prefixes = ("candidate_delay", "hist_", "road_", "phase_", "instability_penalty")
    vehicle_cols: List[str] = []
    for col in df.columns:
        if col in {"candidate_delay_ms", "candidate_delay_s"} or col.startswith(vehicle_prefixes):
            if pd.api.types.is_numeric_dtype(df[col]):
                vehicle_cols.append(col)
    fs = pd.read_csv(V262_FEATURE_SELECTION, encoding="utf-8-sig", low_memory=False)
    sp64 = fs[
        fs["row_type"].astype(str).eq("feature")
        & fs["in_sp64"].astype(str).str.lower().eq("true")
    ]["column"].dropna().astype(str).tolist()
    bio_cols = [col for col in sp64 if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if "bio260_floor_status_ok" in df.columns:
        bio_cols.append("bio260_floor_status_ok")
    return list(dict.fromkeys(vehicle_cols)), list(dict.fromkeys(bio_cols))


def load_events() -> pd.DataFrame:
    if not V263_EVENTS.exists():
        raise FileNotFoundError(f"缺少 v263 event table：{V263_EVENTS}")
    df = pd.read_csv(V263_EVENTS, encoding="utf-8-sig", low_memory=False)
    df["target_gain_latest_vs_keep0"] = pd.to_numeric(df["gain_latest_vs_keep0"], errors="coerce")
    df["target_keep0_tail_rmse"] = pd.to_numeric(df["keep0_tail_rmse_v241"], errors="coerce")
    df["target_latest_tail_rmse"] = pd.to_numeric(df["latest_tail_rmse_v241"], errors="coerce")
    df["target_oracle_gap_after_latest"] = (
        pd.to_numeric(df["latest_tail_rmse_v241"], errors="coerce")
        - pd.to_numeric(df["oracle_tail_rmse_v241"], errors="coerce")
    )
    df["target_wait_better"] = (
        pd.to_numeric(df["latest_tail_rmse_v241"], errors="coerce")
        < pd.to_numeric(df["keep0_tail_rmse_v241"], errors="coerce")
    ).astype(float)
    df["target_bad_top10"] = df["bad_top10_split_v241"].fillna(False).astype(bool).astype(float)
    return df


def fit_scores(events: pd.DataFrame, vehicle_cols: List[str], bio_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = events["split"].astype(str).eq("train").to_numpy()
    model_specs = [
        ("score_vehicle_gain", vehicle_cols, "target_gain_latest_vs_keep0"),
        ("score_vehicle_bio_gain", vehicle_cols + bio_cols, "target_gain_latest_vs_keep0"),
        ("score_bio_only_gain", bio_cols, "target_gain_latest_vs_keep0"),
        ("score_vehicle_keep0_risk", vehicle_cols, "target_keep0_tail_rmse"),
        ("score_vehicle_bio_keep0_risk", vehicle_cols + bio_cols, "target_keep0_tail_rmse"),
        ("score_bio_only_keep0_risk", bio_cols, "target_keep0_tail_rmse"),
        ("score_vehicle_badprob", vehicle_cols, "target_bad_top10"),
        ("score_vehicle_bio_badprob", vehicle_cols + bio_cols, "target_bad_top10"),
        ("score_bio_only_badprob", bio_cols, "target_bad_top10"),
        ("score_vehicle_oracle_gap", vehicle_cols, "target_oracle_gap_after_latest"),
        ("score_vehicle_bio_oracle_gap", vehicle_cols + bio_cols, "target_oracle_gap_after_latest"),
    ]
    audits = []
    out = events.copy()
    for name, cols, target in model_specs:
        pred, audit = V258.train_model(out, cols, target, train_mask)
        out[name] = pred
        audits.append(
            {
                "score": name,
                "target": target,
                "feature_n": len(cols),
                "vehicle_feature_n": sum(col in vehicle_cols for col in cols),
                "bio260_feature_n": sum(col in bio_cols for col in cols),
            }
        )
    return out, pd.DataFrame(audits)


def safe_auc(y: pd.Series, score: pd.Series) -> float:
    yy = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    ss = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yy) & np.isfinite(ss)
    if mask.sum() < 2 or len(np.unique(yy[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(yy[mask], ss[mask]))


def score_diagnostics(events: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split in ["train", "val", "test"]:
        sub = events[events["split"].astype(str).eq(split)].copy()
        for score in score_cols:
            rows.append(
                {
                    "split": split,
                    "score": score,
                    "n": int(len(sub)),
                    "auc_wait_better": safe_auc(sub["target_wait_better"], sub[score]),
                    "auc_bad_top10": safe_auc(sub["target_bad_top10"], sub[score]),
                    "spearman_gain": float(sub[[score, "target_gain_latest_vs_keep0"]].corr(method="spearman").iloc[0, 1]),
                    "spearman_keep0_rmse": float(sub[[score, "target_keep0_tail_rmse"]].corr(method="spearman").iloc[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def threshold_for_wait_rate(val_scores: pd.Series, wait_rate: float) -> float:
    vals = pd.to_numeric(val_scores, errors="coerce").to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return float("inf")
    if wait_rate >= 1.0:
        return float(np.nanmin(finite) - 1e-9)
    if wait_rate <= 0.0:
        return float(np.nanmax(finite) + 1e-9)
    return float(np.quantile(finite, 1.0 - wait_rate))


def make_selected(events: pd.DataFrame, score: str, wait_rate: float, threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in events.iterrows():
        choose_latest = float(row[score]) >= threshold
        selected_delay = 1000 if choose_latest else 0
        selected_rmse = float(row["latest_tail_rmse_v241"] if choose_latest else row["keep0_tail_rmse_v241"])
        rows.append(
            {
                "strategy": f"{score}_wr{wait_rate:.2f}",
                "score": score,
                "target_wait_rate": float(wait_rate),
                "threshold_from_val": float(threshold),
                "event_uid": str(row["event_uid"]),
                "split": str(row["split"]),
                "subject": str(row.get("subject", "")),
                "recording": str(row.get("recording", "")),
                "selected_delay_ms": selected_delay,
                "selected_tail_rmse_v241": selected_rmse,
                "keep0_tail_rmse_v241": float(row["keep0_tail_rmse_v241"]),
                "latest_tail_rmse_v241": float(row["latest_tail_rmse_v241"]),
                "oracle_tail_rmse_v241": float(row["oracle_tail_rmse_v241"]),
                "delta_selected_minus_keep0": float(selected_rmse - row["keep0_tail_rmse_v241"]),
                "delta_selected_minus_latest": float(selected_rmse - row["latest_tail_rmse_v241"]),
                "bad_top10": bool(row["bad_top10_split_v241"]),
                "very_bad_top5": bool(row["very_bad_top5_split_v241"]),
                "normal": bool(row["normal_curve_current0"]),
                "observe_later_like": bool(row["observe_later_like_current0"]),
                "strong_steer": bool(row["strong_steer_current0"]),
                "reverse": bool(row["reverse_current0"]),
                "early_best_after_400": bool(int(row["oracle_delay_ms"]) >= 400),
            }
        )
    return pd.DataFrame(rows)


def baseline_selected(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for label, delay_col, rmse_col in [
        ("policy_keep_0ms_anchor", 0, "keep0_tail_rmse_v241"),
        ("policy_wait_to_latest_anchor", 1000, "latest_tail_rmse_v241"),
    ]:
        df = events.copy()
        out = pd.DataFrame(
            {
                "strategy": label,
                "score": label,
                "target_wait_rate": 0.0 if delay_col == 0 else 1.0,
                "threshold_from_val": np.nan,
                "event_uid": df["event_uid"].astype(str),
                "split": df["split"].astype(str),
                "subject": df["subject"].astype(str),
                "recording": df["recording"].astype(str),
                "selected_delay_ms": delay_col,
                "selected_tail_rmse_v241": pd.to_numeric(df[rmse_col], errors="coerce"),
                "keep0_tail_rmse_v241": pd.to_numeric(df["keep0_tail_rmse_v241"], errors="coerce"),
                "latest_tail_rmse_v241": pd.to_numeric(df["latest_tail_rmse_v241"], errors="coerce"),
                "oracle_tail_rmse_v241": pd.to_numeric(df["oracle_tail_rmse_v241"], errors="coerce"),
                "bad_top10": df["bad_top10_split_v241"].fillna(False).astype(bool),
                "very_bad_top5": df["very_bad_top5_split_v241"].fillna(False).astype(bool),
                "normal": df["normal_curve_current0"].fillna(False).astype(bool),
                "observe_later_like": df["observe_later_like_current0"].fillna(False).astype(bool),
                "strong_steer": df["strong_steer_current0"].fillna(False).astype(bool),
                "reverse": df["reverse_current0"].fillna(False).astype(bool),
                "early_best_after_400": pd.to_numeric(df["oracle_delay_ms"], errors="coerce").fillna(0).astype(int) >= 400,
            }
        )
        out["delta_selected_minus_keep0"] = out["selected_tail_rmse_v241"] - out["keep0_tail_rmse_v241"]
        out["delta_selected_minus_latest"] = out["selected_tail_rmse_v241"] - out["latest_tail_rmse_v241"]
        rows.append(out)
    oracle = events.copy()
    out_oracle = pd.DataFrame(
        {
            "strategy": "oracle_best_anchor_upper_bound",
            "score": "oracle_best_anchor_upper_bound",
            "target_wait_rate": np.nan,
            "threshold_from_val": np.nan,
            "event_uid": oracle["event_uid"].astype(str),
            "split": oracle["split"].astype(str),
            "subject": oracle["subject"].astype(str),
            "recording": oracle["recording"].astype(str),
            "selected_delay_ms": pd.to_numeric(oracle["oracle_delay_ms"], errors="coerce").fillna(0).astype(int),
            "selected_tail_rmse_v241": pd.to_numeric(oracle["oracle_tail_rmse_v241"], errors="coerce"),
            "keep0_tail_rmse_v241": pd.to_numeric(oracle["keep0_tail_rmse_v241"], errors="coerce"),
            "latest_tail_rmse_v241": pd.to_numeric(oracle["latest_tail_rmse_v241"], errors="coerce"),
            "oracle_tail_rmse_v241": pd.to_numeric(oracle["oracle_tail_rmse_v241"], errors="coerce"),
            "bad_top10": oracle["bad_top10_split_v241"].fillna(False).astype(bool),
            "very_bad_top5": oracle["very_bad_top5_split_v241"].fillna(False).astype(bool),
            "normal": oracle["normal_curve_current0"].fillna(False).astype(bool),
            "observe_later_like": oracle["observe_later_like_current0"].fillna(False).astype(bool),
            "strong_steer": oracle["strong_steer_current0"].fillna(False).astype(bool),
            "reverse": oracle["reverse_current0"].fillna(False).astype(bool),
            "early_best_after_400": pd.to_numeric(oracle["oracle_delay_ms"], errors="coerce").fillna(0).astype(int) >= 400,
        }
    )
    out_oracle["delta_selected_minus_keep0"] = out_oracle["selected_tail_rmse_v241"] - out_oracle["keep0_tail_rmse_v241"]
    out_oracle["delta_selected_minus_latest"] = out_oracle["selected_tail_rmse_v241"] - out_oracle["latest_tail_rmse_v241"]
    rows.append(out_oracle)
    return pd.concat(rows, ignore_index=True)


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    summary = V258.summarize_selected(selected)
    # v258 summary drops score/wait-rate metadata because it groups by strategy.
    meta = selected[["strategy", "score", "target_wait_rate", "threshold_from_val"]].drop_duplicates("strategy")
    return summary.merge(meta, on="strategy", how="left")


def build_frontier(events: pd.DataFrame, score_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val = events[events["split"].astype(str).eq("val")]
    selected_rows = [baseline_selected(events)]
    thresholds = []
    for score in score_cols:
        for wait_rate in WAIT_RATES:
            threshold = threshold_for_wait_rate(val[score], wait_rate)
            thresholds.append({"score": score, "target_wait_rate": wait_rate, "threshold_from_val": threshold})
            selected_rows.append(make_selected(events, score, wait_rate, threshold))
    selected = pd.concat(selected_rows, ignore_index=True)
    summary = summarize_selected(selected)
    return selected, summary, pd.DataFrame(thresholds)


def plot_frontier(summary: pd.DataFrame) -> Path:
    path = FIGURES / "v265_test_badtop10_wait_frontier.png"
    focus_scores = [
        "score_vehicle_gain",
        "score_vehicle_bio_gain",
        "score_vehicle_keep0_risk",
        "score_vehicle_bio_keep0_risk",
        "score_vehicle_badprob",
        "score_vehicle_bio_badprob",
        "score_bio_only_badprob",
    ]
    sub = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["score"].isin(focus_scores)
    ].copy()
    if sub.empty:
        return path
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for score, group in sub.groupby("score", sort=False):
        group = group.sort_values("selected_latest_rate")
        ax.plot(group["selected_latest_rate"], group["selected_tail_rmse_mean"], marker="o", linewidth=1.8, label=score.replace("score_", ""))
    base = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["strategy"].isin(["policy_keep_0ms_anchor", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"])
    ]
    for _, row in base.iterrows():
        ax.axhline(float(row["selected_tail_rmse_mean"]), linestyle="--", alpha=0.45, linewidth=1.2)
        ax.text(0.01, float(row["selected_tail_rmse_mean"]), str(row["strategy"]).replace("policy_", ""), fontsize=8, va="bottom")
    ax.set_xlabel("test selected_latest_rate")
    ax.set_ylabel("test bad_top10 selected tail RMSE")
    ax.set_title("v265: 同等等待预算下，生理风险分数是否优于车辆分数")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_input_hashes() -> None:
    rows = []
    for label, path in [
        ("v265_script", Path(__file__)),
        ("v263_event_wait_gate_predictions", V263_EVENTS),
        ("v262_feature_selection", V262_FEATURE_SELECTION),
        ("v258_reused_training_utils", V258_SCRIPT),
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
            rows.append({"relative_path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size})
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
    score_diag: pd.DataFrame,
    feature_audit: pd.DataFrame,
    figures: List[Path],
) -> None:
    lines: List[str] = []
    lines.append("# v265 physiology uncertainty / wait frontier")
    lines.append("")
    lines.append("## 本轮问题")
    lines.append("")
    lines.append("- v260-v264 说明生理不能直接改善轨迹、selector、wait gate 或 online KNN。")
    lines.append("- v265 检查最后一个合理用途：生理是否能作为不确定性/风险校准信号，在固定等待预算下更会挑选需要 wait-latest 的样本。")
    lines.append("- 所有风险模型只在 train 拟合，等待比例阈值只在 val 定标，test 只报告。")
    lines.append("")
    lines.append("## 特征块")
    lines.append("")
    lines.append(feature_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## 分数诊断")
    lines.append("")
    diag_focus = score_diag[score_diag["split"].isin(["val", "test"])].copy()
    lines.append(diag_focus.to_markdown(index=False))
    lines.append("")
    lines.append("## Test bad_top10 等待前沿")
    lines.append("")
    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & (
            summary["score"].isin(
                [
                    "score_vehicle_gain",
                    "score_vehicle_bio_gain",
                    "score_bio_only_gain",
                    "score_vehicle_keep0_risk",
                    "score_vehicle_bio_keep0_risk",
                    "score_vehicle_badprob",
                    "score_vehicle_bio_badprob",
                    "score_bio_only_badprob",
                ]
            )
            | summary["strategy"].isin(
                ["policy_keep_0ms_anchor", "policy_wait_to_latest_anchor", "oracle_best_anchor_upper_bound"]
            )
        )
    ].copy()
    lines.append(
        focus[
            [
                "strategy",
                "score",
                "target_wait_rate",
                "n",
                "selected_tail_rmse_mean",
                "selected_latest_rate",
                "delta_selected_minus_keep0_mean",
                "delta_selected_minus_latest_mean",
                "improve_rate_vs_keep0",
            ]
        ].to_markdown(index=False)
    )
    lines.append("")
    lines.append("## 判读")
    lines.append("")
    bad = summary[summary["split"].eq("test") & summary["event_group"].eq("bad_top10")].copy()
    best_rows = []
    for score, group in bad[bad["score"].astype(str).str.startswith("score_")].groupby("score"):
        row = group.sort_values("selected_tail_rmse_mean").iloc[0]
        best_rows.append(row)
        lines.append(
            f"- {score}: best tail={float(row['selected_tail_rmse_mean']):.4f}, latest_rate={float(row['selected_latest_rate']):.3f}, target_wait_rate={float(row['target_wait_rate']):.2f}."
        )
    best = pd.DataFrame(best_rows) if best_rows else pd.DataFrame()
    if not best.empty:
        vehicle_best = best[best["score"].eq("score_vehicle_gain")]
        bio_best = best[best["score"].eq("score_vehicle_bio_gain")]
        if len(vehicle_best) and len(bio_best):
            delta = float(bio_best["selected_tail_rmse_mean"].iloc[0] - vehicle_best["selected_tail_rmse_mean"].iloc[0])
            lines.append(f"- vehicle+bio gain 相对 vehicle gain 的最佳前沿改变量为 {delta:+.4f}。")
        vbad = best[best["score"].eq("score_vehicle_badprob")]
        vbbad = best[best["score"].eq("score_vehicle_bio_badprob")]
        if len(vbad) and len(vbbad):
            delta = float(vbbad["selected_tail_rmse_mean"].iloc[0] - vbad["selected_tail_rmse_mean"].iloc[0])
            lines.append(f"- vehicle+bio badprob 相对 vehicle badprob 的最佳前沿改变量为 {delta:+.4f}。")
    lines.append("- 若 bio 分数不能在同等等待预算下稳定低于 vehicle 分数，则当前生理不能作为可部署风险校准增量。")
    lines.append("")
    lines.append("## 关键图")
    lines.append("")
    for fig in figures:
        lines.append(f"- `{fig.relative_to(OUT)}`")
    (REPORTS / "v265_physio_uncertainty_wait_frontier_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[v265] physiology uncertainty / wait frontier")
    clean_out_dir()
    np.random.seed(SEED)

    events = load_events()
    vehicle_cols, bio_cols = feature_columns(events)
    events, feature_audit = fit_scores(events, vehicle_cols, bio_cols)
    score_cols = [col for col in events.columns if col.startswith("score_")]
    selected, summary, thresholds = build_frontier(events, score_cols)
    diag = score_diagnostics(events, score_cols)
    figures = [plot_frontier(summary)]

    write_csv(events, TABLES / "v265_event_risk_scores.csv")
    write_csv(selected, TABLES / "v265_selected_wait_frontier_by_policy.csv")
    write_csv(summary, TABLES / "v265_wait_frontier_summary.csv")
    write_csv(thresholds, TABLES / "v265_val_thresholds_by_wait_rate.csv")
    write_csv(diag, TABLES / "v265_score_diagnostics.csv")
    write_csv(feature_audit, TABLES / "v265_feature_block_audit.csv")
    write_input_hashes()
    write_file_inventory()
    write_report(summary, diag, feature_audit, figures)
    write_file_inventory()

    zip_ok = make_zip()
    guardrail = {
        "pass": bool(zip_ok),
        "zip_testzip": bool(zip_ok),
        "train_only_score_models": True,
        "val_only_wait_rate_thresholds": True,
        "policy_space": "keep0_or_wait_latest_by_risk_score",
        "event_n": int(events["event_uid"].nunique()),
        "vehicle_feature_n": int(len(vehicle_cols)),
        "bio260_sp64_feature_n": int(len(bio_cols)),
        "score_n": int(len(score_cols)),
    }
    (LOGS / "guardrail_check.json").write_text(json.dumps(guardrail, ensure_ascii=False, indent=2), encoding="utf-8")
    if not guardrail["pass"]:
        raise AssertionError("v265 guardrail 未通过：\n" + json.dumps(guardrail, ensure_ascii=False, indent=2))
    write_file_inventory()

    focus = summary[
        summary["split"].eq("test")
        & summary["event_group"].eq("bad_top10")
        & summary["score"].isin(["score_vehicle_gain", "score_vehicle_bio_gain", "score_vehicle_badprob", "score_vehicle_bio_badprob", "score_bio_only_badprob"])
    ].copy()
    best = focus.sort_values("selected_tail_rmse_mean").groupby("score", as_index=False).head(1)
    print(f"[v265] report={REPORTS / 'v265_physio_uncertainty_wait_frontier_cn.md'}")
    print(f"[v265] zip={ZIP_PATH}")
    print(best[["score", "target_wait_rate", "selected_tail_rmse_mean", "selected_latest_rate", "delta_selected_minus_latest_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
