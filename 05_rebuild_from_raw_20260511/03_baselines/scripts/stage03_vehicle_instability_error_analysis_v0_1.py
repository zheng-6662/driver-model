# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
FORMAL_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_formal_baselines_v0_1"
FORMAL_PER_SAMPLE = FORMAL_DIR / "tables" / "formal_baseline_per_sample_metrics.csv"
FORMAL_METRICS = FORMAL_DIR / "tables" / "formal_baseline_metrics.csv"
OLD_PER_SAMPLE = (
    ROOT
    / "03_baselines"
    / "oldcode_vehicle_direct_full_clean_on_instability_v0_1"
    / "tables"
    / "oldcode_vehicle_direct_full_per_sample_metrics.csv"
)
OLD_METRICS = (
    ROOT
    / "03_baselines"
    / "oldcode_vehicle_direct_full_clean_on_instability_v0_1"
    / "tables"
    / "oldcode_vehicle_direct_full_metrics.csv"
)
SAMPLES_MASTER = ROOT / "02_samples" / "vehicle_instability_highconf_v0_1" / "tables" / "samples_master.csv"
OUT_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_error_analysis_v0_1"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

WINDOW = "pre2_label2_old_main"
SPLIT_STRATEGY = "session_level_split"
SPLIT = "test"
FORMAL_MODEL = "ridge_vehicle_context_no_subject"
OLD_MODEL = "active_legacy_best"

ERROR_FLAGS = [
    "high_rmse_top20pct",
    "wrong_side_flag",
    "large_response_missed_flag",
    "severe_amp_under_flag",
    "multi_segment_missed_flag",
    "multi_segment_overpred_flag",
    "multi_segment_mismatch_flag",
    "reversal_mismatch_flag",
    "tail_drift_flag",
    "zero_crossing_mismatch_flag",
    "peak_time_large_error_flag",
    "onset_delay_large_error_flag",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_selected() -> pd.DataFrame:
    formal = pd.read_csv(FORMAL_PER_SAMPLE)
    selected = formal[
        (formal["window_config_id"] == WINDOW)
        & (formal["split_strategy"] == SPLIT_STRATEGY)
        & (formal["split"] == SPLIT)
        & (formal["model_name"] == FORMAL_MODEL)
    ].copy()
    if selected.empty:
        raise ValueError("No formal selected samples found")

    samples = pd.read_csv(SAMPLES_MASTER)
    meta_cols = [
        "sample_id",
        "road_type_anchor",
        "event_level",
        "instability_role",
        "road_design_module_name",
        "road_design_risk_class",
        "road_design_mapping_reliability",
        "old_v400_road_type_mode",
        "old_v400_phase_mode",
        "physio_available",
        "eeg_available",
        "all_three_modalities_available",
        "eval_label_morphology",
    ]
    selected = selected.merge(samples[[c for c in meta_cols if c in samples.columns]], on="sample_id", how="left")

    if OLD_PER_SAMPLE.exists():
        old = pd.read_csv(OLD_PER_SAMPLE)
        old = old[
            (old["window_config_id"] == WINDOW)
            & (old["split_strategy"] == SPLIT_STRATEGY)
            & (old["split"] == SPLIT)
            & (old["model_name"] == OLD_MODEL)
        ].copy()
        old_cols = [
            "sample_id",
            "sample_rmse",
            "gt_peak_abs",
            "pred_peak_abs",
            "wrong_side",
            "large_response_recalled",
            "severe_amp_under",
            "tail_drift_risk",
            "reversal_count_exact",
            "pred_multi_segment",
        ]
        old = old[[c for c in old_cols if c in old.columns]].rename(
            columns={
                "sample_rmse": "old_deep_sample_rmse",
                "gt_peak_abs": "old_deep_gt_peak_abs",
                "pred_peak_abs": "old_deep_pred_peak_abs",
                "wrong_side": "old_deep_wrong_side",
                "large_response_recalled": "old_deep_large_response_recalled",
                "severe_amp_under": "old_deep_severe_amp_under",
                "tail_drift_risk": "old_deep_tail_drift_risk",
                "reversal_count_exact": "old_deep_reversal_count_exact",
                "pred_multi_segment": "old_deep_pred_multi_segment",
            }
        )
        selected = selected.merge(old, on="sample_id", how="left")
    return selected


def assign_primary_error(row: pd.Series) -> str:
    if bool(row["wrong_side_flag"]):
        return "01_wrong_side"
    if bool(row["large_response_missed_flag"]):
        return "02_large_response_missed"
    if bool(row["severe_amp_under_flag"]):
        return "03_severe_amp_under"
    if bool(row["multi_segment_missed_flag"]):
        return "04_multi_segment_missed"
    if bool(row["multi_segment_overpred_flag"]):
        return "05_multi_segment_overpred"
    if bool(row["reversal_mismatch_flag"]):
        return "06_reversal_mismatch"
    if bool(row["tail_drift_flag"]):
        return "07_tail_drift"
    if bool(row["peak_time_large_error_flag"]):
        return "08_peak_time_error"
    if bool(row["high_rmse_top20pct"]):
        return "09_high_rmse_other"
    return "10_no_major_flag"


def build_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high_rmse_thr = float(np.nanpercentile(out["sample_rmse"], 80))
    peak_time_thr = 0.5
    onset_thr = 0.5
    out["high_rmse_threshold_test_p80"] = high_rmse_thr
    out["high_rmse_top20pct"] = out["sample_rmse"] >= high_rmse_thr
    out["wrong_side_flag"] = out["wrong_side"].astype(int) == 1
    out["large_response_missed_flag"] = (out["is_large_response"].astype(int) == 1) & (
        out["large_response_recalled"].astype(int) == 0
    )
    out["severe_amp_under_flag"] = out["severe_amp_under"].astype(int) == 1
    out["multi_segment_missed_flag"] = (out["gt_multi_segment"].astype(int) == 1) & (
        out["pred_multi_segment"].astype(int) == 0
    )
    out["multi_segment_overpred_flag"] = (out["gt_multi_segment"].astype(int) == 0) & (
        out["pred_multi_segment"].astype(int) == 1
    )
    out["multi_segment_mismatch_flag"] = out["gt_multi_segment"].astype(int) != out["pred_multi_segment"].astype(int)
    out["reversal_mismatch_flag"] = out["reversal_count_exact"].astype(int) == 0
    out["tail_drift_flag"] = out["tail_drift_risk"].astype(int) == 1
    out["zero_crossing_mismatch_flag"] = out["zero_crossing_mismatch"].astype(int) == 1
    out["peak_time_large_error_flag"] = pd.to_numeric(out["peak_time_abs_error_s"], errors="coerce") >= peak_time_thr
    out["onset_delay_large_error_flag"] = pd.to_numeric(out["onset_delay_abs_error_s"], errors="coerce") >= onset_thr
    out["formal_error_flag_count"] = out[ERROR_FLAGS].sum(axis=1).astype(int)
    out["primary_error_type"] = out.apply(assign_primary_error, axis=1)
    if "old_deep_sample_rmse" in out.columns:
        out["formal_minus_old_deep_rmse"] = out["sample_rmse"] - out["old_deep_sample_rmse"]
        out["formal_better_than_old_deep"] = out["formal_minus_old_deep_rmse"] < 0
        out["shared_with_old_deep_bad_top20pct"] = False
        old_thr = float(np.nanpercentile(out["old_deep_sample_rmse"].dropna(), 80)) if out["old_deep_sample_rmse"].notna().any() else np.nan
        out["old_deep_high_rmse_threshold_test_p80"] = old_thr
        if np.isfinite(old_thr):
            out["old_deep_high_rmse_top20pct"] = out["old_deep_sample_rmse"] >= old_thr
            out["shared_with_old_deep_bad_top20pct"] = out["high_rmse_top20pct"] & out["old_deep_high_rmse_top20pct"]
    keep = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
        "road_type_anchor",
        "event_level",
        "instability_role",
        "road_design_module_name",
        "road_design_risk_class",
        "road_design_mapping_reliability",
        "old_v400_road_type_mode",
        "old_v400_phase_mode",
        "eval_label_morphology",
        "sample_rmse",
        "gt_peak_abs",
        "pred_peak_abs",
        "peak_amp_ratio_pred_over_gt",
        "wrong_side",
        "large_response_recalled",
        "severe_amp_under",
        "peak_time_abs_error_s",
        "onset_delay_abs_error_s",
        "tail_abs_error",
        "tail_drift_risk",
        "zero_crossing_mismatch",
        "gt_reversal_count",
        "pred_reversal_count",
        "reversal_count_exact",
        "gt_multi_segment",
        "pred_multi_segment",
        "is_large_response",
        "is_difficult_peak_top20",
        *ERROR_FLAGS,
        "formal_error_flag_count",
        "primary_error_type",
        "old_deep_sample_rmse",
        "old_deep_pred_peak_abs",
        "old_deep_wrong_side",
        "old_deep_large_response_recalled",
        "old_deep_severe_amp_under",
        "old_deep_tail_drift_risk",
        "old_deep_reversal_count_exact",
        "old_deep_pred_multi_segment",
        "formal_minus_old_deep_rmse",
        "formal_better_than_old_deep",
        "shared_with_old_deep_bad_top20pct",
        "old_deep_high_rmse_top20pct",
        "high_rmse_threshold_test_p80",
        "old_deep_high_rmse_threshold_test_p80",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values("sample_rmse", ascending=False)


def flag_summary(tax: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(tax)
    for flag in ERROR_FLAGS:
        subset = tax[tax[flag].astype(bool)]
        rows.append(
            {
                "error_flag": flag,
                "n_samples": int(len(subset)),
                "rate": float(len(subset) / max(n, 1)),
                "mean_rmse": float(subset["sample_rmse"].mean()) if len(subset) else np.nan,
                "median_gt_peak_abs": float(subset["gt_peak_abs"].median()) if len(subset) else np.nan,
                "old_deep_mean_rmse": float(subset["old_deep_sample_rmse"].mean()) if "old_deep_sample_rmse" in subset.columns and len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["n_samples", "mean_rmse"], ascending=[False, False])


def group_summary(tax: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in tax.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, grp in tax.groupby(group_col, dropna=False):
        row = {
            group_col: key,
            "n_samples": int(len(grp)),
            "mean_rmse": float(grp["sample_rmse"].mean()),
            "wrong_side_rate": float(grp["wrong_side_flag"].mean()),
            "large_response_missed_rate": float(grp["large_response_missed_flag"].mean()),
            "severe_amp_under_rate": float(grp["severe_amp_under_flag"].mean()),
            "multi_segment_missed_rate": float(grp["multi_segment_missed_flag"].mean()),
            "multi_segment_overpred_rate": float(grp["multi_segment_overpred_flag"].mean()),
            "multi_segment_mismatch_rate": float(grp["multi_segment_mismatch_flag"].mean()),
            "reversal_mismatch_rate": float(grp["reversal_mismatch_flag"].mean()),
            "tail_drift_rate": float(grp["tail_drift_flag"].mean()),
            "high_rmse_top20pct_rate": float(grp["high_rmse_top20pct"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_rmse", "n_samples"], ascending=[False, False])


def old_comparison_summary(tax: pd.DataFrame) -> pd.DataFrame:
    if "old_deep_sample_rmse" not in tax.columns:
        return pd.DataFrame()
    valid = tax.dropna(subset=["old_deep_sample_rmse"]).copy()
    formal_aggregate_rmse = np.nan
    old_aggregate_rmse = np.nan
    if FORMAL_METRICS.exists():
        fm = pd.read_csv(FORMAL_METRICS)
        row = fm[
            (fm["window_config_id"] == WINDOW)
            & (fm["split_strategy"] == SPLIT_STRATEGY)
            & (fm["split"] == SPLIT)
            & (fm["model_name"] == FORMAL_MODEL)
        ]
        if len(row):
            formal_aggregate_rmse = float(row["rmse_steer"].iloc[0])
    if OLD_METRICS.exists():
        om = pd.read_csv(OLD_METRICS)
        row = om[
            (om["checkpoint_tag"] == OLD_MODEL)
            & (om["sample_window_config_id"] == WINDOW)
            & (om["sample_split_strategy"] == SPLIT_STRATEGY)
            & (om["sample_split"] == SPLIT)
        ]
        if len(row):
            old_aggregate_rmse = float(row["sample_rmse_steer"].iloc[0])
    rows = [
        {
            "comparison": "formal_ridge_context_vs_old_vehicle_direct_active",
            "n_samples": int(len(valid)),
            "formal_aggregate_rmse": formal_aggregate_rmse,
            "old_deep_aggregate_rmse": old_aggregate_rmse,
            "formal_mean_rmse": float(valid["sample_rmse"].mean()),
            "old_deep_mean_rmse": float(valid["old_deep_sample_rmse"].mean()),
            "formal_better_n": int(valid["formal_better_than_old_deep"].sum()),
            "formal_better_rate": float(valid["formal_better_than_old_deep"].mean()),
            "shared_bad_top20pct_n": int(valid["shared_with_old_deep_bad_top20pct"].sum()),
            "formal_bad_top20pct_n": int(valid["high_rmse_top20pct"].sum()),
            "old_deep_bad_top20pct_n": int(valid["old_deep_high_rmse_top20pct"].sum()) if "old_deep_high_rmse_top20pct" in valid else 0,
        }
    ]
    return pd.DataFrame(rows)


def plot_flag_counts(summary: pd.DataFrame) -> Path:
    out = FIG_DIR / "formal_error_flag_counts.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    data = summary.sort_values("n_samples", ascending=True)
    ax.barh(data["error_flag"], data["n_samples"], color="#4c78a8")
    ax.set_xlabel("test samples")
    ax.set_title("Formal vehicle baseline error flags")
    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(row["n_samples"] + 0.5, i, f"{row['rate']:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_old_scatter(tax: pd.DataFrame) -> Path:
    out = FIG_DIR / "formal_vs_old_deep_rmse_scatter.png"
    if "old_deep_sample_rmse" not in tax.columns:
        return out
    valid = tax.dropna(subset=["old_deep_sample_rmse"])
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(valid["high_rmse_top20pct"].astype(bool), "#d62728", "#4c78a8")
    ax.scatter(valid["old_deep_sample_rmse"], valid["sample_rmse"], c=colors, alpha=0.75, edgecolor="white", linewidth=0.4)
    max_val = float(np.nanmax([valid["old_deep_sample_rmse"].max(), valid["sample_rmse"].max()]))
    ax.plot([0, max_val], [0, max_val], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("old vehicle_direct sample RMSE")
    ax.set_ylabel("formal ridge-context sample RMSE")
    ax.set_title("Formal shallow baseline vs old deep comparison")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_error_matrix(tax: pd.DataFrame) -> Path:
    out = FIG_DIR / "top_bad_sample_error_matrix.png"
    top = tax.sort_values("sample_rmse", ascending=False).head(24).copy()
    mat = top[ERROR_FLAGS].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(ERROR_FLAGS)))
    ax.set_xticklabels(ERROR_FLAGS, rotation=45, ha="right", fontsize=8)
    labels = [f"{r.subject}:{r.anchor:.1f}s" for r in top.rename(columns={"session_stamp": "session"}).assign(anchor=top["sample_id"].str.extract(r"__(\d{9})__")[0].astype(float) / 1000.0).itertuples()]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Top bad samples and physical error flags")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_subject_heatmap(subject_summary: pd.DataFrame) -> Path:
    out = FIG_DIR / "subject_error_rate_heatmap.png"
    if subject_summary.empty:
        return out
    cols = [
        "wrong_side_rate",
        "large_response_missed_rate",
        "severe_amp_under_rate",
        "multi_segment_missed_rate",
        "multi_segment_overpred_rate",
        "multi_segment_mismatch_rate",
        "reversal_mismatch_rate",
        "tail_drift_rate",
    ]
    data = subject_summary.sort_values("mean_rmse", ascending=False).set_index("subject")
    mat = data[cols].fillna(0.0).to_numpy()
    fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_title("Subject-level error rates")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def write_reports(tax: pd.DataFrame, summaries: dict[str, pd.DataFrame], figures: dict[str, Path]) -> dict[str, Any]:
    flag_table = summaries["error_flags"]
    primary_error = tax["primary_error_type"].value_counts().rename_axis("primary_error_type").reset_index(name="n_samples")
    subject = summaries["subject"].head(12)
    morphology = summaries["morphology"]
    old_cmp = summaries["old_comparison"]
    top_bad = tax.head(12)[
        [
            "sample_id",
            "subject",
            "sample_rmse",
            "gt_peak_abs",
            "pred_peak_abs",
            "primary_error_type",
            "wrong_side_flag",
            "large_response_missed_flag",
            "severe_amp_under_flag",
            "multi_segment_missed_flag",
            "multi_segment_overpred_flag",
            "multi_segment_mismatch_flag",
            "reversal_mismatch_flag",
            "tail_drift_flag",
            "old_deep_sample_rmse",
        ]
    ]
    best_flag = flag_table.iloc[0].to_dict() if len(flag_table) else {}
    summary = {
        "n_test_samples": int(len(tax)),
        "model_name": FORMAL_MODEL,
        "window": WINDOW,
        "split_strategy": SPLIT_STRATEGY,
        "dominant_error_flag": best_flag.get("error_flag", ""),
        "dominant_error_count": int(best_flag.get("n_samples", 0)) if best_flag else 0,
        "wrong_side_count": int(tax["wrong_side_flag"].sum()),
        "large_response_missed_count": int(tax["large_response_missed_flag"].sum()),
        "severe_amp_under_count": int(tax["severe_amp_under_flag"].sum()),
        "multi_segment_missed_count": int(tax["multi_segment_missed_flag"].sum()),
        "multi_segment_overpred_count": int(tax["multi_segment_overpred_flag"].sum()),
        "multi_segment_mismatch_count": int(tax["multi_segment_mismatch_flag"].sum()),
        "old_comparison": old_cmp.to_dict("records"),
        "server_used": False,
        "credential_file_read": False,
    }
    (LOG_DIR / "stage03_error_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 阶段 3 v0.1 坏样本物理错误分型分析

生成时间：2026-05-12

## 目的

阶段 3 v0.1 的最优浅层车辆基线 `ridge_vehicle_context_no_subject` 虽然是当前新流程车辆-only 起点，但固定图显示它仍有明显物理错误。本分析把 test 样本逐条打上错误标签，判断下一步应优先强化车辆模型哪一部分。

## 输入

- 正式车辆基线逐样本指标：`{FORMAL_PER_SAMPLE.as_posix()}`
- 正式样本清单：`{SAMPLES_MASTER.as_posix()}`
- 旧 `vehicle_direct` clean 对照逐样本指标：`{OLD_PER_SAMPLE.as_posix()}`

## 样本范围

- 窗口：`{WINDOW}`
- split：`{SPLIT_STRATEGY}` / `{SPLIT}`
- 模型：`{FORMAL_MODEL}`
- test 样本数：{len(tax)}

## 错误标签计数

{flag_table.to_string(index=False)}

## 主错误类型计数

{primary_error.to_string(index=False)}

## 分响应类型错误

{morphology.to_string(index=False)}

## 分被试错误，按 mean RMSE 前 12

{subject.to_string(index=False)}

## 与旧 deep vehicle_direct 对照

{old_cmp.to_string(index=False) if len(old_cmp) else '未找到可比旧 deep 指标。'}

说明：`formal_aggregate_rmse` 和 `old_deep_aggregate_rmse` 是主指标表中的整体 RMSE；`formal_mean_rmse` 和 `old_deep_mean_rmse` 是逐样本 RMSE 的算术平均。两者口径不同，不能混用。旧 deep 的整体 RMSE 仍略低于 formal ridge，但 formal ridge 在更多单个样本上逐样本 RMSE 更小，说明 formal ridge 的错误更集中在少数高幅/复杂响应样本上。

## Top 12 坏样本

{top_bad.to_string(index=False)}

## 图

- 错误标签柱状图：`{figures['flag_counts'].as_posix()}`
- 与旧 deep RMSE 散点图：`{figures['old_scatter'].as_posix()}`
- Top bad 错误矩阵：`{figures['error_matrix'].as_posix()}`
- 分被试错误热图：`{figures['subject_heatmap'].as_posix()}`

## 当前判断

车辆-only 浅层基线的主要问题不是单一 RMSE，而是高比例的复杂响应结构错误：反向修正计数不匹配、多段修正过度预测或漏检、尾段漂移、严重幅值不足和错侧同时存在。下一步更适合先增强车辆时序/结构化响应基线，而不是直接声称连续风格或生理提供增量。
"""
    (REPORT_DIR / "stage03_vehicle_instability_error_analysis_v0_1_cn.md").write_text(report, encoding="utf-8")

    user = f"""# 阶段 3 用户查看版：车辆基线坏样本错误分型

生成时间：2026-05-12

## 为什么做

阶段 3 v0.1 已经有车辆-only 基线，但只看 RMSE 不够。我们需要知道模型到底错在哪里，才能决定下一步是强化车辆模型，还是以后再看风格/生理是否能补充。

## 检查了什么

- 错侧。
- 大幅响应漏召回。
- 严重幅值不足。
- 多段修正漏检。
- 反向修正数量不匹配。
- 尾段漂移。
- 零线穿越错误。
- 峰值时间和启动延迟错误。
- 和旧 `vehicle_direct` deep 对照的坏样本重叠。

## 目前发现

test 样本 {len(tax)} 个。错误最多的标签是 `{best_flag.get('error_flag', '')}`，数量 {int(best_flag.get('n_samples', 0)) if best_flag else 0}。错侧样本 {summary['wrong_side_count']} 个，大幅响应漏召回 {summary['large_response_missed_count']} 个，严重幅值不足 {summary['severe_amp_under_count']} 个，多段修正漏检 {summary['multi_segment_missed_count']} 个，多段修正过度预测 {summary['multi_segment_overpred_count']} 个。

和旧 deep 对照时，旧 deep 的整体 RMSE 仍略低；但 formal ridge 在 92/139 个单样本上逐样本 RMSE 更小，说明 formal ridge 的坏样本更集中，不能只看平均数。

## 哪些结果可信

这些错误标签只来自 test 集评估结果，不参与训练，不参与 split，也不用于标准化。它们用于解释模型失败类型。

## 哪些结果还不能下结论

不能因为某些错误多，就说生理一定能解决。现在只能说明车辆-only 浅层基线在哪些物理响应上不够好。

## 下一阶段是否可以继续

建议先做更强的车辆时序/结构化响应基线，再进入风格或生理增量验证。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_error_analysis_v0_1_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/formal_error_flag_counts.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/top_bad_sample_error_matrix.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/per_sample_error_taxonomy.csv`
"""
    (REPORT_DIR / "stage03_vehicle_instability_error_analysis_user_summary_cn.md").write_text(user, encoding="utf-8")
    return summary


def main() -> None:
    ensure_dirs()
    selected = load_selected()
    tax = build_taxonomy(selected)
    tax.to_csv(TABLE_DIR / "per_sample_error_taxonomy.csv", index=False, encoding="utf-8-sig")

    summaries = {
        "error_flags": flag_summary(tax),
        "subject": group_summary(tax, "subject"),
        "morphology": group_summary(tax, "eval_label_morphology"),
        "road_type": group_summary(tax, "road_type_anchor"),
        "event_level": group_summary(tax, "event_level"),
        "old_comparison": old_comparison_summary(tax),
    }
    for name, df in summaries.items():
        df.to_csv(TABLE_DIR / f"{name}_summary.csv", index=False, encoding="utf-8-sig")

    figures = {
        "flag_counts": plot_flag_counts(summaries["error_flags"]),
        "old_scatter": plot_old_scatter(tax),
        "error_matrix": plot_error_matrix(tax),
        "subject_heatmap": plot_subject_heatmap(summaries["subject"]),
    }
    summary = write_reports(tax, summaries, figures)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
