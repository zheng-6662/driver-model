# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
BASELINE_ROOT = ROOT / "03_baselines"
REPORT_ROOT = ROOT / "09_reports"
CLEAN_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1"
MANIFEST_PATH = (
    ROOT
    / "02_samples"
    / "vehicle_instability_response_task_decision_v0_1"
    / "tables"
    / "sample_response_task_manifest.csv"
)
PER_SAMPLE_PATH = CLEAN_ROOT / "tables" / "clean_task_vehicle_per_sample_metrics.csv"
OUT_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_clean_task_bad_sample_review_v0_1"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
LOG_DIR = OUT_ROOT / "logs"

TARGET_TRACK = "B_response3s_strict_core"
TARGET_MODEL = "rbf_kernel_ridge_context_no_subject"
TARGET_SPLIT = "test"

FLAG_COLUMNS = [
    "high_rmse_top20_flag",
    "wrong_side_flag",
    "severe_amp_under_flag",
    "large_response_missed_flag",
    "tail_drift_flag",
    "zero_crossing_mismatch_flag",
    "reversal_mismatch_flag",
    "multi_segment_mismatch_flag",
    "peak_time_large_error_flag",
    "onset_delay_large_error_flag",
    "amplitude_large_error_flag",
]

FLAG_CN = {
    "high_rmse_top20_flag": "RMSE最高20%",
    "wrong_side_flag": "主峰错侧",
    "severe_amp_under_flag": "严重幅值不足",
    "large_response_missed_flag": "大幅响应漏召回",
    "tail_drift_flag": "尾段漂移/未回正",
    "zero_crossing_mismatch_flag": "零线穿越错误",
    "reversal_mismatch_flag": "反向修正计数不匹配",
    "multi_segment_mismatch_flag": "多段修正结构不匹配",
    "peak_time_large_error_flag": "峰值时间误差大",
    "onset_delay_large_error_flag": "启动延迟误差大",
    "amplitude_large_error_flag": "峰值幅值误差大",
}

PRIMARY_FAILURE_ORDER = [
    ("wrong_side_flag", "wrong_side"),
    ("large_response_missed_flag", "large_response_missed"),
    ("severe_amp_under_flag", "severe_amplitude_under"),
    ("reversal_mismatch_flag", "reversal_structure_mismatch"),
    ("multi_segment_mismatch_flag", "multi_segment_structure_mismatch"),
    ("tail_drift_flag", "tail_drift_or_return_error"),
    ("zero_crossing_mismatch_flag", "zero_crossing_mismatch"),
    ("peak_time_large_error_flag", "peak_timing_error"),
    ("onset_delay_large_error_flag", "onset_timing_error"),
    ("amplitude_large_error_flag", "amplitude_large_error"),
]

FLAG_PLOT_LABEL = {
    "high_rmse_top20_flag": "high RMSE top20%",
    "wrong_side_flag": "wrong-side",
    "severe_amp_under_flag": "severe amp under",
    "large_response_missed_flag": "large response missed",
    "tail_drift_flag": "tail drift/return",
    "zero_crossing_mismatch_flag": "zero-cross mismatch",
    "reversal_mismatch_flag": "reversal mismatch",
    "multi_segment_mismatch_flag": "multi-segment mismatch",
    "peak_time_large_error_flag": "peak timing error",
    "onset_delay_large_error_flag": "onset timing error",
    "amplitude_large_error_flag": "amp error large",
}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if len(vals) else float("nan")


def bool_rate(series: pd.Series) -> float:
    vals = series.astype(bool)
    return float(vals.mean()) if len(vals) else float("nan")


def simple_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        rendered = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def load_target_rows() -> pd.DataFrame:
    if not PER_SAMPLE_PATH.exists():
        raise FileNotFoundError(PER_SAMPLE_PATH)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    per = pd.read_csv(PER_SAMPLE_PATH)
    target = per[
        (per["track_id"].astype(str) == TARGET_TRACK)
        & (per["split"].astype(str) == TARGET_SPLIT)
        & (per["model_name"].astype(str) == TARGET_MODEL)
    ].copy()
    if target.empty:
        raise RuntimeError(f"No rows for {TARGET_TRACK}/{TARGET_SPLIT}/{TARGET_MODEL}")

    manifest_cols = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
        "window_config_id",
        "road_type_anchor",
        "road_design_module_name",
        "road_design_instance_name",
        "event_level",
        "instability_role",
        "response_task_class",
        "response_task_class_cn",
        "response_task_track",
        "task_sample_role",
        "eval_label_morphology",
        "eval_label_peak_direction",
        "eval_label_peak_time_rel_s",
        "eval_label_onset_time_rel_s",
        "eval_label_tail_abs",
        "event_duration_s",
        "anchor_time_rel_s",
        "vehicle_raw_relative_path",
        "leakage_risk_level",
        "sample_trace_status",
    ]
    manifest = pd.read_csv(MANIFEST_PATH, usecols=lambda c: c in set(manifest_cols))
    merged = target.merge(
        manifest,
        on=["sample_id", "event_uid", "subject", "session_stamp", "window_config_id"],
        how="left",
        validate="one_to_one",
    )
    return merged


def add_failure_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    out = df.copy()
    rmse_threshold = float(out["sample_rmse"].quantile(0.8))
    amp_err_threshold = float(out["peak_amp_abs_error"].quantile(0.75))

    out["high_rmse_top20_flag"] = out["sample_rmse"].astype(float) >= rmse_threshold
    out["wrong_side_flag"] = out["wrong_side"].astype(int) == 1
    out["severe_amp_under_flag"] = out["severe_amp_under"].astype(int) == 1
    out["large_response_missed_flag"] = (
        (out["is_large_response"].astype(int) == 1)
        & (out["large_response_recalled"].astype(int) == 0)
    )
    out["tail_drift_flag"] = out["tail_drift_risk"].astype(int) == 1
    out["zero_crossing_mismatch_flag"] = out["zero_crossing_mismatch"].astype(int) == 1
    out["reversal_mismatch_flag"] = out["reversal_count_exact"].astype(int) == 0
    out["multi_segment_mismatch_flag"] = (
        out["gt_multi_segment"].astype(int) != out["pred_multi_segment"].astype(int)
    )
    out["peak_time_large_error_flag"] = out["peak_time_abs_error_s"].astype(float) >= 0.6
    out["onset_delay_large_error_flag"] = out["onset_delay_abs_error_s"].astype(float) >= 0.6
    out["amplitude_large_error_flag"] = out["peak_amp_abs_error"].astype(float) >= amp_err_threshold

    failure_tags: list[str] = []
    primary_failures: list[str] = []
    for _, row in out.iterrows():
        tags = [name for name in FLAG_COLUMNS if bool(row[name])]
        failure_tags.append(";".join(tags))
        primary = "not_high_rmse_top20" if not bool(row["high_rmse_top20_flag"]) else "high_rmse_without_specific_flag"
        if bool(row["high_rmse_top20_flag"]):
            for flag, label in PRIMARY_FAILURE_ORDER:
                if bool(row[flag]):
                    primary = label
                    break
        primary_failures.append(primary)
    out["failure_tags"] = failure_tags
    out["primary_failure_type"] = primary_failures
    out["rmse_top20_threshold"] = rmse_threshold
    out["amplitude_error_p75_threshold"] = amp_err_threshold
    return out, rmse_threshold


def build_flag_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    high = flagged[flagged["high_rmse_top20_flag"]].copy()
    for flag in FLAG_COLUMNS:
        rows.append(
            {
                "flag": flag,
                "flag_cn": FLAG_CN[flag],
                "overall_count": int(flagged[flag].sum()),
                "overall_rate": bool_rate(flagged[flag]),
                "high_rmse_top20_count": int(high[flag].sum()) if len(high) else 0,
                "high_rmse_top20_rate": bool_rate(high[flag]) if len(high) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_group_summary(flagged: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = flagged.copy()
    data[group_col] = data[group_col].fillna("unknown").astype(str)
    for value, group in data.groupby(group_col):
        row: dict[str, Any] = {
            group_col: value,
            "n_samples": int(len(group)),
            "mean_rmse": safe_mean(group["sample_rmse"]),
            "median_rmse": float(pd.to_numeric(group["sample_rmse"], errors="coerce").median()),
            "mean_gt_peak_abs": safe_mean(group["gt_peak_abs"]),
            "mean_pred_peak_abs": safe_mean(group["pred_peak_abs"]),
            "high_rmse_top20_rate": bool_rate(group["high_rmse_top20_flag"]),
        }
        for flag in FLAG_COLUMNS[1:]:
            row[f"{flag}_rate"] = bool_rate(group[flag])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["high_rmse_top20_rate", "mean_rmse", "n_samples"],
        ascending=[False, False, False],
    )


def build_top_bad(flagged: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    cols = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
        "road_design_module_name",
        "road_design_instance_name",
        "eval_label_morphology",
        "sample_rmse",
        "gt_peak_abs",
        "pred_peak_abs",
        "peak_amp_abs_error",
        "peak_amp_ratio_pred_over_gt",
        "wrong_side_flag",
        "severe_amp_under_flag",
        "large_response_missed_flag",
        "tail_drift_flag",
        "zero_crossing_mismatch_flag",
        "reversal_mismatch_flag",
        "multi_segment_mismatch_flag",
        "peak_time_abs_error_s",
        "onset_delay_abs_error_s",
        "primary_failure_type",
        "failure_tags",
        "vehicle_raw_relative_path",
    ]
    return flagged.sort_values("sample_rmse", ascending=False)[cols].head(top_n).reset_index(drop=True)


def plot_flag_summary(summary: pd.DataFrame) -> None:
    plot = summary.iloc[::-1].copy()
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.barh(y - 0.18, plot["overall_rate"], height=0.34, label="all B test", color="#4c78a8")
    ax.barh(y + 0.18, plot["high_rmse_top20_rate"], height=0.34, label="high-RMSE top20%", color="#e45756")
    ax.set_yticks(y)
    ax.set_yticklabels([FLAG_PLOT_LABEL.get(v, v) for v in plot["flag"]])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("rate")
    ax.set_title("B_response3s_strict_core RBF KRR failure flag rates")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(FIG_DIR / "b_track_rbf_failure_flag_rates.png", dpi=180)
    plt.close(fig)


def plot_top_bad(top_bad: pd.DataFrame) -> None:
    plot = top_bad.iloc[::-1].copy()
    labels = plot["subject"].astype(str) + " " + plot["anchor"] if "anchor" in plot.columns else (
        plot["subject"].astype(str) + " | " + plot["event_uid"].astype(str).str[-10:]
    )
    colors = np.where(plot["wrong_side_flag"], "#e45756", np.where(plot["severe_amp_under_flag"], "#f58518", "#4c78a8"))
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(labels, plot["sample_rmse"], color=colors)
    ax.set_xlabel("sample RMSE")
    ax.set_title("Worst B-track RBF KRR samples")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor="#e45756", label="wrong-side"),
            Patch(facecolor="#f58518", label="severe amp under"),
            Patch(facecolor="#4c78a8", label="other"),
        ],
        loc="lower right",
    )
    fig.savefig(FIG_DIR / "b_track_rbf_top_bad_rmse.png", dpi=180)
    plt.close(fig)


def plot_peak_scatter(flagged: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    colors = np.where(
        flagged["wrong_side_flag"],
        "#e45756",
        np.where(flagged["severe_amp_under_flag"], "#f58518", "#4c78a8"),
    )
    ax.scatter(
        flagged["gt_peak_abs"],
        flagged["pred_peak_abs"],
        s=45 + 120 * flagged["sample_rmse"] / max(float(flagged["sample_rmse"].max()), 1e-6),
        c=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.7,
    )
    lim = max(float(flagged["gt_peak_abs"].max()), float(flagged["pred_peak_abs"].max())) * 1.1
    ax.plot([0, lim], [0, lim], color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("GT peak abs")
    ax.set_ylabel("Pred peak abs")
    ax.set_title("B-track RBF KRR peak amplitude scatter")
    ax.grid(alpha=0.25)
    fig.savefig(FIG_DIR / "b_track_rbf_peak_amp_scatter.png", dpi=180)
    plt.close(fig)


def write_reports(flagged: pd.DataFrame, flag_summary: pd.DataFrame, top_bad: pd.DataFrame, rmse_threshold: float) -> None:
    high = flagged[flagged["high_rmse_top20_flag"]]
    top_primary = top_bad["primary_failure_type"].value_counts().to_dict()
    lines = [
        "# 阶段 3：B 轨道 RBF KRR 坏样本物理复查 v0.1",
        "",
        "## 为什么做",
        "",
        "clean-task 车辆-only 对照显示，B_response3s_strict_core 上 RBF KRR 是按验证集选出的车辆-only 参考候选，但坏样本图仍有明显多段和反向响应失败。因此这里不训练新模型，只复查它在 test 集上的失败类型。",
        "",
        "## 输入和边界",
        "",
        f"- 目标轨道：`{TARGET_TRACK}`。",
        f"- 目标模型：`{TARGET_MODEL}`。",
        f"- 目标 split：`{TARGET_SPLIT}`，共 {len(flagged)} 个样本。",
        "- 只使用 clean-task 逐样本指标表和响应任务 manifest；不使用生理、脑电、连续风格、驾驶员 ID 或服务器。",
        "",
        "## 核心发现",
        "",
        f"- high-RMSE top20% 阈值：sample RMSE >= {rmse_threshold:.6f}，对应 {len(high)} 个样本。",
        f"- 全部 B test 样本 mean RMSE={flagged['sample_rmse'].mean():.6f}，median RMSE={flagged['sample_rmse'].median():.6f}。",
        f"- high-RMSE top20% 样本 mean RMSE={high['sample_rmse'].mean():.6f}，mean GT peak={high['gt_peak_abs'].mean():.6f}。",
        "- top 坏样本的主要失败类型计数：" + json.dumps(top_primary, ensure_ascii=False),
        "",
        "## 失败标记汇总",
        "",
        simple_markdown_table(flag_summary),
        "",
        "## 结论",
        "",
        "B 轨道 RBF KRR 可以作为当前车辆-only 参考候选继续复查，但它还不能说明车辆历史已经充分解决失稳响应预测。最明显的剩余问题不是单一 RMSE，而是反向修正、多段修正、幅值/方向和尾段回正的组合错误。下一步应优先做结构化车辆-only 响应分解，而不是直接进入连续风格或生理有效性结论。",
        "",
        "## 推荐查看",
        "",
        f"1. `{TABLE_DIR / 'b_track_rbf_bad_sample_table.csv'}`",
        f"2. `{TABLE_DIR / 'b_track_rbf_top_bad_samples.csv'}`",
        f"3. `{FIG_DIR / 'b_track_rbf_failure_flag_rates.png'}`",
        f"4. `{FIG_DIR / 'b_track_rbf_top_bad_rmse.png'}`",
        f"5. `{FIG_DIR / 'b_track_rbf_peak_amp_scatter.png'}`",
    ]
    tech = "\n".join(lines) + "\n"
    (REPORT_ROOT / "stage03_vehicle_instability_clean_task_bad_sample_review_v0_1_cn.md").write_text(
        tech, encoding="utf-8"
    )

    user_lines = [
        "# 阶段 3 用户查看版：B 轨道车辆-only 坏样本复查 v0.1",
        "",
        "## 这一步为什么做",
        "",
        "上一轮结果里，3 秒响应覆盖任务的 RBF KRR 是当前最稳的车辆-only 候选，但预测图里仍然能看到不少反向修正、多段修正和大幅动作没有预测好。这里先把这些失败类型数清楚，避免只看 RMSE 就进入生理或风格阶段。",
        "",
        "## 这一步检查了什么",
        "",
        f"- 只检查 B 轨道 test 集 {len(flagged)} 个样本。",
        "- 只检查车辆-only RBF KRR，没有训练新模型。",
        "- 检查错侧、严重幅值不足、大幅响应漏召回、尾段漂移、零线穿越、反向修正、多段修正、峰值时间和启动延迟。",
        "",
        "## 目前发现了什么",
        "",
        f"- 最差 20% 的阈值是 RMSE >= {rmse_threshold:.3f}，共有 {len(high)} 个坏样本。",
        f"- 这 {len(high)} 个坏样本里，平均真实主峰幅值是 {high['gt_peak_abs'].mean():.3f}，说明坏样本并不只是微小噪声样本。",
        "- 主要剩余问题仍集中在结构化响应：反向修正、多段修正、幅值不足/错侧和尾段回正，而不是简单调一个 RMSE 损失就能完全解决。",
        "",
        "## 哪些结论可信",
        "",
        "可信的是：当前 B 轨道车辆-only RBF KRR 比旧的混合样本车辆-only 对照更适合作为下一步参考，但它仍有明确物理错误。",
        "",
        "## 哪些结果还不能下结论",
        "",
        "还不能说连续风格、生理或 EEG 有效，也不能说 KNN/template 是主线。A 轨道样本太少，B 轨道仍有结构错误，所以还需要车辆-only 结构化建模。",
        "",
        "## 下一步建议",
        "",
        "下一步优先做车辆-only 的响应分解：先预测方向、幅值、峰值时间、反向/多段修正类型，再预测轨迹。只有这个强车辆参考稳定后，才适合进入风格和生理增量验证。",
        "",
        "## 推荐查看",
        "",
        f"1. `{FIG_DIR / 'b_track_rbf_failure_flag_rates.png'}`",
        f"2. `{FIG_DIR / 'b_track_rbf_top_bad_rmse.png'}`",
        f"3. `{TABLE_DIR / 'b_track_rbf_top_bad_samples.csv'}`",
    ]
    (REPORT_ROOT / "stage03_vehicle_instability_clean_task_bad_sample_review_user_summary_cn.md").write_text(
        "\n".join(user_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    target = load_target_rows()
    flagged, rmse_threshold = add_failure_flags(target)
    flag_summary = build_flag_summary(flagged)
    by_morphology = build_group_summary(flagged, "eval_label_morphology")
    by_subject = build_group_summary(flagged, "subject")
    by_road_module = build_group_summary(flagged, "road_design_module_name")
    top_bad = build_top_bad(flagged)

    flagged.to_csv(TABLE_DIR / "b_track_rbf_bad_sample_table.csv", index=False, encoding="utf-8-sig")
    flag_summary.to_csv(TABLE_DIR / "b_track_rbf_failure_summary.csv", index=False, encoding="utf-8-sig")
    by_morphology.to_csv(TABLE_DIR / "b_track_rbf_failure_by_morphology.csv", index=False, encoding="utf-8-sig")
    by_subject.to_csv(TABLE_DIR / "b_track_rbf_failure_by_subject.csv", index=False, encoding="utf-8-sig")
    by_road_module.to_csv(TABLE_DIR / "b_track_rbf_failure_by_road_module.csv", index=False, encoding="utf-8-sig")
    top_bad.to_csv(TABLE_DIR / "b_track_rbf_top_bad_samples.csv", index=False, encoding="utf-8-sig")

    plot_flag_summary(flag_summary)
    plot_top_bad(top_bad)
    plot_peak_scatter(flagged)

    write_reports(flagged, flag_summary, top_bad, rmse_threshold)

    summary = {
        "target_track": TARGET_TRACK,
        "target_model": TARGET_MODEL,
        "target_split": TARGET_SPLIT,
        "n_samples": int(len(flagged)),
        "high_rmse_top20_threshold": rmse_threshold,
        "high_rmse_top20_n": int(flagged["high_rmse_top20_flag"].sum()),
        "mean_rmse": float(flagged["sample_rmse"].mean()),
        "median_rmse": float(flagged["sample_rmse"].median()),
        "flag_summary_path": str(TABLE_DIR / "b_track_rbf_failure_summary.csv"),
        "top_bad_path": str(TABLE_DIR / "b_track_rbf_top_bad_samples.csv"),
        "server_used": False,
        "credential_file_read": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "uses_subject_id_as_model_input": False,
        "new_training_run": False,
    }
    (LOG_DIR / "clean_task_bad_sample_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
