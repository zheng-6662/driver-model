# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
REPORT_ROOT = ROOT / "09_reports"
BASELINE_ROOT = ROOT / "03_baselines"
SAMPLE_ROOT = ROOT / "02_samples" / "vehicle_instability_highconf_v0_1"
SAMPLES_PATH = SAMPLE_ROOT / "tables" / "samples_master.csv"
BAD_ATTR_PATH = (
    BASELINE_ROOT
    / "stage03_vehicle_instability_bad_event_failure_attribution_v0_1"
    / "tables"
    / "bad_event_failure_attribution_table.csv"
)
OUTPUT_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_label_window_coverage_audit_v0_1"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
LOG_DIR = OUTPUT_ROOT / "logs"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_formal_baselines_v0_1 as formal_v01  # noqa: E402


WINDOWS = [
    "pre1_label2_event_trigger",
    "pre2_label2_old_main",
    "pre3_label3_response_coverage",
]
MAIN_2S = "pre2_label2_old_main"
DIAG_3S = "pre3_label3_response_coverage"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def bool_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.astype(bool).mean())


def markdown_code_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    part = df if max_rows is None else df.head(max_rows)
    return "```text\n" + part.to_string(index=False) + "\n```"


def finite_tail(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(valid[-1]) if valid.size else float("nan")


def value_at_time(values: np.ndarray, time_axis: np.ndarray, target_s: float) -> float:
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return float("nan")
    idx = int(np.nanargmin(np.abs(time_axis - target_s)))
    if valid[idx]:
        return float(values[idx])
    valid_idx = np.where(valid)[0]
    nearest = valid_idx[np.argmin(np.abs(time_axis[valid_idx] - target_s))]
    return float(values[int(nearest)])


def post_window_change(values: np.ndarray, time_axis: np.ndarray, start_s: float) -> dict[str, float]:
    valid = np.isfinite(values)
    mask = valid & (time_axis >= start_s)
    if mask.sum() < 2:
        return {
            "post2_range_abs": float("nan"),
            "post2_change_from_2s_abs": float("nan"),
        }
    post = values[mask].astype(np.float64)
    val_start = value_at_time(values, time_axis, start_s)
    return {
        "post2_range_abs": float(np.nanmax(post) - np.nanmin(post)),
        "post2_change_from_2s_abs": float(np.nanmax(np.abs(post - val_start))),
    }


def compute_window_metrics(samples: pd.DataFrame, window_id: str) -> pd.DataFrame:
    y, y_mask, _input_values, _input_time, label_time, meta = formal_v01.load_window(window_id, samples)
    rows: list[dict[str, Any]] = []
    label_end_default = float(label_time[-1])
    near_end_margin = max(0.20, min(0.35, 0.12 * label_end_default))
    for i in range(y.shape[0]):
        row = meta.iloc[i]
        valid = y_mask[i] & np.isfinite(y[i])
        gt = np.where(valid, y[i], np.nan)
        peak = eval_utils.peak_stats(gt, label_time)
        tail_signed = finite_tail(gt)
        tail_abs = abs(tail_signed) if math.isfinite(tail_signed) else float("nan")
        peak_abs = safe_float(peak["peak_abs"])
        label_end = safe_float(row.get("label_end_rel_s"), label_end_default)
        tail_over_peak = tail_abs / max(peak_abs, 1e-6) if math.isfinite(tail_abs) else float("nan")
        gt_peak_time = safe_float(peak["peak_time_s"])
        peak_near_label_end = bool(math.isfinite(gt_peak_time) and gt_peak_time >= label_end - near_end_margin)
        tail_unsettled = bool(
            math.isfinite(tail_over_peak)
            and peak_abs >= 0.25
            and tail_abs >= max(0.20, 0.30 * peak_abs)
        )
        event_end = safe_float(row.get("event_end_rel_s"))
        event_duration_exceeds_label = bool(math.isfinite(event_end) and event_end > label_end + 0.20)
        post2 = post_window_change(gt.astype(np.float64), label_time.astype(np.float64), 2.0)
        post2_substantial_change = bool(
            window_id == DIAG_3S
            and math.isfinite(post2["post2_change_from_2s_abs"])
            and peak_abs >= 0.25
            and post2["post2_change_from_2s_abs"] >= max(0.25, 0.20 * peak_abs)
        )
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "event_uid": str(row["event_uid"]),
                "subject": str(row.get("subject", "")),
                "session_stamp": str(row.get("session_stamp", "")),
                "window_config_id": window_id,
                "default_split": str(row.get("default_split", "")),
                "random_event_split": str(row.get("random_event_split", "")),
                "session_level_split": str(row.get("session_level_split", "")),
                "subject_level_split": str(row.get("subject_level_split", "")),
                "event_type": str(row.get("event_type", "")),
                "event_level": str(row.get("event_level", "")),
                "road_type_anchor": str(row.get("road_type_anchor", "")),
                "road_design_risk_class": str(row.get("road_design_risk_class", "")),
                "label_end_rel_s": label_end,
                "event_end_rel_s": event_end,
                "event_duration_s": safe_float(row.get("event_duration_s")),
                "label_valid_ratio": float(np.isfinite(gt).mean()),
                "gt_peak_abs": peak_abs,
                "gt_peak_signed": safe_float(peak["peak_signed"]),
                "gt_peak_time_s": gt_peak_time,
                "gt_peak_direction": int(peak["peak_direction"]),
                "gt_reversal_count": eval_utils.reversal_count(gt),
                "gt_tail_abs": tail_abs,
                "gt_tail_signed": tail_signed,
                "gt_tail_over_peak": tail_over_peak,
                "gt_peak_near_label_end": peak_near_label_end,
                "gt_tail_unsettled": tail_unsettled,
                "label_response_unsettled_flag": bool(peak_near_label_end or tail_unsettled),
                "event_duration_exceeds_label": event_duration_exceeds_label,
                "post2_range_abs": post2["post2_range_abs"],
                "post2_change_from_2s_abs": post2["post2_change_from_2s_abs"],
                "post2_substantial_change": post2_substantial_change,
            }
        )
    return pd.DataFrame(rows)


def build_event_policy_table(window_metrics: pd.DataFrame, bad_attr: pd.DataFrame) -> pd.DataFrame:
    main = window_metrics[window_metrics["window_config_id"] == MAIN_2S].copy()
    diag = window_metrics[window_metrics["window_config_id"] == DIAG_3S].copy()
    cols = [
        "event_uid",
        "subject",
        "session_stamp",
        "default_split",
        "session_level_split",
        "subject_level_split",
        "event_type",
        "event_level",
        "road_type_anchor",
        "road_design_risk_class",
        "event_end_rel_s",
        "event_duration_s",
        "gt_peak_abs",
        "gt_peak_time_s",
        "gt_tail_abs",
        "gt_tail_over_peak",
        "gt_peak_near_label_end",
        "gt_tail_unsettled",
        "label_response_unsettled_flag",
        "event_duration_exceeds_label",
    ]
    main = main[cols].rename(
        columns={
            "gt_peak_abs": "peak2_abs",
            "gt_peak_time_s": "peak2_time_s",
            "gt_tail_abs": "tail2_abs",
            "gt_tail_over_peak": "tail2_over_peak",
            "gt_peak_near_label_end": "peak2_near_end",
            "gt_tail_unsettled": "tail2_unsettled",
            "label_response_unsettled_flag": "label2_response_unsettled",
            "event_duration_exceeds_label": "event_exceeds_2s_label",
        }
    )
    diag_cols = [
        "event_uid",
        "gt_peak_abs",
        "gt_peak_time_s",
        "gt_tail_abs",
        "gt_tail_over_peak",
        "gt_peak_near_label_end",
        "gt_tail_unsettled",
        "label_response_unsettled_flag",
        "event_duration_exceeds_label",
        "post2_range_abs",
        "post2_change_from_2s_abs",
        "post2_substantial_change",
    ]
    diag = diag[diag_cols].rename(
        columns={
            "gt_peak_abs": "peak3_abs",
            "gt_peak_time_s": "peak3_time_s",
            "gt_tail_abs": "tail3_abs",
            "gt_tail_over_peak": "tail3_over_peak",
            "gt_peak_near_label_end": "peak3_near_end",
            "gt_tail_unsettled": "tail3_unsettled",
            "label_response_unsettled_flag": "label3_response_unsettled",
            "event_duration_exceeds_label": "event_exceeds_3s_label",
        }
    )
    out = main.merge(diag, on="event_uid", how="left")
    out["peak_after_2s_in_3s_label"] = out["peak3_time_s"] > 2.0
    out["peak_gain_after_2s_abs"] = (out["peak3_abs"] - out["peak2_abs"]).clip(lower=0.0)
    out["peak_gain_after_2s_ratio"] = out["peak_gain_after_2s_abs"] / out["peak3_abs"].replace(0, np.nan)
    out["old_2s_may_miss_future_peak"] = (
        (out["peak_after_2s_in_3s_label"])
        & (out["peak3_abs"] >= 0.50)
        & (out["peak_gain_after_2s_ratio"] >= 0.10)
    )
    out["old_2s_may_miss_future_change"] = out["post2_substantial_change"].fillna(False).astype(bool)
    out["label_window_2s_needs_review"] = (
        out["old_2s_may_miss_future_peak"]
        | out["old_2s_may_miss_future_change"]
        | out["label2_response_unsettled"].fillna(False).astype(bool)
    )
    # Event duration is a context flag: the non-steering instability segment can last
    # longer than the response label without necessarily invalidating the label.
    out["label3_still_needs_review"] = out["label3_response_unsettled"].fillna(False).astype(bool)
    conditions = [
        out["old_2s_may_miss_future_peak"],
        out["old_2s_may_miss_future_change"],
        out["label3_still_needs_review"],
        out["label_window_2s_needs_review"],
    ]
    choices = [
        "use_3s_or_longer_label_for_late_peak_review",
        "use_3s_label_or_split_continuing_response",
        "review_continuous_event_or_longer_than_3s",
        "review_2s_tail_or_anchor",
    ]
    out["recommended_window_policy"] = np.select(conditions, choices, default="two_second_label_probably_ok")

    if not bad_attr.empty:
        bad_cols = [
            "event_uid",
            "recurrence_rank",
            "primary_attribution",
            "mean_sample_rmse",
            "vehicle_only_structure_gap",
            "label_window_may_be_short",
            "consensus_reversal_failure",
            "figure_png",
        ]
        bad = bad_attr[bad_cols].copy()
        bad["top_recurrent_bad_event"] = True
        out = out.merge(bad, on="event_uid", how="left")
    else:
        out["top_recurrent_bad_event"] = False
    out["top_recurrent_bad_event"] = out["top_recurrent_bad_event"].where(
        out["top_recurrent_bad_event"].notna(), False
    ).astype(bool)
    return out


def summarize_windows(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_id, part in window_metrics.groupby("window_config_id", sort=False):
        rows.append(
            {
                "window_config_id": window_id,
                "n_samples": int(len(part)),
                "label_end_rel_s": float(part["label_end_rel_s"].median()),
                "peak_near_end_rate": bool_rate(part["gt_peak_near_label_end"]),
                "tail_unsettled_rate": bool_rate(part["gt_tail_unsettled"]),
                "response_unsettled_rate": bool_rate(part["label_response_unsettled_flag"]),
                "event_duration_exceeds_label_rate": bool_rate(part["event_duration_exceeds_label"]),
                "median_peak_abs": float(part["gt_peak_abs"].median()),
                "median_peak_time_s": float(part["gt_peak_time_s"].median()),
                "median_tail_over_peak": float(part["gt_tail_over_peak"].median()),
                "median_reversal_count": float(part["gt_reversal_count"].median()),
                "mean_label_valid_ratio": float(part["label_valid_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_event_policy(event_policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_counts = (
        event_policy["recommended_window_policy"]
        .value_counts(dropna=False)
        .rename_axis("recommended_window_policy")
        .reset_index(name="n_events")
    )
    policy_counts["rate"] = policy_counts["n_events"] / len(event_policy)

    split_summary = []
    for split_col in ["default_split", "session_level_split", "subject_level_split"]:
        for split_name, part in event_policy.groupby(split_col, dropna=False):
            split_summary.append(
                {
                    "split_strategy": split_col,
                    "split_name": split_name,
                    "n_events": int(len(part)),
                    "old_2s_may_miss_future_peak_rate": bool_rate(part["old_2s_may_miss_future_peak"]),
                    "old_2s_may_miss_future_change_rate": bool_rate(part["old_2s_may_miss_future_change"]),
                    "label2_needs_review_rate": bool_rate(part["label_window_2s_needs_review"]),
                    "label3_still_needs_review_rate": bool_rate(part["label3_still_needs_review"]),
                }
            )
    split_summary_df = pd.DataFrame(split_summary)

    subject_summary = []
    for subject, part in event_policy.groupby("subject", dropna=False):
        subject_summary.append(
            {
                "subject": subject,
                "n_events": int(len(part)),
                "label2_needs_review_rate": bool_rate(part["label_window_2s_needs_review"]),
                "label3_still_needs_review_rate": bool_rate(part["label3_still_needs_review"]),
                "late_peak_after_2s_rate": bool_rate(part["old_2s_may_miss_future_peak"]),
                "post2_change_rate": bool_rate(part["old_2s_may_miss_future_change"]),
                "top_recurrent_bad_events": int(part["top_recurrent_bad_event"].sum()),
            }
        )
    subject_summary_df = pd.DataFrame(subject_summary).sort_values(
        ["top_recurrent_bad_events", "label2_needs_review_rate", "n_events"],
        ascending=[False, False, False],
    )
    return policy_counts, split_summary_df, subject_summary_df


def plot_window_rates(window_summary: pd.DataFrame) -> Path:
    plot_cols = [
        "peak_near_end_rate",
        "tail_unsettled_rate",
        "response_unsettled_rate",
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(window_summary))
    width = 0.23
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for j, col in enumerate(plot_cols):
        ax.bar(x + (j - 1.0) * width, window_summary[col], width=width, label=col, color=colors[j])
    ax.set_xticks(x)
    ax.set_xticklabels(window_summary["window_config_id"], rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Response-label coverage flags by window")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "label_window_coverage_rates_by_window.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_policy_counts(policy_counts: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    part = policy_counts.sort_values("n_events", ascending=True)
    ax.barh(part["recommended_window_policy"], part["n_events"], color="#4c78a8")
    ax.set_xlabel("Events")
    ax.set_title("Recommended label/window policy from coverage audit")
    for i, (_idx, row) in enumerate(part.iterrows()):
        ax.text(float(row["n_events"]) + 3, i, f"{int(row['n_events'])} ({row['rate']:.1%})", va="center")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "label_window_policy_counts.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_peak_tail_scatter(event_policy: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    base = event_policy[~event_policy["top_recurrent_bad_event"]]
    bad = event_policy[event_policy["top_recurrent_bad_event"]]
    ax.scatter(
        base["peak3_time_s"],
        base["tail3_over_peak"],
        s=18,
        alpha=0.45,
        color="#9aa0a6",
        label="all other events",
    )
    if len(bad):
        ax.scatter(
            bad["peak3_time_s"],
            bad["tail3_over_peak"],
            s=55,
            alpha=0.90,
            color="#d62728",
            label="Top recurrent bad events",
        )
        for _, row in bad.iterrows():
            ax.text(row["peak3_time_s"], row["tail3_over_peak"], str(int(row["recurrence_rank"])), fontsize=8)
    ax.axvline(2.0, color="#1f77b4", linestyle="--", linewidth=1.2, label="2s label end")
    ax.axvline(2.75, color="#ff7f0e", linestyle=":", linewidth=1.2, label="near 3s end")
    ax.axhline(0.30, color="#2ca02c", linestyle="--", linewidth=1.1, label="tail/peak=0.30")
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(-0.02, min(1.65, max(1.05, float(np.nanpercentile(event_policy["tail3_over_peak"], 98)) + 0.1)))
    ax.set_xlabel("3s label peak time relative to anchor (s)")
    ax.set_ylabel("3s label tail abs / peak abs")
    ax.set_title("Peak timing and tail state in 3s response-coverage labels")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "label_window_peak_tail_scatter_pre3.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_reports(
    window_summary: pd.DataFrame,
    policy_counts: pd.DataFrame,
    event_policy: pd.DataFrame,
    bad_overlay: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    n_events = len(event_policy)
    late_peak_n = int(event_policy["old_2s_may_miss_future_peak"].sum())
    post2_change_n = int(event_policy["old_2s_may_miss_future_change"].sum())
    label2_review_n = int(event_policy["label_window_2s_needs_review"].sum())
    label3_review_n = int(event_policy["label3_still_needs_review"].sum())
    bad_n = int(event_policy["top_recurrent_bad_event"].sum())
    bad_label2_review_n = int(bad_overlay["label_window_2s_needs_review"].sum()) if len(bad_overlay) else 0
    bad_label3_review_n = int(bad_overlay["label3_still_needs_review"].sum()) if len(bad_overlay) else 0
    pre2 = window_summary[window_summary["window_config_id"] == MAIN_2S].iloc[0]
    pre3 = window_summary[window_summary["window_config_id"] == DIAG_3S].iloc[0]
    policy_md = markdown_code_table(policy_counts)
    window_md = markdown_code_table(window_summary)

    user = f"""# 阶段 3 用户查看版：标签窗口覆盖审计 v0.1

## 为什么做

上一轮 Top 12 复发坏样本里，很多事件被自动归因为“标签窗口或样本规则需要复核”。这一步不训练新模型，只检查当前 2 秒标签窗口是否经常没有覆盖完整方向盘响应。

## 检查了什么

- 正式高置信失稳样本 `vehicle_instability_highconf_v0_1` 的 {n_events} 个事件。
- 当前主窗口 `pre2_label2_old_main`：事件前 2 秒车辆历史，预测事件后 2 秒方向盘响应。
- 诊断窗口 `pre3_label3_response_coverage`：事件前 3 秒车辆历史，预测事件后 3 秒方向盘响应。
- 是否出现 2 秒之后还有更大峰值、2 秒之后方向盘仍有明显变化、3 秒末端仍未稳定、事件持续时间超过标签窗口等情况。

## 目前发现

- {late_peak_n}/{n_events} 个事件在 3 秒标签里显示主峰出现在 2 秒之后，说明旧 2 秒标签可能漏掉后续更大响应。
- {post2_change_n}/{n_events} 个事件在 2 秒之后仍有明显方向盘变化。
- {label2_review_n}/{n_events} 个事件被标记为“2 秒标签需要复核”。
- {label3_review_n}/{n_events} 个事件即使用 3 秒标签仍需要复核，通常代表连续失稳、长事件或尾段没有回正。
- Top 复发坏样本中有 {bad_label2_review_n}/{bad_n} 个需要复核 2 秒窗口，{bad_label3_review_n}/{bad_n} 个即使 3 秒窗口也仍需复核。

## 哪些结果可信

这一步只读取已生成的 `samples_master.csv` 和处理后的车辆标签数组，不使用生理、脑电、连续风格、驾驶员 ID，也没有训练新模型。它适合用来决定下一步是否应该修样本规则和标签窗口。

## 哪些结果还不能下结论

尾段没有回到 0 不一定都是错误。某些真实驾驶响应可能本来就需要保持方向盘角度，或者事件本身持续超过 3 秒。因此这些旗标只能说明“需要复核”，不能直接说明样本无效。

## 下一阶段是否可以继续

建议暂时不要继续堆新模型。下一步应先决定正式主标签到底采用 2 秒即时响应、3 秒响应覆盖，还是把长失稳事件拆成“启动响应”和“持续控制”两个任务。这个决定会影响后续所有车辆、风格和生理增量实验。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_event_policy_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_bad_event_overlay.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_policy_counts.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_peak_tail_scatter_pre3.png`
"""
    (REPORT_ROOT / "stage03_vehicle_instability_label_window_coverage_audit_user_summary_cn.md").write_text(
        user, encoding="utf-8"
    )

    technical = f"""# 阶段 3：标签窗口覆盖审计 v0.1

## 目的

复发坏样本归因显示，当前失败不一定都来自车辆-only 模型结构，也可能来自 2 秒标签窗口覆盖不足、连续事件未拆分或锚点附近已发生响应。本审计把该问题扩展到正式高置信失稳样本全集。

## 输入

- 样本清单：`{SAMPLES_PATH.as_posix()}`
- 处理后数组：`pre1_label2_event_trigger.npz`、`pre2_label2_old_main.npz`、`pre3_label3_response_coverage.npz`
- 复发坏样本归因表：`{BAD_ATTR_PATH.as_posix()}`

## 窗口级结果

{window_md}

## 事件级窗口策略计数

{policy_md}

## 关键数字

- 事件数：{n_events}
- 2 秒后出现更大峰值：{late_peak_n}/{n_events} ({late_peak_n / n_events:.2%})
- 2 秒后仍有明显变化：{post2_change_n}/{n_events} ({post2_change_n / n_events:.2%})
- 2 秒标签需复核：{label2_review_n}/{n_events} ({label2_review_n / n_events:.2%})
- 3 秒标签仍需复核：{label3_review_n}/{n_events} ({label3_review_n / n_events:.2%})
- 主 2 秒窗口 response_unsettled_rate：{pre2['response_unsettled_rate']:.2%}
- 3 秒诊断窗口 response_unsettled_rate：{pre3['response_unsettled_rate']:.2%}

## 图表

- 窗口旗标率：`{figures['rates'].as_posix()}`
- 推荐窗口策略计数：`{figures['policy'].as_posix()}`
- 3 秒峰值时间和尾段散点：`{figures['scatter'].as_posix()}`

## 解释边界

本审计不训练模型，不评估连续风格、生理或 EEG 有效性。`label_window_2s_needs_review` 和 `label3_still_needs_review` 是规则旗标，不是人工最终判定。长事件、保持转向和真实连续控制会让尾段不回零，因此下一步需要把任务定义拆清楚。

## 建议

1. 如果目标是“事件触发后即时响应”，保留 2 秒标签，但需要单独处理持续失稳和尾段未稳定样本。
2. 如果目标是“覆盖完整方向盘响应”，应把 3 秒或更长窗口作为正式候选，并重新跑车辆-only 基线。
3. 对 3 秒仍未稳定的样本，优先考虑事件拆分或长事件标签，而不是直接把这些样本丢给生理模型解释。
"""
    (REPORT_ROOT / "stage03_vehicle_instability_label_window_coverage_audit_v0_1_cn.md").write_text(
        technical, encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    samples = pd.read_csv(SAMPLES_PATH)
    metrics = [compute_window_metrics(samples, w) for w in WINDOWS]
    window_metrics = pd.concat(metrics, ignore_index=True)
    bad_attr = pd.read_csv(BAD_ATTR_PATH) if BAD_ATTR_PATH.exists() else pd.DataFrame()
    event_policy = build_event_policy_table(window_metrics, bad_attr)
    window_summary = summarize_windows(window_metrics)
    policy_counts, split_summary, subject_summary = summarize_event_policy(event_policy)
    bad_overlay = event_policy[event_policy["top_recurrent_bad_event"]].copy()

    window_metrics.to_csv(TABLE_DIR / "label_window_sample_metrics.csv", index=False, encoding="utf-8-sig")
    event_policy.to_csv(TABLE_DIR / "label_window_event_policy_table.csv", index=False, encoding="utf-8-sig")
    window_summary.to_csv(TABLE_DIR / "label_window_window_summary.csv", index=False, encoding="utf-8-sig")
    policy_counts.to_csv(TABLE_DIR / "label_window_policy_counts.csv", index=False, encoding="utf-8-sig")
    split_summary.to_csv(TABLE_DIR / "label_window_split_summary.csv", index=False, encoding="utf-8-sig")
    subject_summary.to_csv(TABLE_DIR / "label_window_subject_summary.csv", index=False, encoding="utf-8-sig")
    bad_overlay.to_csv(TABLE_DIR / "label_window_bad_event_overlay.csv", index=False, encoding="utf-8-sig")

    figures = {
        "rates": plot_window_rates(window_summary),
        "policy": plot_policy_counts(policy_counts),
        "scatter": plot_peak_tail_scatter(event_policy),
    }
    write_reports(window_summary, policy_counts, event_policy, bad_overlay, figures)

    n_events = int(len(event_policy))
    summary = {
        "n_events": n_events,
        "n_window_sample_rows": int(len(window_metrics)),
        "late_peak_after_2s_n": int(event_policy["old_2s_may_miss_future_peak"].sum()),
        "post2_substantial_change_n": int(event_policy["old_2s_may_miss_future_change"].sum()),
        "label2_needs_review_n": int(event_policy["label_window_2s_needs_review"].sum()),
        "label3_still_needs_review_n": int(event_policy["label3_still_needs_review"].sum()),
        "top_bad_events_n": int(event_policy["top_recurrent_bad_event"].sum()),
        "top_bad_label2_needs_review_n": int(
            event_policy.loc[event_policy["top_recurrent_bad_event"], "label_window_2s_needs_review"].sum()
        ),
        "top_bad_label3_still_needs_review_n": int(
            event_policy.loc[event_policy["top_recurrent_bad_event"], "label3_still_needs_review"].sum()
        ),
        "window_summary_path": str(TABLE_DIR / "label_window_window_summary.csv").replace("\\", "/"),
        "event_policy_path": str(TABLE_DIR / "label_window_event_policy_table.csv").replace("\\", "/"),
        "bad_overlay_path": str(TABLE_DIR / "label_window_bad_event_overlay.csv").replace("\\", "/"),
        "figures": {k: str(v).replace("\\", "/") for k, v in figures.items()},
        "server_used": False,
        "credential_file_read": False,
        "uses_subject_id_as_model_input": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "model_training_performed": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "label_window_coverage_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
