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
CURVE_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_bad_event_curve_review_v0_1"
FIGURE_INDEX_PATH = CURVE_ROOT / "tables" / "bad_event_curve_figure_index.csv"
MODEL_ERROR_PATH = CURVE_ROOT / "tables" / "bad_event_curve_model_error_table.csv"
OUTPUT_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_bad_event_failure_attribution_v0_1"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
LOG_DIR = OUTPUT_ROOT / "logs"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_formal_baselines_v0_1 as formal_v01  # noqa: E402


RAW_COLS = [
    "StorageTime",
    "zx|SteeringWheel",
    "zx|ay",
    "zx|roll",
    "zx|vyaw",
    "zx|v_km/h",
    "zx1|v_km/h",
    "zx1|lanecurvatureXY",
    "zx1|lateraldistance",
    "zx|lateraldistance",
]
ATTRIBUTION_FLAGS = [
    "label_window_may_be_short",
    "gt_peak_near_label_end",
    "gt_tail_unsettled",
    "event_continues_after_label",
    "pre_anchor_steer_active",
    "pre_anchor_dynamics_active",
    "raw_support_low",
    "consensus_amp_under",
    "consensus_wrong_side",
    "consensus_reversal_failure",
    "vehicle_only_structure_gap",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def to_seconds(storage_time: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(storage_time, errors="coerce")
    out = np.full(len(parsed), np.nan, dtype=np.float64)
    valid = parsed.notna().to_numpy()
    if valid.any():
        out[valid] = parsed[valid].astype("int64").to_numpy(dtype=np.float64) / 1e9
    return out


def read_raw_window(row: pd.Series, pre_s: float = -3.0, post_s: float | None = None) -> pd.DataFrame:
    raw_path = Path(str(row.get("vehicle_raw_absolute_path") or row.get("vehicle_absolute_path")))
    if not raw_path.exists():
        return pd.DataFrame()
    header = pd.read_csv(raw_path, nrows=0)
    usecols = [c for c in RAW_COLS if c in header.columns]
    if "StorageTime" not in usecols:
        return pd.DataFrame()
    raw = pd.read_csv(raw_path, usecols=usecols)
    t_abs = to_seconds(raw["StorageTime"])
    anchor_abs = safe_float(row.get("anchor_time_abs_storage_s"))
    if not math.isfinite(anchor_abs):
        return pd.DataFrame()
    label_end = safe_float(row.get("label_end_rel_s"), 2.0)
    post = max(label_end + 0.75, 3.5) if post_s is None else post_s
    raw = raw.copy()
    raw["t_rel_s"] = t_abs - anchor_abs
    raw = raw[(raw["t_rel_s"] >= pre_s) & (raw["t_rel_s"] <= post)].copy()
    for col in raw.columns:
        if col != "StorageTime":
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw


def robust_scaled(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 4:
        return np.full_like(arr, np.nan, dtype=np.float64)
    center = np.nanmedian(arr)
    scale = np.nanpercentile(arr, 90) - np.nanpercentile(arr, 10)
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = np.nanstd(arr)
    if not math.isfinite(scale) or scale <= 1e-9:
        return np.full_like(arr, np.nan, dtype=np.float64)
    return (arr - center) / scale


def finite_peak(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.nanmax(np.abs(finite))) if finite.size else float("nan")


def raw_evidence(row: pd.Series, gt_peak_abs: float) -> dict[str, Any]:
    raw = read_raw_window(row)
    if raw.empty:
        return {
            "raw_available": False,
            "raw_support_low": True,
            "pre_anchor_steer_ratio": float("nan"),
            "pre_anchor_dynamic_ratio": float("nan"),
            "pre_anchor_steer_active": False,
            "pre_anchor_dynamics_active": False,
            "raw_support_min_core": 0.0,
        }
    t = raw["t_rel_s"].to_numpy(dtype=np.float64)
    focus = np.isfinite(t) & (t >= -3.0) & (t <= max(safe_float(row.get("label_end_rel_s"), 2.0), 3.0))
    core_cols = [c for c in ["zx|SteeringWheel", "zx|ay", "zx|roll", "zx|vyaw"] if c in raw.columns]
    support = {}
    for col in core_cols:
        vals = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float64)
        support[col] = float(np.isfinite(vals[focus]).mean()) if focus.any() else 0.0
    raw_support_min = min(support.values()) if support else 0.0

    pre_mask = np.isfinite(t) & (t >= -0.75) & (t < 0.0)
    post_mask = np.isfinite(t) & (t >= 0.0) & (t <= 1.5)
    if "zx|SteeringWheel" in raw.columns and math.isfinite(gt_peak_abs) and gt_peak_abs > 1e-6:
        steer = pd.to_numeric(raw["zx|SteeringWheel"], errors="coerce").to_numpy(dtype=np.float64)
        anchor_steer = safe_float(row.get("anchor_steer"))
        if math.isfinite(anchor_steer):
            steer = steer - anchor_steer
        pre_steer = finite_peak(steer[pre_mask])
        pre_steer_ratio = pre_steer / max(gt_peak_abs, 1e-6)
    else:
        pre_steer_ratio = float("nan")

    dyn_series = []
    for col in ["zx|ay", "zx|roll", "zx|vyaw"]:
        if col in raw.columns:
            vals = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float64)
            scaled = robust_scaled(vals)
            dyn_series.append(scaled)
    if dyn_series:
        dyn_stack = np.abs(np.vstack(dyn_series))
        dyn = np.full(dyn_stack.shape[1], np.nan, dtype=np.float64)
        has_finite = np.isfinite(dyn_stack).any(axis=0)
        if has_finite.any():
            dyn[has_finite] = np.nanmax(dyn_stack[:, has_finite], axis=0)
        pre_dyn = finite_peak(dyn[pre_mask])
        post_dyn = finite_peak(dyn[post_mask])
        pre_dyn_ratio = pre_dyn / max(post_dyn, 1e-6) if math.isfinite(post_dyn) else float("nan")
    else:
        pre_dyn_ratio = float("nan")

    return {
        "raw_available": True,
        "raw_support_low": raw_support_min < 0.20,
        "pre_anchor_steer_ratio": pre_steer_ratio,
        "pre_anchor_dynamic_ratio": pre_dyn_ratio,
        "pre_anchor_steer_active": bool(math.isfinite(pre_steer_ratio) and pre_steer_ratio >= 0.25),
        "pre_anchor_dynamics_active": bool(math.isfinite(pre_dyn_ratio) and pre_dyn_ratio >= 0.85),
        "raw_support_min_core": raw_support_min,
    }


def load_window_cache(samples: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for window_id in sorted(samples["window_config_id"].dropna().unique()):
        y, y_mask, input_values, input_time, label_time, meta = formal_v01.load_window(window_id, samples)
        cache[window_id] = {
            "y": y,
            "y_mask": y_mask,
            "label_time": label_time,
            "meta": meta,
            "sample_to_idx": {str(v): int(i) for i, v in enumerate(meta["sample_id"].astype(str).to_numpy())},
        }
    return cache


def gt_evidence(row: pd.Series, window_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    window_id = str(row["window_config_id"])
    sample_id = str(row["sample_id"])
    cache = window_cache[window_id]
    idx = cache["sample_to_idx"][sample_id]
    y = cache["y"][idx]
    mask = cache["y_mask"][idx] & np.isfinite(y)
    label_time = cache["label_time"]
    gt = np.where(mask, y, np.nan)
    peak = eval_utils.peak_stats(gt, label_time)
    gt_rev = eval_utils.reversal_count(gt)
    valid = gt[np.isfinite(gt)]
    label_end = safe_float(row.get("label_end_rel_s"), float(label_time[-1]))
    tail_abs = abs(float(valid[-1])) if valid.size else float("nan")
    peak_abs = safe_float(peak["peak_abs"])
    peak_time = safe_float(peak["peak_time_s"])
    tail_ratio = tail_abs / max(peak_abs, 1e-6) if math.isfinite(tail_abs) and math.isfinite(peak_abs) else float("nan")
    event_end_rel = safe_float(row.get("event_end_rel_s")) - safe_float(row.get("anchor_time_rel_s"))
    return {
        "gt_peak_abs": peak_abs,
        "gt_peak_time_s": peak_time,
        "gt_reversal_count": int(gt_rev),
        "gt_tail_abs": tail_abs,
        "gt_tail_over_peak": tail_ratio,
        "gt_peak_near_label_end": bool(math.isfinite(peak_time) and (label_end - peak_time) <= 0.35),
        "gt_tail_unsettled": bool(math.isfinite(tail_ratio) and tail_ratio >= 0.45),
        "event_end_rel_to_anchor_s": event_end_rel,
        "event_continues_after_label": bool(math.isfinite(event_end_rel) and event_end_rel > label_end + 0.25),
        "label_end_rel_s": label_end,
    }


def aggregate_model_evidence(error_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_uid, grp in error_df.groupby("event_uid", sort=False):
        best = grp.sort_values("sample_rmse").iloc[0]
        rows.append(
            {
                "event_uid": event_uid,
                "model_count": int(len(grp)),
                "mean_sample_rmse": float(grp["sample_rmse"].mean()),
                "best_model": best["model_name"],
                "best_sample_rmse": float(best["sample_rmse"]),
                "severe_amp_under_rate": float(grp["severe_amp_under"].mean()),
                "wrong_side_rate": float(grp["wrong_side"].mean()),
                "reversal_exact_rate": float(grp["reversal_count_exact"].mean()),
                "mean_pred_over_gt_peak_ratio": float(grp["pred_over_gt_peak_ratio"].replace([np.inf, -np.inf], np.nan).mean()),
            }
        )
    return pd.DataFrame(rows)


def assign_attribution(row: pd.Series) -> tuple[str, str, str]:
    reasons = []
    next_steps = []
    if row["label_window_may_be_short"]:
        reasons.append("标签窗口/事件持续时间需要复核")
        next_steps.append("先检查是否要延长标签窗口或截断长事件")
    if row["pre_anchor_steer_active"] or row["pre_anchor_dynamics_active"]:
        reasons.append("锚点附近事件可能已经开始")
        next_steps.append("复核锚点是否偏晚或是否属于连续失稳片段")
    if row["raw_support_low"]:
        reasons.append("原始车辆核心信号有效点偏少")
        next_steps.append("回到原始 CSV/插值规则检查信号质量")
    if row["vehicle_only_structure_gap"]:
        reasons.append("车辆-only 候选共同漏幅值/反向修正")
        next_steps.append("优先设计响应分解或关键点+残差车辆模型")
    if row["consensus_wrong_side"]:
        reasons.append("多个车辆-only 候选错侧")
        next_steps.append("检查方向标签、锚点与事件类型是否一致")
    if not reasons:
        reasons.append("未发现明显锚点/窗口/质量问题，但仍是高误差样本")
        next_steps.append("纳入结构化车辆模型和困难样本识别验证")

    if row["label_window_may_be_short"] or row["pre_anchor_steer_active"] or row["pre_anchor_dynamics_active"] or row["raw_support_low"]:
        primary = "sample_rule_or_raw_signal_review"
    elif row["vehicle_only_structure_gap"] or row["consensus_wrong_side"]:
        primary = "vehicle_only_model_structure_gap"
    else:
        primary = "hard_vehicle_only_case"
    return primary, "；".join(reasons), "；".join(dict.fromkeys(next_steps))


def build_attribution_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(SAMPLES_PATH)
    fig_idx = pd.read_csv(FIGURE_INDEX_PATH)
    error_df = pd.read_csv(MODEL_ERROR_PATH)
    model_ev = aggregate_model_evidence(error_df)
    window_cache = load_window_cache(samples)
    sample_by_id = samples.set_index("sample_id", drop=False)

    rows = []
    for fig_row in fig_idx.itertuples(index=False):
        sample_id = str(fig_row.sample_id)
        sample = sample_by_id.loc[sample_id]
        gt = gt_evidence(sample, window_cache)
        raw = raw_evidence(sample, gt["gt_peak_abs"])
        model = model_ev[model_ev["event_uid"] == fig_row.event_uid].iloc[0].to_dict()
        flags = {
            "label_window_may_be_short": bool(
                gt["gt_peak_near_label_end"] or gt["gt_tail_unsettled"]
            ),
            "consensus_amp_under": bool(model["severe_amp_under_rate"] >= 0.80),
            "consensus_wrong_side": bool(model["wrong_side_rate"] >= 0.60),
            "consensus_reversal_failure": bool(
                gt["gt_reversal_count"] >= 2 and model["reversal_exact_rate"] <= 0.20
            ),
        }
        flags["vehicle_only_structure_gap"] = bool(
            (flags["consensus_amp_under"] or model["mean_pred_over_gt_peak_ratio"] < 0.55)
            and flags["consensus_reversal_failure"]
        )
        row = {
            "recurrence_rank": int(fig_row.recurrence_rank),
            "event_uid": fig_row.event_uid,
            "sample_id": sample_id,
            "subject": fig_row.subject,
            "session_stamp": fig_row.session_stamp,
            "config_id": fig_row.config_id,
            "window_config_id": fig_row.window_config_id,
            "figure_png": fig_row.figure_png,
            **gt,
            **raw,
            **model,
            **flags,
        }
        primary, reason, next_step = assign_attribution(pd.Series(row))
        row["primary_attribution"] = primary
        row["reason_cn"] = reason
        row["recommended_next_step_cn"] = next_step
        rows.append(row)

    attribution = pd.DataFrame(rows).sort_values("recurrence_rank").reset_index(drop=True)
    flag_counts = pd.DataFrame(
        [{"flag": flag, "count": int(attribution[flag].sum()), "rate": float(attribution[flag].mean())}
         for flag in ATTRIBUTION_FLAGS]
    )
    primary_counts = (
        attribution.groupby("primary_attribution", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return attribution, flag_counts, primary_counts


def plot_flag_heatmap(attribution: pd.DataFrame) -> Path:
    matrix = attribution[ATTRIBUTION_FLAGS].astype(int).to_numpy()
    y_labels = [f"#{int(r.recurrence_rank)} {r.subject}" for r in attribution.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(13.5, 5.8), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(ATTRIBUTION_FLAGS)), ATTRIBUTION_FLAGS, rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels)), y_labels)
    ax.set_title("Bad event failure attribution flags")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(j, i, "1", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, ticks=[0, 1], label="flag")
    out = FIG_DIR / "bad_event_failure_attribution_flags.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_primary_counts(primary_counts: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.3), constrained_layout=True)
    ax.barh(primary_counts["primary_attribution"], primary_counts["count"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("Event count")
    ax.set_title("Primary failure attribution among Top 12 recurrent bad events")
    for y, count in enumerate(primary_counts["count"]):
        ax.text(float(count) + 0.05, y, str(int(count)), va="center", fontsize=9)
    out = FIG_DIR / "bad_event_primary_attribution_counts.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def table_to_md(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    sub = df[columns].head(max_rows).copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in sub.to_numpy()]
    return "\n".join([header, sep] + rows)


def write_reports(
    attribution: pd.DataFrame,
    flag_counts: pd.DataFrame,
    primary_counts: pd.DataFrame,
    flag_heatmap: Path,
    primary_fig: Path,
) -> None:
    sample_review_n = int((attribution["primary_attribution"] == "sample_rule_or_raw_signal_review").sum())
    structure_gap_n = int((attribution["primary_attribution"] == "vehicle_only_model_structure_gap").sum())
    report = f"""# 阶段 3：复发坏样本失败来源归因 v0.1

生成时间：2026-05-12

## 目的

对复发坏样本 Top 12 做规则化归因，区分下一步应该先回到阶段 2 修锚点/窗口/原始信号，还是可以进入阶段 3 的结构化车辆模型。此步骤只使用车辆-only 结果和原始车辆片段，不使用生理、脑电、连续风格或驾驶员 ID 作为模型输入。

## 输入

- 曲线图索引：`{FIGURE_INDEX_PATH.as_posix()}`
- 模型逐事件误差表：`{MODEL_ERROR_PATH.as_posix()}`
- 正式样本清单：`{SAMPLES_PATH.as_posix()}`
- 原始车辆 CSV：按样本清单中的路径只读局部片段。

## 规则

- 标签窗口可能偏短：GT 峰值接近标签末端、尾段仍未回正，或事件持续时间超过标签窗口。
- 锚点可能偏晚：锚点前 0.75 秒内方向盘已经有明显响应，或非方向盘车辆动力学在锚点前已经很活跃。
- 原始信号需复核：核心车辆信号有效点比例过低。
- 车辆-only 结构不足：多数候选严重幅值不足，并且 GT 有反向/多段结构但模型反向修正计数基本不匹配。

## 主要结果

- Top 12 中，`sample_rule_or_raw_signal_review` 数量={sample_review_n}。
- Top 12 中，`vehicle_only_model_structure_gap` 数量={structure_gap_n}。
- 这说明不能直接跳到风格/生理阶段；下一步仍应先完成车辆-only 错误来源清理和结构化基线设计。

## 归因表

{table_to_md(attribution, [
    "recurrence_rank",
    "subject",
    "config_id",
    "primary_attribution",
    "gt_peak_abs",
    "gt_peak_time_s",
    "gt_tail_over_peak",
    "severe_amp_under_rate",
    "reversal_exact_rate",
    "reason_cn",
], max_rows=12)}

## 产物

- 归因明细表：`{(TABLE_DIR / "bad_event_failure_attribution_table.csv").as_posix()}`
- 归因旗标统计：`{(TABLE_DIR / "bad_event_failure_flag_counts.csv").as_posix()}`
- 主归因统计：`{(TABLE_DIR / "bad_event_primary_attribution_counts.csv").as_posix()}`
- 归因旗标热图：`{flag_heatmap.as_posix()}`
- 主归因计数图：`{primary_fig.as_posix()}`

## 下一步

先复核被标记为 `sample_rule_or_raw_signal_review` 的事件。如果主要问题来自窗口太短或锚点偏晚，应回到阶段 2 修样本规则；如果复核后这些事件仍可信，再把 `vehicle_only_model_structure_gap` 作为下一版结构化车辆模型的目标。
"""

    user_summary = f"""# 阶段 3 用户查看版：复发坏样本失败来源归因

## 为什么做

上一轮已经把 12 个反复失败事件画出来了。这一步把这些图和表整理成可执行判断：哪些样本可能是锚点或窗口问题，哪些更像车辆-only 模型真的预测不了复杂响应。

## 检查了什么

- 标签窗口是否可能太短。
- 事件锚点前是否已经出现方向盘响应或车辆动力学变化。
- 原始车辆核心信号是否有足够有效点。
- RBF/KNN/template 等车辆-only 候选是否共同幅值不足、错侧或漏反向修正。

## 目前发现

Top 12 中有 {sample_review_n} 个事件优先归为“样本规则或原始信号需复核”，有 {structure_gap_n} 个事件优先归为“车辆-only 结构不足”。这意味着下一步不能直接进入生理或风格增量验证，要先把这些坏样本分清楚。

## 哪些结果可信

这一步没有训练新模型，没有使用生理、脑电、连续风格或驾驶员 ID，只读取已有车辆-only 误差表、样本清单和原始车辆片段。它适合决定下一步工程路线。

## 哪些还不能下结论

这些规则是自动初筛，不等于最终人工判定。特别是“锚点可能偏晚”和“窗口可能偏短”需要结合单事件曲线看。

## 下一阶段是否可以继续

可以继续阶段 3，但推荐先复核 `sample_rule_or_raw_signal_review` 的事件。若大部分确实是样本规则问题，应回到阶段 2 修 manifest；若不是，再进入响应分解、关键点残差或多假设车辆模型。

## 推荐优先查看

1. `{(TABLE_DIR / "bad_event_failure_attribution_table.csv").as_posix()}`
2. `{flag_heatmap.as_posix()}`
3. `{primary_fig.as_posix()}`
4. `{CURVE_ROOT / "figures" / "bad_event_curve_contact_sheet.png"}`
"""

    (REPORT_ROOT / "stage03_vehicle_instability_bad_event_failure_attribution_v0_1_cn.md").write_text(
        report, encoding="utf-8"
    )
    (REPORT_ROOT / "stage03_vehicle_instability_bad_event_failure_attribution_user_summary_cn.md").write_text(
        user_summary, encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    attribution, flag_counts, primary_counts = build_attribution_table()
    attribution.to_csv(TABLE_DIR / "bad_event_failure_attribution_table.csv", index=False, encoding="utf-8-sig")
    flag_counts.to_csv(TABLE_DIR / "bad_event_failure_flag_counts.csv", index=False, encoding="utf-8-sig")
    primary_counts.to_csv(TABLE_DIR / "bad_event_primary_attribution_counts.csv", index=False, encoding="utf-8-sig")
    flag_heatmap = plot_flag_heatmap(attribution)
    primary_fig = plot_primary_counts(primary_counts)
    write_reports(attribution, flag_counts, primary_counts, flag_heatmap, primary_fig)
    summary = {
        "n_events": int(len(attribution)),
        "sample_rule_or_raw_signal_review_n": int(
            (attribution["primary_attribution"] == "sample_rule_or_raw_signal_review").sum()
        ),
        "vehicle_only_model_structure_gap_n": int(
            (attribution["primary_attribution"] == "vehicle_only_model_structure_gap").sum()
        ),
        "flag_counts": flag_counts.to_dict(orient="records"),
        "server_used": False,
        "credential_file_read": False,
        "uses_subject_id_as_model_input": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "bad_event_failure_attribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
