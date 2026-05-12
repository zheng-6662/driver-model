# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
BASELINE_ROOT = ROOT / "03_baselines"
REPORT_ROOT = ROOT / "09_reports"
SAMPLE_ROOT = ROOT / "02_samples" / "vehicle_instability_highconf_v0_1"
SAMPLES_PATH = SAMPLE_ROOT / "tables" / "samples_master.csv"
REVIEW_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_robustness_bad_sample_review_v0_1"
REPRESENTATIVE_BAD_EVENTS_PATH = REVIEW_ROOT / "tables" / "robustness_representative_bad_events.csv"
OUTPUT_ROOT = BASELINE_ROOT / "stage03_vehicle_instability_bad_event_curve_review_v0_1"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
PER_EVENT_FIG_DIR = FIG_DIR / "per_event"
LOG_DIR = OUTPUT_ROOT / "logs"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402
import stage03_vehicle_instability_formal_baselines_v0_1 as formal_v01  # noqa: E402
import stage03_vehicle_instability_strong_vehicle_baselines_v0_1 as strong_v01  # noqa: E402


TOP_N = 12
RAW_PRE_S = -3.0
RAW_POST_S = 4.0
ROBUSTNESS_CONFIGS: dict[str, tuple[str, str]] = {
    "random_main": ("pre2_label2_old_main", "random_event_split"),
    "subject_main": ("pre2_label2_old_main", "subject_level_split"),
    "session_pre1": ("pre1_label2_event_trigger", "session_level_split"),
    "session_pre3": ("pre3_label3_response_coverage", "session_level_split"),
}
PLOT_MODELS = [
    "formal_ridge_vehicle_context_no_subject",
    "rbf_kernel_ridge_context_no_subject",
    "knn_template_context_no_subject",
    "direction_gated_knn_template_no_subject",
    "peak_scaled_template_context_no_subject",
]
DISPLAY_NAMES = {
    "formal_ridge_vehicle_context_no_subject": "formal ridge",
    "rbf_kernel_ridge_context_no_subject": "RBF KRR",
    "knn_template_context_no_subject": "KNN template",
    "direction_gated_knn_template_no_subject": "dir-gated KNN",
    "peak_scaled_template_context_no_subject": "peak-scaled template",
}
MODEL_COLORS = {
    "formal_ridge_vehicle_context_no_subject": "#7f7f7f",
    "rbf_kernel_ridge_context_no_subject": "#1f77b4",
    "knn_template_context_no_subject": "#ff7f0e",
    "direction_gated_knn_template_no_subject": "#2ca02c",
    "peak_scaled_template_context_no_subject": "#d62728",
}
RAW_COLS = [
    "StorageTime",
    "zx|SteeringWheel",
    "zx|ay",
    "zx|roll",
    "zx|vyaw",
    "zx|v_km/h",
    "zx|vx",
    "zx1|v_km/h",
    "zx1|lanecurvatureXY",
    "zx1|lateraldistance",
    "zx|lateraldistance",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, PER_EVENT_FIG_DIR, LOG_DIR, REPORT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def sanitize_filename(text: str, max_len: int = 120) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return safe[:max_len] or "event"


def set_strong_context(window_id: str, split_strategy: str) -> None:
    strong_v01.WINDOW_ID = window_id
    strong_v01.SPLIT_STRATEGY = split_strategy


def split_indices(meta: pd.DataFrame, split_strategy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = meta[split_strategy].astype(str).to_numpy()
    return np.where(split == "train")[0], np.where(split == "val")[0], np.where(split == "test")[0]


def build_predictions_for_config(
    samples: pd.DataFrame, config_id: str
) -> dict[str, Any]:
    window_id, split_strategy = ROBUSTNESS_CONFIGS[config_id]
    set_strong_context(window_id, split_strategy)
    y, y_mask, input_values, input_time, label_time, meta = formal_v01.load_window(window_id, samples)
    train_idx, val_idx, test_idx = split_indices(meta, split_strategy)
    if not (len(train_idx) and len(val_idx) and len(test_idx)):
        raise RuntimeError(f"{config_id}: incomplete split for {split_strategy}")

    formal_preds, _ = formal_v01.build_predictions(
        y, y_mask, input_values, input_time, label_time, meta, split_strategy
    )
    predictions: dict[str, np.ndarray] = {
        "formal_ridge_vehicle_context_no_subject": formal_preds["ridge_vehicle_context_no_subject"],
    }
    x_rich, _ = strong_v01.build_rich_vehicle_features(
        input_values, input_time, meta, train_idx, include_context=True
    )
    x_scaled, _ = strong_v01.standardize_train_only(x_rich, train_idx)
    x_dist, _ = strong_v01.make_distance_features(x_scaled, train_idx, n_components=96)
    peaks = strong_v01.peak_arrays(y, y_mask, label_time)

    fitters = [
        (
            "rbf_kernel_ridge_context_no_subject",
            lambda: strong_v01.fit_rbf_kernel_ridge_direct(x_dist, y, train_idx, val_idx, y_mask),
        ),
        (
            "knn_template_context_no_subject",
            lambda: strong_v01.fit_knn_template(
                "knn_template_context_no_subject", x_dist, y, y_mask, train_idx, val_idx
            ),
        ),
        (
            "direction_gated_knn_template_no_subject",
            lambda: strong_v01.fit_direction_gated_knn_template(x_dist, y, y_mask, peaks, train_idx, val_idx),
        ),
        (
            "peak_scaled_template_context_no_subject",
            lambda: strong_v01.fit_peak_scaled_template(x_dist, y, y_mask, peaks, train_idx, val_idx),
        ),
    ]
    model_info: list[dict[str, Any]] = []
    for model_name, fit in fitters:
        pred, info = fit()
        predictions[model_name] = pred
        info["model_name"] = model_name
        model_info.append(info)

    return {
        "config_id": config_id,
        "window_id": window_id,
        "split_strategy": split_strategy,
        "y": y,
        "y_mask": y_mask,
        "input_values": input_values,
        "input_time": input_time,
        "label_time": label_time,
        "meta": meta,
        "predictions": predictions,
        "model_info": model_info,
    }


def to_seconds(storage_time: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(storage_time, errors="coerce")
    out = np.full(len(parsed), np.nan, dtype=np.float64)
    valid = parsed.notna().to_numpy()
    if valid.any():
        out[valid] = parsed[valid].astype("int64").to_numpy(dtype=np.float64) / 1e9
    return out


def read_raw_vehicle_window(row: pd.Series) -> pd.DataFrame:
    raw_path = Path(str(row.get("vehicle_raw_absolute_path") or row.get("vehicle_absolute_path")))
    if not raw_path.exists():
        return pd.DataFrame()
    header = pd.read_csv(raw_path, nrows=0)
    usecols = [col for col in RAW_COLS if col in header.columns]
    if "StorageTime" not in usecols:
        return pd.DataFrame()
    raw = pd.read_csv(raw_path, usecols=usecols)
    t_abs = to_seconds(raw["StorageTime"])
    anchor_abs = safe_float(row.get("anchor_time_abs_storage_s"))
    if not math.isfinite(anchor_abs):
        return pd.DataFrame()
    raw = raw.copy()
    raw["t_rel_s"] = t_abs - anchor_abs
    label_end = safe_float(row.get("label_end_rel_s"), RAW_POST_S)
    event_end = safe_float(row.get("event_end_rel_s")) - safe_float(row.get("anchor_time_rel_s"))
    post = max(RAW_POST_S, label_end, event_end if math.isfinite(event_end) else RAW_POST_S)
    raw = raw[(raw["t_rel_s"] >= RAW_PRE_S) & (raw["t_rel_s"] <= post + 0.2)].copy()
    for col in raw.columns:
        if col not in {"StorageTime"}:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw


def zscore_series(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() < 3:
        return np.full_like(arr, np.nan, dtype=np.float64)
    center = np.nanmedian(arr)
    scale = np.nanpercentile(arr, 90) - np.nanpercentile(arr, 10)
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = np.nanstd(arr)
    if not math.isfinite(scale) or scale <= 1e-9:
        return np.full_like(arr, np.nan, dtype=np.float64)
    return (arr - center) / scale


def sample_model_metrics(
    y: np.ndarray,
    y_mask: np.ndarray,
    pred: np.ndarray,
    label_time: np.ndarray,
    idx: int,
    model_name: str,
) -> dict[str, Any]:
    valid = y_mask[idx] & np.isfinite(y[idx]) & np.isfinite(pred[idx])
    if valid.sum() == 0:
        return {
            "model_name": model_name,
            "sample_rmse": float("nan"),
            "gt_peak_abs": float("nan"),
            "pred_peak_abs": float("nan"),
            "wrong_side": None,
            "severe_amp_under": None,
            "reversal_count_exact": None,
        }
    gt = np.where(valid, y[idx], np.nan)
    pr = np.where(valid, pred[idx], np.nan)
    gt_peak = eval_utils.peak_stats(gt, label_time)
    pr_peak = eval_utils.peak_stats(pr, label_time)
    gt_rev = eval_utils.reversal_count(gt)
    pr_rev = eval_utils.reversal_count(pr)
    return {
        "model_name": model_name,
        "sample_rmse": eval_utils.rmse(gt[None, :], pr[None, :], valid[None, :]),
        "gt_peak_abs": gt_peak["peak_abs"],
        "pred_peak_abs": pr_peak["peak_abs"],
        "pred_over_gt_peak_ratio": pr_peak["peak_abs"] / max(gt_peak["peak_abs"], 1e-6),
        "wrong_side": int(gt_peak["peak_direction"] != pr_peak["peak_direction"]),
        "severe_amp_under": int(pr_peak["peak_abs"] < 0.5 * max(gt_peak["peak_abs"], 1e-6)),
        "gt_reversal_count": gt_rev,
        "pred_reversal_count": pr_rev,
        "reversal_count_exact": int(gt_rev == pr_rev),
    }


def add_window_marks(ax: plt.Axes, row: pd.Series) -> None:
    input_start = safe_float(row.get("input_start_rel_s"), -2.0)
    input_end = safe_float(row.get("input_end_rel_s"), 0.0)
    label_start = safe_float(row.get("label_start_rel_s"), 0.0)
    label_end = safe_float(row.get("label_end_rel_s"), 2.0)
    anchor_rel = safe_float(row.get("anchor_time_rel_s"), 0.0)
    event_start = safe_float(row.get("event_start_rel_s")) - anchor_rel
    event_end = safe_float(row.get("event_end_rel_s")) - anchor_rel
    ax.axvspan(input_start, input_end, color="#d9e8ff", alpha=0.45, label="input window")
    ax.axvspan(label_start, label_end, color="#e2f2df", alpha=0.45, label="label window")
    ax.axvline(0.0, color="#0057d9", linewidth=1.4, linestyle="-", label="anchor")
    if math.isfinite(event_start):
        ax.axvline(event_start, color="#ff9900", linewidth=1.0, linestyle="--", label="event start")
    if math.isfinite(event_end):
        ax.axvline(event_end, color="#ff9900", linewidth=1.0, linestyle=":", label="event end")


def plot_scaled_raw_signal(ax: plt.Axes, raw: pd.DataFrame, col: str, label: str) -> None:
    vals = zscore_series(raw[col])
    finite = np.isfinite(raw["t_rel_s"].to_numpy(dtype=np.float64)) & np.isfinite(vals)
    if finite.sum() > 2:
        ax.plot(raw["t_rel_s"].to_numpy(dtype=np.float64)[finite], vals[finite], linewidth=1.1, label=label)


def plot_event(
    row: pd.Series,
    config_data: dict[str, Any],
    idx: int,
    rank: int,
    model_metrics: list[dict[str, Any]],
) -> tuple[Path, Path]:
    y = config_data["y"]
    y_mask = config_data["y_mask"]
    label_time = config_data["label_time"]
    predictions = config_data["predictions"]
    raw = read_raw_vehicle_window(row)
    safe_event = sanitize_filename(str(row["event_uid"]))
    png_path = PER_EVENT_FIG_DIR / f"rank{rank:02d}_{config_data['config_id']}_{safe_event}.png"
    pdf_path = PER_EVENT_FIG_DIR / f"rank{rank:02d}_{config_data['config_id']}_{safe_event}.pdf"

    fig = plt.figure(figsize=(13.5, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 1.3, 1.2, 0.8])
    ax_steer = fig.add_subplot(gs[0])
    ax_dyn = fig.add_subplot(gs[1], sharex=ax_steer)
    ax_context = fig.add_subplot(gs[2], sharex=ax_steer)
    ax_text = fig.add_subplot(gs[3])

    add_window_marks(ax_steer, row)
    if not raw.empty and "zx|SteeringWheel" in raw.columns:
        anchor_steer = safe_float(row.get("anchor_steer"))
        raw_steer = raw["zx|SteeringWheel"] - anchor_steer if math.isfinite(anchor_steer) else raw["zx|SteeringWheel"]
        raw_t = raw["t_rel_s"].to_numpy(dtype=np.float64)
        raw_y = pd.to_numeric(raw_steer, errors="coerce").to_numpy(dtype=np.float64)
        raw_valid = np.isfinite(raw_t) & np.isfinite(raw_y)
        if raw_valid.sum() > 2:
            ax_steer.plot(
                raw_t[raw_valid],
                raw_y[raw_valid],
                color="#b0b0b0",
                linewidth=1.0,
                alpha=0.85,
                label="raw steer delta",
            )

    valid = y_mask[idx] & np.isfinite(y[idx])
    ax_steer.plot(label_time[valid], y[idx][valid], color="black", linewidth=2.2, label="GT steer delta")
    for model_name in PLOT_MODELS:
        if model_name not in predictions:
            continue
        pred = predictions[model_name][idx]
        ax_steer.plot(
            label_time[valid],
            pred[valid],
            linewidth=1.3,
            color=MODEL_COLORS.get(model_name),
            alpha=0.95,
            label=DISPLAY_NAMES.get(model_name, model_name),
        )
    ax_steer.set_ylabel("Steer delta")
    ax_steer.grid(alpha=0.25)
    ax_steer.set_title(
        f"Rank {rank} recurrent bad event | {config_data['config_id']} | {row['subject']} {row['session_stamp']}"
    )
    ax_steer.legend(loc="upper left", fontsize=8, ncols=3)

    add_window_marks(ax_dyn, row)
    dyn_cols = [
        ("zx|ay", "ay"),
        ("zx|roll", "roll"),
        ("zx|vyaw", "yaw rate"),
    ]
    if raw.empty:
        ax_dyn.text(0.5, 0.5, "raw vehicle window unavailable", ha="center", va="center")
    else:
        for col, label in dyn_cols:
            if col in raw.columns:
                plot_scaled_raw_signal(ax_dyn, raw, col, label)
    ax_dyn.set_ylabel("Vehicle dynamics\nrobust-scaled")
    ax_dyn.grid(alpha=0.25)
    ax_dyn.legend(loc="upper left", fontsize=8, ncols=3)

    add_window_marks(ax_context, row)
    context_cols = [
        ("zx1|v_km/h", "lead speed"),
        ("zx|v_km/h", "ego speed"),
        ("zx1|lanecurvatureXY", "curvature"),
        ("zx1|lateraldistance", "lateral dist"),
    ]
    if not raw.empty:
        for col, label in context_cols:
            if col in raw.columns:
                plot_scaled_raw_signal(ax_context, raw, col, label)
    ax_context.set_ylabel("Road/context\nrobust-scaled")
    ax_context.set_xlabel("Time relative to anchor (s)")
    ax_context.grid(alpha=0.25)
    ax_context.legend(loc="upper left", fontsize=8, ncols=4)

    best = sorted(model_metrics, key=lambda r: safe_float(r["sample_rmse"], float("inf")))[:2]
    worst = sorted(model_metrics, key=lambda r: safe_float(r["sample_rmse"], -float("inf")), reverse=True)[:2]
    lines = [
        f"event_uid: {row['event_uid']}",
        f"sample_id: {row['sample_id']}",
        f"road={row.get('road_type_anchor', 'NA')} old_phase={row.get('old_v400_phase_mode', 'NA')} "
        f"level={row.get('event_level', 'NA')} morphology={row.get('eval_label_morphology', 'NA')}",
        "best RMSE: "
        + "; ".join(
            f"{DISPLAY_NAMES.get(r['model_name'], r['model_name'])}={safe_float(r['sample_rmse']):.3f}"
            for r in best
        ),
        "worst RMSE: "
        + "; ".join(
            f"{DISPLAY_NAMES.get(r['model_name'], r['model_name'])}={safe_float(r['sample_rmse']):.3f}"
            for r in worst
        ),
    ]
    ax_text.axis("off")
    ax_text.text(0.01, 0.95, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    label_end = safe_float(row.get("label_end_rel_s"), 2.0)
    focus_xmax = max(label_end, 3.0) + 0.35
    for ax in [ax_steer, ax_dyn, ax_context]:
        ax.set_xlim(RAW_PRE_S, focus_xmax)

    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def make_contact_sheet(index_df: pd.DataFrame) -> tuple[Path, Path]:
    rows = index_df.head(TOP_N).copy()
    n_cols = 3
    n_rows = int(math.ceil(len(rows) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, max(5, 4.6 * n_rows)), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    for ax, (_, row) in zip(axes_arr, rows.iterrows()):
        image_path = Path(str(row["figure_png"]))
        if image_path.exists():
            ax.imshow(plt.imread(image_path))
        ax.set_title(
            f"#{int(row['recurrence_rank'])} {row['subject']} {row['config_id']}\n"
            f"RMSE worst={safe_float(row['worst_sample_rmse']):.3f}",
            fontsize=9,
        )
        ax.axis("off")
    png = FIG_DIR / "bad_event_curve_contact_sheet.png"
    pdf = FIG_DIR / "bad_event_curve_contact_sheet.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def table_to_md(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    sub = df[columns].head(max_rows).copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in sub.to_numpy()]
    return "\n".join([header, sep] + body)


def write_reports(index_df: pd.DataFrame, error_df: pd.DataFrame, contact_png: Path) -> None:
    top = index_df.iloc[0]
    severe_rate = float(error_df["severe_amp_under"].dropna().mean()) if len(error_df) else float("nan")
    wrong_rate = float(error_df["wrong_side"].dropna().mean()) if len(error_df) else float("nan")
    reversal_exact_rate = float(error_df["reversal_count_exact"].dropna().mean()) if len(error_df) else float("nan")
    report = f"""# 阶段 3：复发坏样本详细曲线复盘 v0.1

生成时间：2026-05-12

## 目的

上一轮已经知道 RBF/KNN/template 在若干配置下会反复失败。本轮不训练新路线，不引入生理、脑电或连续风格，只把复发最高的坏事件画成可复核曲线，检查失败更像锚点/窗口问题、原始车辆局部异常，还是车辆-only 模型表达不足。

## 输入

- 代表坏样本表：`{REPRESENTATIVE_BAD_EVENTS_PATH.as_posix()}`
- 正式样本清单：`{SAMPLES_PATH.as_posix()}`
- 处理后车辆窗口：`{formal_v01.ARRAY_DIR.as_posix()}`
- 原始车辆 CSV：只按 `samples_master.csv` 中每个事件的 `vehicle_raw_absolute_path` 读取片段；未修改原始文件。

## 方法

- 选取复发坏样本 Top {TOP_N}。
- 对每个事件使用其 `worst_config` 对应的窗口和 split。
- 复用已提交的 formal ridge、RBF KRR、KNN template、direction-gated KNN、peak-scaled template 逻辑，仅为绘图重建预测曲线。
- 图中同时画事件锚点、输入窗口、标签窗口、事件结束线、原始方向盘相对锚点变化、GT 方向盘增量、候选模型预测、原始车辆动力学与道路上下文波形。

## 主要发现

- 复发最高事件：`{top['event_uid']}`，subject=`{top['subject']}`，config=`{top['config_id']}`。
- Top {TOP_N} 事件 * 5 个车辆-only 候选模型的逐样本曲线中，严重幅值不足率={severe_rate:.3f}，错侧率={wrong_rate:.3f}，反向修正计数完全匹配率={reversal_exact_rate:.3f}。
- 这些图仍不能单独证明“生理有效”或“Transformer 更好”；它们的用途是把车辆-only 当前失败类型具体化，为下一版结构化车辆模型提供目标。

## 图表索引 Top12

{table_to_md(index_df, ["recurrence_rank", "event_uid", "subject", "config_id", "worst_sample_rmse", "figure_png"], max_rows=12)}

## 产物

- 图索引：`{(TABLE_DIR / "bad_event_curve_figure_index.csv").as_posix()}`
- 模型逐事件误差表：`{(TABLE_DIR / "bad_event_curve_model_error_table.csv").as_posix()}`
- 总览拼图：`{contact_png.as_posix()}`
- 单事件图目录：`{PER_EVENT_FIG_DIR.as_posix()}`

## 下一步

优先人工抽看 Top 12 曲线中是否存在明显锚点偏早/偏晚、标签窗口没有覆盖完整响应、原始 `ay/roll/vyaw` 局部异常或道路上下文突变。如果这些问题不能解释大部分失败，再进入结构化车辆响应模型：方向/幅值/峰值时间/反向修正/多段修正分解，或关键点 + 残差轨迹模型。
"""

    user_summary = f"""# 阶段 3 用户查看版：复发坏样本详细曲线复盘

## 为什么做

前面已经看到 RBF/KNN/template 的平均误差有改善，但一些事件在很多配置下都会失败。这个阶段把这些反复失败的事件画成曲线，方便直接看问题是不是来自事件锚点、窗口、原始车辆信号，还是车辆-only 模型确实表达不了复杂响应。

## 检查了什么

- 复发坏样本 Top {TOP_N}。
- 每个事件的输入窗口、标签窗口和事件锚点。
- 原始车辆方向盘、横向加速度、横滚、横摆角速度、速度、曲率、横向位置等波形。
- GT 方向盘响应与 RBF/KNN/template 等车辆-only 候选预测。

## 目前发现

复发最高的事件仍是 `{top['event_uid']}`。在 Top {TOP_N} 事件的 5 个候选预测里，严重幅值不足率={severe_rate:.3f}，错侧率={wrong_rate:.3f}，反向修正计数完全匹配率={reversal_exact_rate:.3f}。这说明平均 RMSE 变低以后，复杂物理响应仍然没有被稳定解决。

## 哪些结果可信

这一步只做复盘和绘图，没有引入生理、脑电、连续风格或驾驶员 ID。图里的原始车辆波形来自 `samples_master.csv` 指向的原始车辆 CSV，只读取不修改。

## 哪些结果还不能下结论

现在还不能说失败一定是模型结构造成的，也不能说生理数据会解决这些失败。必须先看曲线里是否有锚点偏差、窗口覆盖不足或原始数据异常。

## 下一阶段是否可以继续

可以继续阶段 3，但不是进入生理或风格；下一步应该先根据这些曲线决定结构化车辆模型怎么做。

## 推荐优先查看

1. `{contact_png.as_posix()}`
2. `{(TABLE_DIR / "bad_event_curve_figure_index.csv").as_posix()}`
3. `{(TABLE_DIR / "bad_event_curve_model_error_table.csv").as_posix()}`
4. `{PER_EVENT_FIG_DIR.as_posix()}`
"""

    (REPORT_ROOT / "stage03_vehicle_instability_bad_event_curve_review_v0_1_cn.md").write_text(
        report, encoding="utf-8"
    )
    (REPORT_ROOT / "stage03_vehicle_instability_bad_event_curve_review_user_summary_cn.md").write_text(
        user_summary, encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    samples = pd.read_csv(SAMPLES_PATH)
    reps = pd.read_csv(REPRESENTATIVE_BAD_EVENTS_PATH).head(TOP_N).copy()

    configs_needed = sorted({str(v) for v in reps["worst_config"].dropna().unique()})
    prediction_cache = {
        config_id: build_predictions_for_config(samples, config_id)
        for config_id in configs_needed
        if config_id in ROBUSTNESS_CONFIGS
    }

    figure_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for _, rep in reps.iterrows():
        rank = int(rep["recurrence_rank"])
        config_id = str(rep["worst_config"])
        if config_id not in prediction_cache:
            continue
        data = prediction_cache[config_id]
        meta = data["meta"]
        sample_id = str(rep["worst_sample_id"])
        match = meta.index[meta["sample_id"].astype(str) == sample_id].to_numpy()
        if match.size == 0:
            match = meta.index[meta["event_uid"].astype(str) == str(rep["event_uid"])].to_numpy()
        if match.size == 0:
            continue
        idx = int(match[0])
        row = meta.iloc[idx].copy()
        metrics_for_event = []
        for model_name in PLOT_MODELS:
            metric_row = sample_model_metrics(
                data["y"],
                data["y_mask"],
                data["predictions"][model_name],
                data["label_time"],
                idx,
                model_name,
            )
            metric_row.update(
                {
                    "recurrence_rank": rank,
                    "event_uid": row["event_uid"],
                    "sample_id": row["sample_id"],
                    "subject": row["subject"],
                    "session_stamp": row["session_stamp"],
                    "config_id": config_id,
                    "window_config_id": data["window_id"],
                    "split_strategy": data["split_strategy"],
                }
            )
            metrics_for_event.append(metric_row)
            error_rows.append(metric_row)
        png_path, pdf_path = plot_event(row, data, idx, rank, metrics_for_event)
        figure_rows.append(
            {
                "recurrence_rank": rank,
                "event_uid": row["event_uid"],
                "sample_id": row["sample_id"],
                "subject": row["subject"],
                "session_stamp": row["session_stamp"],
                "config_id": config_id,
                "window_config_id": data["window_id"],
                "split_strategy": data["split_strategy"],
                "worst_model": rep["worst_model"],
                "worst_sample_rmse": rep["worst_sample_rmse"],
                "raw_vehicle_path": row.get("vehicle_raw_absolute_path", ""),
                "figure_png": png_path.as_posix(),
                "figure_pdf": pdf_path.as_posix(),
            }
        )

    index_df = pd.DataFrame(figure_rows).sort_values("recurrence_rank").reset_index(drop=True)
    error_df = pd.DataFrame(error_rows).sort_values(["recurrence_rank", "model_name"]).reset_index(drop=True)
    index_df.to_csv(TABLE_DIR / "bad_event_curve_figure_index.csv", index=False, encoding="utf-8-sig")
    error_df.to_csv(TABLE_DIR / "bad_event_curve_model_error_table.csv", index=False, encoding="utf-8-sig")
    contact_png, contact_pdf = make_contact_sheet(index_df)
    write_reports(index_df, error_df, contact_png)

    summary = {
        "top_n_requested": TOP_N,
        "n_events_plotted": int(len(index_df)),
        "n_model_event_rows": int(len(error_df)),
        "configs_rebuilt_for_plotting": configs_needed,
        "contact_sheet_png": contact_png.as_posix(),
        "contact_sheet_pdf": contact_pdf.as_posix(),
        "figure_index": (TABLE_DIR / "bad_event_curve_figure_index.csv").as_posix(),
        "server_used": False,
        "credential_file_read": False,
        "uses_subject_id_as_model_input": False,
        "uses_physio": False,
        "uses_eeg": False,
        "uses_continuous_style": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "bad_event_curve_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
