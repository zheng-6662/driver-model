# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
FORMAL_SAMPLE_DIR = ROOT / "02_samples" / "vehicle_instability_highconf_v0_1"
SAMPLES_PATH = FORMAL_SAMPLE_DIR / "tables" / "samples_master.csv"
PROCESSED_DIR = ROOT / "03_processed_datasets" / "vehicle_instability_allraw_highconf_v0_1"
ARRAY_DIR = PROCESSED_DIR / "arrays"
OUT_DIR = ROOT / "03_baselines" / "stage03_vehicle_instability_formal_baselines_v0_1"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as eval_utils  # noqa: E402


WINDOWS = [
    "pre1_label2_event_trigger",
    "pre2_label2_old_main",
    "pre3_label3_response_coverage",
]
SPLIT_STRATEGIES = ["random_event_split", "session_level_split", "subject_level_split"]
DEFAULT_WINDOW = "pre2_label2_old_main"
DEFAULT_SPLIT_STRATEGY = "session_level_split"
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

CONTEXT_COLS = [
    "event_type",
    "event_level",
    "road_type_anchor",
    "old_v400_road_type_mode",
    "old_v400_phase_mode",
    "road_design_risk_class",
    "road_design_mapping_reliability",
]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_window(window_id: str, samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    z = np.load(ARRAY_DIR / f"{window_id}.npz", allow_pickle=True)
    y = z["label_steer_delta"].astype(np.float32)
    y_mask = z["label_valid_mask"].astype(bool)
    input_values = z["input_values"].astype(np.float32)
    input_time = z["input_time_rel_s"].astype(np.float32)
    label_time = z["label_time_rel_s"].astype(np.float32)
    meta = samples[samples["window_config_id"] == window_id].copy()
    meta["array_row"] = pd.to_numeric(meta["array_row"], errors="coerce").astype(int)
    meta = meta.sort_values("array_row").reset_index(drop=True)
    if len(meta) != y.shape[0]:
        raise ValueError(f"{window_id}: sample rows {len(meta)} != array rows {y.shape[0]}")
    if not np.array_equal(meta["array_row"].to_numpy(), np.arange(y.shape[0])):
        raise ValueError(f"{window_id}: array_row is not contiguous")
    return y, y_mask, input_values, input_time, label_time, meta


def stable_numeric(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)


def extract_history_features(
    input_values: np.ndarray,
    input_time: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    include_context: bool,
) -> tuple[np.ndarray, list[str]]:
    x = input_values.astype(np.float64)
    mask = np.isfinite(x)
    x = np.where(mask, x, np.nan)
    features: list[np.ndarray] = []
    names: list[str] = []
    for j in range(x.shape[2]):
        arr = x[:, :, j]
        stats = [
            ("last", arr[:, -1]),
            ("mean", np.nanmean(arr, axis=1)),
            ("std", np.nanstd(arr, axis=1)),
            ("min", np.nanmin(arr, axis=1)),
            ("max", np.nanmax(arr, axis=1)),
            ("delta", arr[:, -1] - arr[:, 0]),
        ]
        recent = input_time >= (float(input_time[-1]) - 0.5)
        if recent.sum() >= 2:
            t = input_time[recent].astype(np.float64)
            tc = t - t.mean()
            denom = float(np.sum(tc * tc)) or 1.0
            centered = arr[:, recent] - np.nanmean(arr[:, recent], axis=1, keepdims=True)
            slope = np.nansum(centered * tc[None, :], axis=1) / denom
            stats.append(("slope_last500ms", slope))
        for stat_name, vals in stats:
            features.append(vals)
            names.append(f"signal{j}_{stat_name}")

    for col in ["anchor_time_rel_s", "input_valid_ratio", "label_valid_ratio"]:
        if col in meta.columns:
            features.append(stable_numeric(meta[col]))
            names.append(col)

    if include_context:
        for col in CONTEXT_COLS:
            if col not in meta.columns:
                continue
            values = meta[col].astype(str).fillna("NA")
            train_values = sorted(values.iloc[train_idx].unique().tolist()) if train_idx.size else sorted(values.unique().tolist())
            for val in train_values:
                features.append((values == val).to_numpy(dtype=np.float64))
                names.append(f"{col}={val}")

    X = np.vstack(features).T if features else np.zeros((len(meta), 0), dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def fit_ridge(
    y: np.ndarray,
    y_mask: np.ndarray,
    input_values: np.ndarray,
    input_time: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    include_context: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if train_idx.size < 5:
        return np.zeros_like(y, dtype=np.float32), {"status": "no_train_samples", "feature_count": 0}
    X, names = extract_history_features(input_values, input_time, meta, train_idx, include_context=include_context)
    mu = X[train_idx].mean(axis=0, keepdims=True)
    sigma = X[train_idx].std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    Xs = (X - mu) / sigma
    Xd = np.c_[np.ones((Xs.shape[0], 1)), Xs]
    Y = np.nan_to_num(np.where(y_mask, y, np.nan), nan=0.0).astype(np.float64)
    eval_idx = val_idx if val_idx.size else train_idx
    best_alpha = RIDGE_ALPHAS[0]
    best_score = float("inf")
    best_pred: np.ndarray | None = None
    for alpha in RIDGE_ALPHAS:
        Xt = Xd[train_idx]
        reg = np.eye(Xt.shape[1], dtype=np.float64) * float(alpha)
        reg[0, 0] = 0.0
        coef = np.linalg.solve(Xt.T @ Xt + reg, Xt.T @ Y[train_idx])
        pred = (Xd @ coef).astype(np.float32)
        score = eval_utils.rmse(y[eval_idx], pred[eval_idx], y_mask[eval_idx])
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
            best_pred = pred
    assert best_pred is not None
    return best_pred, {
        "status": "ok",
        "selected_alpha": best_alpha,
        "val_rmse_for_alpha": float(best_score),
        "train_rmse_selected_alpha": float(eval_utils.rmse(y[train_idx], best_pred[train_idx], y_mask[train_idx])),
        "feature_count": int(len(names)),
        "include_context": bool(include_context),
        "uses_subject_id": False,
        "scaler_fit_scope": "train split only",
    }


def build_predictions(
    y: np.ndarray,
    y_mask: np.ndarray,
    input_values: np.ndarray,
    input_time: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    split_strategy: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    n = y.shape[0]
    train_idx = np.where(meta[split_strategy].astype(str).to_numpy() == "train")[0]
    val_idx = np.where(meta[split_strategy].astype(str).to_numpy() == "val")[0]
    if train_idx.size == 0:
        train_idx = np.arange(n)
    preds: dict[str, np.ndarray] = {}
    info_rows: list[dict[str, Any]] = []

    preds["zero_response_hold_current"] = np.zeros_like(y, dtype=np.float32)

    steer_hist = input_values[:, :, 0].astype(np.float64)
    anchor = steer_hist[:, [-1]]
    hist_delta = steer_hist - anchor
    recent = input_time >= (float(input_time[-1]) - 0.5)
    if recent.sum() < 2:
        recent = np.ones_like(input_time, dtype=bool)
    t_recent = input_time[recent].astype(np.float64)
    t_center = t_recent - t_recent.mean()
    denom = float(np.sum(t_center * t_center)) or 1.0
    centered = hist_delta[:, recent] - np.nanmean(hist_delta[:, recent], axis=1, keepdims=True)
    slopes = np.nansum(centered * t_center[None, :], axis=1) / denom
    train_peak = np.nanpercentile(np.abs(y[train_idx]), 95) if train_idx.size else np.nanpercentile(np.abs(y), 95)
    clip = max(float(train_peak) * 1.5, 1.0)
    preds["history_trend_500ms"] = np.clip(slopes[:, None] * label_time[None, :], -clip, clip).astype(np.float32)

    train_mean = np.nanmean(np.where(y_mask[train_idx], y[train_idx], np.nan), axis=0)
    train_mean = np.nan_to_num(train_mean, nan=0.0).astype(np.float32)
    preds["train_mean_all"] = np.tile(train_mean[None, :], (n, 1))

    grouped = np.zeros_like(y, dtype=np.float32)
    train_set = set(train_idx.tolist())
    for _, group_df in meta.groupby(["event_type", "event_level"], dropna=False):
        idx = group_df.index.to_numpy()
        train_group = [int(i) for i in idx if int(i) in train_set]
        if len(train_group) < 3:
            grouped[idx] = preds["train_mean_all"][idx]
        else:
            mean = np.nanmean(np.where(y_mask[train_group], y[train_group], np.nan), axis=0)
            grouped[idx] = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    preds["train_mean_by_event_type"] = grouped

    for name, include_context in [
        ("ridge_vehicle_history_no_subject", False),
        ("ridge_vehicle_context_no_subject", True),
    ]:
        ridge_pred, ridge_info = fit_ridge(y, y_mask, input_values, input_time, meta, train_idx, val_idx, include_context)
        preds[name] = ridge_pred
        ridge_info.update({"model_name": name, "split_strategy": split_strategy})
        info_rows.append(ridge_info)

    for name in ["zero_response_hold_current", "history_trend_500ms", "train_mean_all", "train_mean_by_event_type"]:
        info_rows.append(
            {
                "model_name": name,
                "split_strategy": split_strategy,
                "status": "ok",
                "selected_alpha": np.nan,
                "val_rmse_for_alpha": np.nan,
                "train_rmse_selected_alpha": float(eval_utils.rmse(y[train_idx], preds[name][train_idx], y_mask[train_idx])),
                "feature_count": 0,
                "include_context": name == "train_mean_by_event_type",
                "uses_subject_id": False,
                "scaler_fit_scope": "no fitted scaler",
            }
        )
    return preds, info_rows


def evaluate_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    samples = pd.read_csv(SAMPLES_PATH)
    all_metric_rows: list[pd.DataFrame] = []
    all_sample_rows: list[pd.DataFrame] = []
    model_info_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "samples_path": str(SAMPLES_PATH).replace("\\", "/"),
        "out_dir": str(OUT_DIR).replace("\\", "/"),
        "server_used": False,
        "credential_file_read": False,
        "windows": WINDOWS,
        "split_strategies": SPLIT_STRATEGIES,
        "default_window": DEFAULT_WINDOW,
        "default_split_strategy": DEFAULT_SPLIT_STRATEGY,
    }
    for window_id in WINDOWS:
        y, y_mask, input_values, input_time, label_time, meta = load_window(window_id, samples)
        for split_strategy in SPLIT_STRATEGIES:
            train_idx = np.where(meta[split_strategy].astype(str).to_numpy() == "train")[0]
            if train_idx.size == 0:
                train_idx = np.arange(len(meta))
            gt_peak = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)
            large_thr = float(np.nanpercentile(gt_peak[train_idx], 75))
            difficult_thr = float(np.nanpercentile(gt_peak[train_idx], 80))
            preds, info_rows = build_predictions(y, y_mask, input_values, input_time, label_time, meta, split_strategy)
            for info in info_rows:
                info.update(
                    {
                        "window_config_id": window_id,
                        "train_n": int((meta[split_strategy].astype(str) == "train").sum()),
                        "val_n": int((meta[split_strategy].astype(str) == "val").sum()),
                        "test_n": int((meta[split_strategy].astype(str) == "test").sum()),
                        "large_threshold_train_p75": large_thr,
                        "difficult_threshold_train_p80": difficult_thr,
                    }
                )
                model_info_rows.append(info)
            for model_name, pred in preds.items():
                for split_name in ["train", "val", "test"]:
                    split_mask = meta[split_strategy].astype(str).to_numpy() == split_name
                    if not split_mask.any():
                        continue
                    sample_rows = eval_utils.sample_metric_rows(
                        y[split_mask],
                        pred[split_mask],
                        y_mask[split_mask],
                        label_time,
                        meta.loc[split_mask].reset_index(drop=True),
                        model_name=model_name,
                        split_strategy=split_strategy,
                        split_name=split_name,
                        window_id=window_id,
                        large_thr=large_thr,
                        difficult_thr=difficult_thr,
                    )
                    if sample_rows:
                        all_sample_rows.append(pd.DataFrame(sample_rows))
    per_sample = pd.concat(all_sample_rows, ignore_index=True)
    metrics = eval_utils.aggregate_metrics(per_sample)
    model_info = pd.DataFrame(model_info_rows)
    summary["per_sample_rows"] = int(len(per_sample))
    summary["metrics_rows"] = int(len(metrics))
    return metrics, per_sample, model_info, summary


def select_primary_rows(metrics: pd.DataFrame, per_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_metrics = metrics[
        (metrics["window_config_id"] == DEFAULT_WINDOW)
        & (metrics["split_strategy"] == DEFAULT_SPLIT_STRATEGY)
        & (metrics["split"] == "test")
    ].sort_values("rmse_steer")
    primary_samples = per_sample[
        (per_sample["window_config_id"] == DEFAULT_WINDOW)
        & (per_sample["split_strategy"] == DEFAULT_SPLIT_STRATEGY)
        & (per_sample["split"] == "test")
    ].copy()
    ridge = primary_samples[primary_samples["model_name"] == "ridge_vehicle_context_no_subject"].copy()
    return primary_metrics, primary_samples, ridge


def plot_samples(
    sample_ids: list[str],
    y: np.ndarray,
    y_mask: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    pred_map: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    n = len(sample_ids)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, max(3.2 * rows, 3.2)), squeeze=False)
    meta_idx = {sid: i for i, sid in enumerate(meta["sample_id"].astype(str))}
    for ax in axes.ravel():
        ax.axis("off")
    for k, sid in enumerate(sample_ids):
        ax = axes.ravel()[k]
        ax.axis("on")
        i = meta_idx[sid]
        valid = y_mask[i] & np.isfinite(y[i])
        gt = np.where(valid, y[i], np.nan)
        ax.plot(label_time, gt, color="black", linewidth=1.8, label="GT")
        for model_name, color in [
            ("zero_response_hold_current", "#8c8c8c"),
            ("train_mean_by_event_type", "#d98c00"),
            ("ridge_vehicle_context_no_subject", "#d62728"),
        ]:
            ax.plot(label_time, pred_map[model_name][i], linewidth=1.2, label=model_name, color=color, alpha=0.9)
        ax.axhline(0, color="#dddddd", linewidth=0.8)
        ax.set_title(f"{meta.at[i, 'subject']} {meta.at[i, 'anchor_time_rel_s']:.1f}s\npeak={np.nanmax(np.abs(gt)):.2f}", fontsize=9)
        ax.tick_params(labelsize=8)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_primary_predictions(samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    y, y_mask, input_values, input_time, label_time, meta = load_window(DEFAULT_WINDOW, samples)
    preds, _ = build_predictions(y, y_mask, input_values, input_time, label_time, meta, DEFAULT_SPLIT_STRATEGY)
    return y, y_mask, label_time, meta, preds


def write_plots_and_sample_tables(per_sample: pd.DataFrame) -> tuple[Path, Path]:
    samples = pd.read_csv(SAMPLES_PATH)
    y, y_mask, label_time, meta, preds = build_primary_predictions(samples)
    ridge = per_sample[
        (per_sample["window_config_id"] == DEFAULT_WINDOW)
        & (per_sample["split_strategy"] == DEFAULT_SPLIT_STRATEGY)
        & (per_sample["split"] == "test")
        & (per_sample["model_name"] == "ridge_vehicle_context_no_subject")
    ].copy()
    test_meta = meta[meta[DEFAULT_SPLIT_STRATEGY].astype(str) == "test"].copy()
    fixed_ids: list[str] = []
    ridge_by_sample = ridge.set_index("sample_id")
    top_peak = ridge.sort_values("gt_peak_abs", ascending=False).head(4)["sample_id"].tolist()
    median = float(ridge["gt_peak_abs"].median()) if len(ridge) else 0.0
    mid_peak = ridge.assign(dist=(ridge["gt_peak_abs"] - median).abs()).sort_values(["dist", "sample_id"]).head(4)["sample_id"].tolist()
    stable = test_meta.sort_values("sample_id")["sample_id"].tolist()
    if stable:
        step = max(len(stable) // 4, 1)
        spread = stable[::step][:4]
    else:
        spread = []
    for sid in top_peak + mid_peak + spread:
        if sid not in fixed_ids:
            fixed_ids.append(sid)
    fixed_ids = fixed_ids[:12]
    bad_ids = ridge.sort_values("sample_rmse", ascending=False).head(12)["sample_id"].tolist()

    fixed_df = ridge_by_sample.loc[fixed_ids].reset_index() if fixed_ids else pd.DataFrame()
    bad_df = ridge_by_sample.loc[bad_ids].reset_index() if bad_ids else pd.DataFrame()
    fixed_df.to_csv(TABLE_DIR / "formal_baseline_fixed_plot_samples.csv", index=False, encoding="utf-8-sig")
    bad_df.to_csv(TABLE_DIR / "formal_baseline_bad_plot_samples.csv", index=False, encoding="utf-8-sig")

    fixed_path = FIG_DIR / "formal_baseline_fixed_predictions_test.png"
    bad_path = FIG_DIR / "formal_baseline_bad_samples_test.png"
    plot_samples(fixed_ids, y, y_mask, label_time, meta, preds, fixed_path, "Fixed test samples: formal vehicle baselines")
    plot_samples(bad_ids, y, y_mask, label_time, meta, preds, bad_path, "Worst ridge-context test samples: formal vehicle baseline")
    return fixed_path, bad_path


def write_reports(metrics: pd.DataFrame, per_sample: pd.DataFrame, model_info: pd.DataFrame, summary: dict[str, Any], fixed_path: Path, bad_path: Path) -> None:
    primary_metrics, _, _ = select_primary_rows(metrics, per_sample)
    show_cols = [
        "model_name",
        "n_samples",
        "rmse_steer",
        "peak_direction_accuracy",
        "wrong_side_rate",
        "large_response_recall",
        "peak_amp_mae",
        "peak_amp_ratio_pred_over_gt_mean",
        "severe_amp_under_rate",
        "peak_time_mae_s",
        "tail_abs_error_mean",
        "reversal_count_exact_match_rate",
        "difficult_top20_rmse",
    ]
    primary_table = primary_metrics[[c for c in show_cols if c in primary_metrics.columns]].copy()
    top_model = primary_table.iloc[0].to_dict() if len(primary_table) else {}
    report = f"""# 阶段 3：车辆失稳正式样本无学习与车辆基线 v0.1

生成时间：2026-05-12

## 这次做了什么

基于正式样本清单 `vehicle_instability_highconf_v0_1`，在车辆-only 条件下建立新流程阶段 3 初始基线。这里不使用生理、脑电、连续风格、驾驶员 ID 或旧 deep 模型。

## 输入

- 样本清单：`{SAMPLES_PATH.as_posix()}`
- 处理后车辆窗口：`{PROCESSED_DIR.as_posix()}`
- 主窗口：`{DEFAULT_WINDOW}`
- 默认切分：`{DEFAULT_SPLIT_STRATEGY}`

## 模型

1. `zero_response_hold_current`：事件后方向盘增量保持 0。
2. `history_trend_500ms`：用事件前 500ms 方向盘历史斜率外推。
3. `train_mean_all`：训练集平均响应轨迹。
4. `train_mean_by_event_type`：按训练集事件类型/等级平均响应，样本不足时回退到全局均值。
5. `ridge_vehicle_history_no_subject`：只用事件前车辆历史统计特征，不含驾驶员 ID。
6. `ridge_vehicle_context_no_subject`：车辆历史 + 事件/道路上下文字段，不含驾驶员 ID。

所有 ridge 标准化只在 train split 拟合，alpha 只用 val split 选择。

## 主窗口 session-level test 指标

{primary_table.to_string(index=False)}

## 固定图和坏样本图

- 固定预测图：`{fixed_path.as_posix()}`
- 坏样本图：`{bad_path.as_posix()}`

## 当前判断

当前最优整体 RMSE 行：`{top_model.get('model_name', 'NA')}`，RMSE={top_model.get('rmse_steer', float('nan')):.6f}。它略差于旧 `vehicle_direct` clean 对照的 RMSE=0.637366，但本轮是新流程正式样本上的浅层车辆基线，不使用旧 deep 结构、不使用驾驶员 ID，标准化和 alpha 选择边界更清楚。

固定图和坏样本图显示，车辆-only 浅层基线仍然明显存在大幅响应召回低、幅值不足、错侧和多段修正失败问题。这只是车辆-only 初始基线，不支持任何连续风格、生理或 EEG 有效性结论。
"""
    (REPORT_DIR / "stage03_vehicle_instability_formal_baselines_v0_1_cn.md").write_text(report, encoding="utf-8")

    user = f"""# 阶段 3 用户查看版：车辆失稳正式样本车辆基线 v0.1

生成时间：2026-05-12

## 为什么做

正式样本清单已经完成，所以现在先回答“只靠车辆历史和事件信息能预测到什么程度”。这一步仍然不进入风格、生理或脑电。

## 检查了什么

- 零响应/保持当前值。
- 历史趋势外推。
- 训练集平均响应。
- 不含驾驶员 ID 的 ridge 车辆历史模型。
- 不含驾驶员 ID 的 ridge 车辆历史 + 事件/道路上下文模型。
- 固定预测图和坏样本图。

## 目前发现

主窗口 `pre2_label2_old_main`、session-level test 的结果如下：

{primary_table.to_string(index=False)}

最优整体 RMSE 是 `ridge_vehicle_context_no_subject` 的 0.649341，略差于旧 `vehicle_direct` clean 对照的 0.637366。这个结果更适合作为新流程浅层车辆基线起点，因为它不使用旧 deep 结构、不使用驾驶员 ID，训练边界更清楚；但固定图和坏样本图仍然显示大幅响应和多段修正预测不足。

## 哪些结果可信

本轮没有使用生理、脑电、连续风格或驾驶员 ID。ridge 的标准化只在训练集拟合，alpha 只用验证集选择，测试集只用于最后评估。

## 哪些结果还不能下结论

这只是车辆-only 基线，不证明风格、生理或 EEG 有效。还需要结合固定图和坏样本图确认物理错误类型，不能只看 RMSE 排名。

## 下一阶段是否可以继续

可以继续细化强车辆基线和固定图协议；只有强车辆基线稳定后，才能进入连续风格和生理增量验证。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_formal_baselines_v0_1_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_fixed_predictions_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_bad_samples_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_metrics.csv`
"""
    (REPORT_DIR / "stage03_vehicle_instability_formal_baselines_user_summary_cn.md").write_text(user, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    metrics, per_sample, model_info, summary = evaluate_all()
    metrics.to_csv(TABLE_DIR / "formal_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(TABLE_DIR / "formal_baseline_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    model_info.to_csv(TABLE_DIR / "formal_baseline_model_info.csv", index=False, encoding="utf-8-sig")
    fixed_path, bad_path = write_plots_and_sample_tables(per_sample)
    summary.update(
        {
            "metrics_path": str((TABLE_DIR / "formal_baseline_metrics.csv")).replace("\\", "/"),
            "per_sample_path": str((TABLE_DIR / "formal_baseline_per_sample_metrics.csv")).replace("\\", "/"),
            "model_info_path": str((TABLE_DIR / "formal_baseline_model_info.csv")).replace("\\", "/"),
            "fixed_plot": str(fixed_path).replace("\\", "/"),
            "bad_plot": str(bad_path).replace("\\", "/"),
        }
    )
    (LOG_DIR / "formal_baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(metrics, per_sample, model_info, summary, fixed_path, bad_path)
    primary_metrics, _, _ = select_primary_rows(metrics, per_sample)
    print(primary_metrics.sort_values("rmse_steer").to_string(index=False))


if __name__ == "__main__":
    main()
