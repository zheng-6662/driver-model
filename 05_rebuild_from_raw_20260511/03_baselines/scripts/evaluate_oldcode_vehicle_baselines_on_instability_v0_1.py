# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as old_eval  # noqa: E402


DATASET_DIR = ROOT / "03_processed_datasets" / "vehicle_instability_allraw_highconf_v0_1"
SAMPLES_PATH = DATASET_DIR / "tables" / "selected_samples_vehicle_instability_highconf_v0_1.csv"
OUT_DIR = ROOT / "03_baselines" / "oldcode_vehicle_baselines_on_instability_v0_1"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"

WINDOWS = [
    "pre1_label2_event_trigger",
    "pre2_label2_old_main",
    "pre3_label3_response_coverage",
]
SPLIT_STRATEGIES = ["random_event_split", "session_level_split", "subject_level_split"]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def no_subject_ridge(
    y: np.ndarray,
    y_mask: np.ndarray,
    input_values: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    input_mask = np.isfinite(input_values)
    x_all, names = old_eval.extract_vehicle_features(input_values, input_mask, meta)
    keep = [not name.startswith("subject=") for name in names]
    kept_names = [name for name, ok in zip(names, keep) if ok]
    if not kept_names or train_idx.size < 5:
        return np.zeros_like(y, dtype=np.float32), {
            "status": "no_train_samples_or_features",
            "feature_count": len(kept_names),
            "removed_subject_onehot": True,
        }
    x = x_all[:, keep].astype(np.float64)
    mu = x[train_idx].mean(axis=0, keepdims=True)
    sigma = x[train_idx].std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    xs = (x - mu) / sigma
    xd = np.c_[np.ones((xs.shape[0], 1)), xs]
    y_train = np.nan_to_num(np.where(y_mask, y, np.nan), nan=0.0).astype(np.float64)
    val_eval = val_idx if val_idx.size else train_idx
    best_alpha = old_eval.RIDGE_ALPHAS[0]
    best_score = float("inf")
    best_pred: np.ndarray | None = None
    for alpha in old_eval.RIDGE_ALPHAS:
        xt = xd[train_idx]
        reg = np.eye(xt.shape[1], dtype=np.float64) * float(alpha)
        reg[0, 0] = 0.0
        coef = np.linalg.solve(xt.T @ xt + reg, xt.T @ y_train[train_idx])
        pred = (xd @ coef).astype(np.float32)
        score = old_eval.rmse(y[val_eval], pred[val_eval], y_mask[val_eval])
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_pred = pred
    assert best_pred is not None
    return best_pred, {
        "status": "ok",
        "selected_alpha": float(best_alpha),
        "val_rmse_for_alpha": float(best_score),
        "train_rmse_selected_alpha": float(old_eval.rmse(y[train_idx], best_pred[train_idx], y_mask[train_idx])),
        "feature_count": len(kept_names),
        "removed_subject_onehot": True,
    }


def load_window(window_id: str, all_samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    z = np.load(DATASET_DIR / "arrays" / f"{window_id}.npz", allow_pickle=True)
    y = z["label_steer_delta"].astype(np.float32)
    y_mask = z["label_valid_mask"].astype(bool)
    input_values = z["input_values"].astype(np.float32)
    input_time = z["input_time_rel_s"].astype(np.float32)
    label_time = z["label_time_rel_s"].astype(np.float32)
    idx_df = pd.read_csv(DATASET_DIR / "tables" / f"sample_index_{window_id}.csv")
    keys = ["sample_id", "event_uid", "subject", "session_stamp", "anchor_time_rel_s", "window_config_id"]
    meta = idx_df.merge(all_samples, on=keys, how="left", suffixes=("", "_selected"))
    return y, y_mask, input_values, input_time, label_time, meta


def write_reports(metrics: pd.DataFrame, best_rows: pd.DataFrame, model_info: pd.DataFrame, summary: dict[str, Any]) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    pre2_session = test[
        (test["window_config_id"] == "pre2_label2_old_main")
        & (test["split_strategy"] == "session_level_split")
    ].copy()
    cols = [
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
    pre2_table = pre2_session[cols].sort_values("rmse_steer") if len(pre2_session) else pd.DataFrame(columns=cols)
    best_table = best_rows[
        [
            "window_config_id",
            "split_strategy",
            "model_name",
            "rmse_steer",
            "peak_direction_accuracy",
            "wrong_side_rate",
            "severe_amp_under_rate",
            "difficult_top20_rmse",
        ]
    ] if len(best_rows) else pd.DataFrame()

    report = f"""# 旧车辆代码在全原始失稳高置信样本上的诊断测试 v0.1

生成时间：2026-05-12

## 这次测试做了什么

这次没有直接继续训练风格/生理模型，而是把 908 个高置信车辆失稳样本转换成旧阶段 3 车辆基线代码可读的窗口格式，然后复用旧车辆基线评价逻辑进行诊断。

## 输入

- 处理后车辆窗口：`{DATASET_DIR.as_posix()}`
- 样本清单：`{SAMPLES_PATH.as_posix()}`
- 旧代码逻辑：`03_baselines/scripts/evaluate_stage3_vehicle_baselines.py`

## 重要边界

1. 这不是正式阶段 3 结论，只是旧代码在新失稳样本上的第一轮诊断。
2. 锚点来自非转向车辆动力学 onset，不是道路弯道 onset。
3. `ridge_vehicle_summary` 沿用旧代码，会包含被试 one-hot；因此同时输出 `ridge_vehicle_no_subject` 作为去掉被试 one-hot 的对照。
4. 不使用生理、脑电、连续风格，不改原始 CSV。
5. 当前结果不能用于证明风格或生理有效。

## pre2 + session-level test 关键表

{pre2_table.to_string(index=False)}

## 各窗口/切分测试集最优行

{best_table.to_string(index=False)}

## 模型拟合信息

{model_info.to_string(index=False)}

## 快速判断

这批失稳样本可以被旧车辆代码读取和评估。后续如果要真正比较旧深度模型，需要用本次输出的旧 manifest 做一个独立的旧模型 smoke/full run，并把结果和这里的无学习/车辆 ridge 诊断结果放在同一张表里。
"""
    (REPORT_DIR / "oldcode_vehicle_baseline_on_instability_v0_1_cn.md").write_text(report, encoding="utf-8")

    user = f"""# 阶段 3 用户查看版：旧车辆代码测试全原始失稳样本 v0.1

生成时间：2026-05-12

## 为什么做

你指出之前 404 个主要是弯道样本，不是你真正要的车辆失稳样本。因此这次先不用旧弯道样本，而是把全原始车辆数据重新筛出的高置信失稳事件喂给旧车辆代码，看看这些样本在旧评价体系下是什么难度。

## 检查了什么

- 908 个高置信车辆失稳事件是否能转成旧代码窗口。
- 旧的无学习基线和车辆 ridge 基线在这些样本上的误差。
- 方向、错侧、幅值不足、峰值时间、尾段误差、反向/多段修正等物理指标。
- 固定预测图和坏样本图。

## 目前发现

pre2 窗口、session-level test 的结果如下：

{pre2_table.to_string(index=False)}

## 哪些结果可信

- 结果使用的是重新筛选的车辆失稳事件，不是那 404 个弯道候选。
- 输入只来自原始车辆 CSV 派生窗口，原始文件未被修改。
- 训练/验证/测试 split 已分开，ridge 标准化和 alpha 选择只在 train/val 内完成。

## 哪些还不能下结论

- 这还不是正式深度模型训练结果。
- `ridge_vehicle_summary` 是旧代码原样逻辑，含被试 one-hot，只能作为旧代码诊断；更公平时应优先看 `ridge_vehicle_no_subject` 或后续 subject-level split。
- 不能由此判断生理、脑电或连续风格有效。

## 推荐优先查看

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_baseline_on_instability_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_baseline_metrics.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_best_test_by_window_split.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/figures/oldcode_fixed_predictions_pre2_session_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/figures/oldcode_bad_samples_pre2_session_test_ridge.png`
"""
    (REPORT_DIR / "stage03_oldcode_instability_user_summary_cn.md").write_text(user, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    all_samples = pd.read_csv(SAMPLES_PATH)
    all_metrics: list[pd.DataFrame] = []
    model_info_rows: list[dict[str, Any]] = []
    fixed_plot_records: list[dict[str, Any]] = []

    for window_id in WINDOWS:
        y, y_mask, input_values, input_time, label_time, meta = load_window(window_id, all_samples)
        gt_peak = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)
        for split_strategy in SPLIT_STRATEGIES:
            train_idx = np.where(meta[split_strategy].to_numpy() == "train")[0]
            val_idx = np.where(meta[split_strategy].to_numpy() == "val")[0]
            large_thr = float(np.nanpercentile(gt_peak[train_idx], 75)) if train_idx.size else float(np.nanpercentile(gt_peak, 75))
            difficult_thr = float(np.nanpercentile(gt_peak[train_idx], 80)) if train_idx.size else float(np.nanpercentile(gt_peak, 80))
            preds, info = old_eval.make_baseline_predictions(y, y_mask, input_values, input_time, label_time, meta, split_strategy)
            no_subject_pred, no_subject_info = no_subject_ridge(y, y_mask, input_values, meta, train_idx, val_idx)
            preds["ridge_vehicle_no_subject"] = no_subject_pred
            info["ridge_vehicle_no_subject"] = no_subject_info

            for model_name, model_info in info.items():
                model_info_rows.append(
                    {
                        "window_config_id": window_id,
                        "split_strategy": split_strategy,
                        "model_name": model_name,
                        **model_info,
                    }
                )

            for split_name in ["train", "val", "test"]:
                split_idx = np.where(meta[split_strategy].to_numpy() == split_name)[0]
                if split_idx.size == 0:
                    continue
                split_meta = meta.iloc[split_idx].reset_index(drop=True)
                for model_name, pred in preds.items():
                    rows = old_eval.sample_metric_rows(
                        y[split_idx],
                        pred[split_idx],
                        y_mask[split_idx],
                        label_time,
                        split_meta,
                        model_name,
                        split_strategy,
                        split_name,
                        window_id,
                        large_thr=large_thr,
                        difficult_thr=difficult_thr,
                    )
                    if rows:
                        all_metrics.append(pd.DataFrame(rows))

            if window_id == "pre2_label2_old_main" and split_strategy == "session_level_split":
                test_idx = np.where(meta[split_strategy].to_numpy() == "test")[0]
                if test_idx.size:
                    order = test_idx[np.argsort(-gt_peak[test_idx])]
                    fixed = order[:6].tolist()
                    mid = test_idx[np.argsort(np.abs(gt_peak[test_idx] - np.nanmedian(gt_peak[test_idx])))]
                    fixed.extend(mid[:6].tolist())
                    fixed = list(dict.fromkeys(fixed))[:12]
                    for rank, idx in enumerate(fixed, start=1):
                        fixed_plot_records.append(
                            {
                                "plot_type": "fixed_pre2_session_test",
                                "rank": rank,
                                "array_row": int(idx),
                                "sample_id": meta.iloc[idx]["sample_id"],
                                "event_uid": meta.iloc[idx]["event_uid"],
                                "subject": meta.iloc[idx]["subject"],
                                "gt_peak_abs": float(gt_peak[idx]),
                            }
                        )
                    old_eval.draw_prediction_grid(
                        FIG_DIR / "oldcode_fixed_predictions_pre2_session_test.png",
                        label_time,
                        y,
                        preds,
                        meta,
                        fixed,
                        "Old vehicle baselines on instability: fixed pre2 session-test",
                    )
                    ridge = preds["ridge_vehicle_summary"]
                    sample_rmse = np.sqrt(np.nanmean(np.square(ridge[test_idx] - y[test_idx]), axis=1))
                    bad = test_idx[np.argsort(-sample_rmse)[:12]].tolist()
                    old_eval.draw_prediction_grid(
                        FIG_DIR / "oldcode_bad_samples_pre2_session_test_ridge.png",
                        label_time,
                        y,
                        preds,
                        meta,
                        bad,
                        "Old vehicle baselines on instability: bad samples ridge",
                    )

    per_sample = pd.concat(all_metrics, ignore_index=True)
    metrics = old_eval.aggregate_metrics(per_sample)
    model_info = pd.DataFrame(model_info_rows)
    fixed_records = pd.DataFrame(fixed_plot_records)

    per_sample.to_csv(TABLE_DIR / "oldcode_instability_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLE_DIR / "oldcode_instability_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    model_info.to_csv(TABLE_DIR / "oldcode_instability_model_info.csv", index=False, encoding="utf-8-sig")
    fixed_records.to_csv(TABLE_DIR / "oldcode_instability_fixed_plot_sample_set.csv", index=False, encoding="utf-8-sig")

    test_metrics = metrics[metrics["split"] == "test"].copy()
    best_rows = (
        test_metrics.sort_values(["window_config_id", "split_strategy", "rmse_steer"])
        .groupby(["window_config_id", "split_strategy"])
        .head(1)
    )
    best_rows.to_csv(TABLE_DIR / "oldcode_instability_best_test_by_window_split.csv", index=False, encoding="utf-8-sig")

    summary = {
        "dataset_dir": str(DATASET_DIR).replace("\\", "/"),
        "windows": WINDOWS,
        "split_strategies": SPLIT_STRATEGIES,
        "models": sorted(per_sample["model_name"].unique().tolist()),
        "metric_rows": int(len(metrics)),
        "per_sample_rows": int(len(per_sample)),
        "best_test": best_rows[
            [
                "window_config_id",
                "split_strategy",
                "model_name",
                "rmse_steer",
                "peak_direction_accuracy",
                "wrong_side_rate",
                "severe_amp_under_rate",
            ]
        ].to_dict(orient="records"),
        "server_used": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "oldcode_instability_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_reports(metrics, best_rows, model_info, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
