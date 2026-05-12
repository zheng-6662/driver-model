# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
SCRIPT_DIR = ROOT / "03_baselines" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stage3_vehicle_baselines as base_eval  # noqa: E402


DATASET_DIR = ROOT / "03_processed_datasets" / "vehicle_road_curvature_v0_2"
BASELINE_DIR = ROOT / "03_baselines" / "stage03_vehicle_baselines_v0_2"
OUT_DIR = ROOT / "03_baselines" / "stage03_vehicle_diagnostics_v0_3"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
REPORT_DIR = ROOT / "09_reports"


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_meta(window_id: str) -> pd.DataFrame:
    all_samples = pd.read_csv(base_eval.SAMPLES_PATH)
    split_table = pd.read_csv(base_eval.SPLIT_PATH)
    split_cols = ["event_uid"] + base_eval.SPLIT_STRATEGIES
    all_samples = all_samples.merge(split_table[split_cols], on="event_uid", how="left")
    idx_df = pd.read_csv(DATASET_DIR / "tables" / f"sample_index_{window_id}.csv")
    keys = [
        "sample_id",
        "event_uid",
        "subject",
        "session_stamp",
        "anchor_time_rel_s",
        "anchor_time_abs_s",
        "window_config_id",
    ]
    return idx_df.merge(all_samples, on=keys, how="left")


def load_window(window_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    z = np.load(DATASET_DIR / "arrays" / f"{window_id}.npz", allow_pickle=True)
    y = z["label_steer_delta"].astype(np.float32)
    y_mask = z["label_valid_mask"].astype(bool)
    input_values = z["input_values"].astype(np.float32)
    label_time = z["label_time_rel_s"].astype(np.float32)
    meta = load_meta(window_id)
    return y, y_mask, input_values, label_time, meta


def no_subject_features(input_values: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    input_mask = np.isfinite(input_values)
    X, feature_names = base_eval.extract_vehicle_features(input_values, input_mask, meta)
    keep = [not name.startswith("subject=") for name in feature_names]
    kept_names = [name for name, ok in zip(feature_names, keep) if ok]
    if not kept_names:
        return np.zeros((X.shape[0], 0), dtype=np.float64), []
    return X[:, keep].astype(np.float64), kept_names


def standardize_from_train(X: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mu = X[train_idx].mean(axis=0, keepdims=True)
    sigma = X[train_idx].std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    return (X - mu) / sigma


def squared_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T
    out = a2 + b2 - 2.0 * (A @ B.T)
    return np.maximum(out, 0.0)


def finite_y(y: np.ndarray, y_mask: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.where(y_mask, y, np.nan), nan=0.0).astype(np.float64)


def ridge_predict(
    Xs: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    Y = finite_y(y, y_mask)
    Xd = np.c_[np.ones((Xs.shape[0], 1)), Xs]
    val_eval = val_idx if val_idx.size else train_idx
    best_alpha = None
    best_score = float("inf")
    best_pred = None
    for alpha in base_eval.RIDGE_ALPHAS:
        Xt = Xd[train_idx]
        reg = np.eye(Xt.shape[1], dtype=np.float64) * float(alpha)
        reg[0, 0] = 0.0
        coef = np.linalg.solve(Xt.T @ Xt + reg, Xt.T @ Y[train_idx])
        pred = (Xd @ coef).astype(np.float32)
        score = base_eval.rmse(y[val_eval], pred[val_eval], y_mask[val_eval])
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_pred = pred
    assert best_pred is not None and best_alpha is not None
    return best_pred, {
        "status": "ok",
        "selected_alpha": float(best_alpha),
        "val_rmse_for_selection": float(best_score),
        "train_rmse": float(base_eval.rmse(y[train_idx], best_pred[train_idx], y_mask[train_idx])),
    }


def knn_predict(
    Xs: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    Y = finite_y(y, y_mask)
    Xtrain = Xs[train_idx]
    dist = squared_dist(Xs, Xtrain)
    train_pos = {int(idx): pos for pos, idx in enumerate(train_idx.tolist())}
    for row_idx, pos in train_pos.items():
        if row_idx < dist.shape[0]:
            dist[row_idx, pos] = np.inf
    k_grid = [1, 3, 5, 9, 15, 25, 45]
    k_grid = [k for k in k_grid if k <= max(1, len(train_idx) - 1)]
    val_eval = val_idx if val_idx.size else train_idx
    best_k = k_grid[0]
    best_score = float("inf")
    best_pred = None
    for k in k_grid:
        nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        pred = np.mean(Y[train_idx][nn], axis=1).astype(np.float32)
        score = base_eval.rmse(y[val_eval], pred[val_eval], y_mask[val_eval])
        if score < best_score:
            best_score = score
            best_k = k
            best_pred = pred
    assert best_pred is not None
    return best_pred, {
        "status": "ok",
        "selected_k": int(best_k),
        "val_rmse_for_selection": float(best_score),
        "train_rmse_leave_one_out": float(base_eval.rmse(y[train_idx], best_pred[train_idx], y_mask[train_idx])),
    }


def rbf_kernel(gamma: float, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.exp(-float(gamma) * squared_dist(A, B))


def rbf_krr_predict(
    Xs: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    Y = finite_y(y, y_mask)
    Xt = Xs[train_idx]
    dtrain = squared_dist(Xt, Xt)
    nonzero = dtrain[dtrain > 1e-9]
    median_d = float(np.median(nonzero)) if nonzero.size else 1.0
    base_gamma = 1.0 / max(median_d, 1e-6)
    gamma_grid = [0.25 * base_gamma, base_gamma, 4.0 * base_gamma]
    alpha_grid = [0.01, 0.1, 1.0, 10.0]
    val_eval = val_idx if val_idx.size else train_idx
    best: tuple[float, float] | None = None
    best_score = float("inf")
    best_pred = None
    eye = np.eye(len(train_idx), dtype=np.float64)
    for gamma in gamma_grid:
        K_train = rbf_kernel(gamma, Xt, Xt)
        K_all = rbf_kernel(gamma, Xs, Xt)
        for alpha in alpha_grid:
            coef = np.linalg.solve(K_train + float(alpha) * eye, Y[train_idx])
            pred = (K_all @ coef).astype(np.float32)
            score = base_eval.rmse(y[val_eval], pred[val_eval], y_mask[val_eval])
            if score < best_score:
                best_score = score
                best = (gamma, alpha)
                best_pred = pred
    assert best is not None and best_pred is not None
    return best_pred, {
        "status": "ok",
        "selected_gamma": float(best[0]),
        "selected_alpha": float(best[1]),
        "median_train_sqdist": float(median_d),
        "val_rmse_for_selection": float(best_score),
        "train_rmse": float(base_eval.rmse(y[train_idx], best_pred[train_idx], y_mask[train_idx])),
    }


def split_indices(meta: pd.DataFrame, split_strategy: str) -> dict[str, np.ndarray]:
    values = meta[split_strategy].astype(str).to_numpy()
    return {name: np.where(values == name)[0] for name in ["train", "val", "test"]}


def evaluate_model_set(
    window_id: str,
    split_strategy: str,
    y: np.ndarray,
    y_mask: np.ndarray,
    input_values: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    idx = split_indices(meta, split_strategy)
    train_idx = idx["train"]
    val_idx = idx["val"]
    gt_peak = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)
    large_thr = float(np.nanpercentile(gt_peak[train_idx], 75)) if train_idx.size else float(np.nanpercentile(gt_peak, 75))
    difficult_thr = float(np.nanpercentile(gt_peak[train_idx], 80)) if train_idx.size else float(np.nanpercentile(gt_peak, 80))
    X, feature_names = no_subject_features(input_values, meta)
    Xs = standardize_from_train(X, train_idx)
    model_outputs: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
        "ridge_vehicle_no_subject": ridge_predict(Xs, y, y_mask, train_idx, val_idx),
        "knn_vehicle_no_subject": knn_predict(Xs, y, y_mask, train_idx, val_idx),
        "rbf_krr_vehicle_no_subject": rbf_krr_predict(Xs, y, y_mask, train_idx, val_idx),
    }

    rows: list[pd.DataFrame] = []
    info_rows: list[dict[str, Any]] = []
    pred_by_name: dict[str, np.ndarray] = {}
    for model_name, (pred, info) in model_outputs.items():
        pred_by_name[model_name] = pred
        info_rows.append(
            {
                "window_config_id": window_id,
                "split_strategy": split_strategy,
                "model_name": model_name,
                "feature_count_no_subject": len(feature_names),
                "excluded_feature_prefix": "subject=",
                **info,
            }
        )
        for split_name, split_idx in idx.items():
            if split_idx.size == 0:
                continue
            sample_rows = base_eval.sample_metric_rows(
                y[split_idx],
                pred[split_idx],
                y_mask[split_idx],
                label_time,
                meta.iloc[split_idx].reset_index(drop=True),
                model_name,
                split_strategy,
                split_name,
                window_id,
                large_thr=large_thr,
                difficult_thr=difficult_thr,
            )
            if sample_rows:
                rows.append(pd.DataFrame(sample_rows))
    per_sample = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return per_sample, pd.DataFrame(info_rows), pred_by_name


def overfit_subset_report(window_id: str = "pre2_label2_old_main", split_strategy: str = "session_level_split") -> pd.DataFrame:
    y, y_mask, input_values, _, meta = load_window(window_id)
    idx = split_indices(meta, split_strategy)
    train_idx = idx["train"]
    test_idx = idx["test"]
    X, _ = no_subject_features(input_values, meta)
    Xs = standardize_from_train(X, train_idx)
    Y = finite_y(y, y_mask)
    rows: list[dict[str, Any]] = []
    for subset_size in [8, 16, 32, 64, 128]:
        if train_idx.size < subset_size:
            continue
        # Use the largest-response train samples so the test is not made easy by tiny labels.
        gt_peak = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)
        subset = train_idx[np.argsort(-gt_peak[train_idx])[:subset_size]]
        Xsub = Xs[subset]
        dsub = squared_dist(Xsub, Xsub)
        nonzero = dsub[dsub > 1e-9]
        median_d = float(np.median(nonzero)) if nonzero.size else 1.0
        gamma = 4.0 / max(median_d, 1e-6)
        alpha = 1e-6
        Ksub = rbf_kernel(gamma, Xsub, Xsub)
        coef = np.linalg.solve(Ksub + alpha * np.eye(len(subset)), Y[subset])
        Kall = rbf_kernel(gamma, Xs, Xsub)
        pred = (Kall @ coef).astype(np.float32)
        rows.append(
            {
                "window_config_id": window_id,
                "split_strategy": split_strategy,
                "model_name": "rbf_krr_overfit_no_subject",
                "subset_size": subset_size,
                "subset_train_rmse": base_eval.rmse(y[subset], pred[subset], y_mask[subset]),
                "full_train_rmse": base_eval.rmse(y[train_idx], pred[train_idx], y_mask[train_idx]),
                "test_rmse": base_eval.rmse(y[test_idx], pred[test_idx], y_mask[test_idx]),
                "gamma": gamma,
                "alpha": alpha,
                "subset_selection": "largest_gt_peak_train_samples",
            }
        )
    return pd.DataFrame(rows)


def diagnostic_tables(stronger_per_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_per = pd.read_csv(BASELINE_DIR / "tables" / "stage03_per_sample_metrics.csv")
    fixed = pd.read_csv(BASELINE_DIR / "tables" / "stage03_fixed_plot_sample_set.csv")
    target = base_per[
        (base_per["window_config_id"] == "pre2_label2_old_main")
        & (base_per["split_strategy"] == "session_level_split")
        & (base_per["split"] == "test")
        & (base_per["model_name"] == "ridge_vehicle_summary")
    ].copy()
    stronger_target = stronger_per_sample[
        (stronger_per_sample["window_config_id"] == "pre2_label2_old_main")
        & (stronger_per_sample["split_strategy"] == "session_level_split")
        & (stronger_per_sample["split"] == "test")
        & (stronger_per_sample["model_name"].isin(["ridge_vehicle_no_subject", "rbf_krr_vehicle_no_subject", "knn_vehicle_no_subject"]))
    ].copy()

    fixed_diag = fixed.merge(target, on=["sample_id", "event_uid", "subject"], how="left", suffixes=("", "_ridge_with_subject"))
    bad_diag = target.sort_values("sample_rmse", ascending=False).head(12).copy()
    bad_diag = bad_diag.merge(
        stronger_target.pivot_table(index="sample_id", columns="model_name", values="sample_rmse", aggfunc="first").reset_index(),
        on="sample_id",
        how="left",
    )
    bins = [0.0, 0.25, 0.5, 1.0, 2.0, np.inf]
    labels = ["0-0.25", "0.25-0.5", "0.5-1.0", "1.0-2.0", ">=2.0"]
    target["gt_peak_abs_bin"] = pd.cut(target["gt_peak_abs"], bins=bins, labels=labels, include_lowest=True)
    bucket = (
        target.groupby(["gt_peak_abs_bin"], observed=False)
        .agg(
            n_samples=("sample_id", "count"),
            rmse_mean=("sample_rmse", "mean"),
            wrong_side_rate=("wrong_side", "mean"),
            severe_under_rate=("severe_amp_under", "mean"),
            peak_amp_ratio_mean=("peak_amp_ratio_pred_over_gt", "mean"),
        )
        .reset_index()
    )
    return fixed_diag, bad_diag, bucket


def build_comparison(stronger_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_DIR / "tables" / "stage03_baseline_metrics.csv")
    baseline = baseline[baseline["split"] == "test"].copy()
    baseline["feature_protocol_note"] = np.where(
        baseline["model_name"] == "ridge_vehicle_summary",
        "v0.2 includes subject one-hot; use as driver-id control, not final pure-vehicle baseline",
        "no learned vehicle model or event average baseline",
    )
    stronger = stronger_metrics[stronger_metrics["split"] == "test"].copy()
    stronger["feature_protocol_note"] = "v0.3 no subject one-hot; pure vehicle/history/event-road features"
    cols = [
        "window_config_id",
        "split_strategy",
        "split",
        "model_name",
        "n_samples",
        "rmse_steer",
        "peak_direction_accuracy",
        "wrong_side_rate",
        "large_response_recall",
        "peak_amp_ratio_pred_over_gt_mean",
        "severe_amp_under_rate",
        "peak_time_mae_s",
        "onset_delay_mae_s",
        "tail_abs_error_mean",
        "reversal_count_exact_match_rate",
        "difficult_top20_rmse",
        "feature_protocol_note",
    ]
    keep_models = {
        "zero_response",
        "train_mean_by_event_type",
        "ridge_vehicle_summary",
        "ridge_vehicle_no_subject",
        "knn_vehicle_no_subject",
        "rbf_krr_vehicle_no_subject",
    }
    combo = pd.concat([baseline[cols], stronger[cols]], ignore_index=True)
    return combo[combo["model_name"].isin(keep_models)].sort_values(
        ["window_config_id", "split_strategy", "rmse_steer", "model_name"]
    )


def draw_bar_chart(df: pd.DataFrame, out_path: Path) -> None:
    rows = df[
        (df["window_config_id"] == "pre2_label2_old_main")
        & (df["split_strategy"] == "session_level_split")
        & (df["split"] == "test")
    ].sort_values("rmse_steer")
    rows = rows[["model_name", "rmse_steer", "wrong_side_rate", "severe_amp_under_rate"]].head(8)
    width, height = 1500, 780
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
        small = font
        title_font = font
    draw.text((40, 25), "Stage 3 v0.3 pre2 session-level test: vehicle baseline comparison", fill=(0, 0, 0), font=title_font)
    max_rmse = max(float(rows["rmse_steer"].max()), 1e-6)
    x0, y0 = 420, 100
    bar_w = 820
    row_h = 74
    for i, row in enumerate(rows.itertuples(index=False)):
        y = y0 + i * row_h
        draw.text((40, y + 10), str(row.model_name), fill=(0, 0, 0), font=small)
        w = int(bar_w * float(row.rmse_steer) / max_rmse)
        color = (40, 110, 190) if "no_subject" in str(row.model_name) else (150, 150, 150)
        draw.rectangle((x0, y + 8, x0 + w, y + 38), fill=color)
        draw.text(
            (x0 + bar_w + 25, y + 6),
            f"RMSE={row.rmse_steer:.3f}  wrong={row.wrong_side_rate:.2f}  under={row.severe_amp_under_rate:.2f}",
            fill=(0, 0, 0),
            font=small,
        )
    draw.text(
        (40, height - 80),
        "Blue = no subject ID. Gray ridge_vehicle_summary from v0.2 includes subject one-hot and is only an ID-control reference.",
        fill=(120, 0, 0),
        font=font,
    )
    img.save(out_path)


def draw_bad_sample_table(bad_diag: pd.DataFrame, out_path: Path) -> None:
    width, height = 1600, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
        small = font
        title_font = font
    draw.text((35, 25), "Stage 3 v0.3 pre2 session-level bad-sample diagnostics", fill=(0, 0, 0), font=title_font)
    headers = ["rank", "subject", "gt_peak", "v0.2 ridge", "ridge no-id", "kNN no-id", "KRR no-id", "wrong", "under"]
    x_positions = [35, 110, 230, 360, 520, 700, 880, 1080, 1190]
    y = 90
    for x, h in zip(x_positions, headers):
        draw.text((x, y), h, fill=(0, 0, 0), font=font)
    y += 36
    for rank, row in enumerate(bad_diag.head(12).itertuples(index=False), start=1):
        vals = [
            rank,
            getattr(row, "subject", ""),
            f"{getattr(row, 'gt_peak_abs', np.nan):.3f}",
            f"{getattr(row, 'sample_rmse', np.nan):.3f}",
            f"{getattr(row, 'ridge_vehicle_no_subject', np.nan):.3f}",
            f"{getattr(row, 'knn_vehicle_no_subject', np.nan):.3f}",
            f"{getattr(row, 'rbf_krr_vehicle_no_subject', np.nan):.3f}",
            int(getattr(row, "wrong_side", 0)),
            int(getattr(row, "severe_amp_under", 0)),
        ]
        fill = (245, 245, 245) if rank % 2 == 0 else (255, 255, 255)
        draw.rectangle((25, y - 4, width - 25, y + 28), fill=fill)
        for x, val in zip(x_positions, vals):
            draw.text((x, y), str(val), fill=(0, 0, 0), font=small)
        y += 36
    draw.text(
        (35, height - 70),
        "These rows are the highest-RMSE v0.2 ridge samples; compare no-ID pure-vehicle alternatives before adding style/physiology.",
        fill=(120, 0, 0),
        font=font,
    )
    img.save(out_path)


def write_report(comparison: pd.DataFrame, overfit: pd.DataFrame, bucket: pd.DataFrame) -> None:
    pre2 = comparison[
        (comparison["window_config_id"] == "pre2_label2_old_main")
        & (comparison["split_strategy"] == "session_level_split")
        & (comparison["split"] == "test")
    ].sort_values("rmse_steer")
    best_no_id = pre2[pre2["model_name"].astype(str).str.contains("no_subject")].head(1)
    best_no_id_text = best_no_id.to_string(index=False) if not best_no_id.empty else "无"
    report = f"""# 阶段 3 v0.3 诊断：纯车辆无被试 ID 基线、坏样本和小样本过拟合

更新时间：2026-05-12

## 为什么补这一版

阶段 3 v0.2 的 `ridge_vehicle_summary` 特征中包含 `subject` one-hot。被试 ID 不是车辆历史或道路事件信息，因此它不能作为最终“纯车辆基线”，只能作为驾驶员 ID 控制/上限参考。v0.3 重新生成去掉 `subject` 的车辆基线，并补充固定图/坏样本的可解释表和小样本过拟合测试。

## 本次新增模型

- `ridge_vehicle_no_subject`：线性 ridge，去掉 subject one-hot。
- `knn_vehicle_no_subject`：基于车辆历史统计和道路/事件特征的 kNN 平均轨迹。
- `rbf_krr_vehicle_no_subject`：RBF kernel ridge，多输出轨迹回归，alpha/gamma 只用 train/val 选择。

这些模型仍然只用阶段 2 生成的低泄漏道路曲率车辆窗口，不使用生理、脑电、连续风格，也不使用 old v400 或 raw dynamic 作为主锚点。

## pre2 + session-level test 对照

{pre2[['model_name','n_samples','rmse_steer','peak_direction_accuracy','wrong_side_rate','large_response_recall','peak_amp_ratio_pred_over_gt_mean','severe_amp_under_rate','peak_time_mae_s','tail_abs_error_mean','reversal_count_exact_match_rate','difficult_top20_rmse','feature_protocol_note']].to_string(index=False)}

## 当前最好的无被试 ID 纯车辆行

{best_no_id_text}

## 小样本过拟合测试

{overfit.to_string(index=False)}

解释：过拟合测试只在 `pre2_label2_old_main + session_level_split` 上运行，用训练集中峰值最大的若干样本拟合 RBF KRR。若子集训练 RMSE 接近 0，但全训练/测试误差仍高，说明当前模型容量和优化能记住小样本，主要问题更可能是泛化、输入信息不足、事件锚点覆盖或响应多模态，而不是评估脚本完全失效。

## 错误桶

{bucket.to_string(index=False)}

## 当前判断

v0.3 修正后，阶段 3 仍不能进入风格/生理有效性结论。下一步应先确认无被试 ID 纯车辆基线是否足够强，并结合坏样本表检查错误是否集中在大幅响应、错侧、严重幅值不足或多段修正样本。只有强车辆基线稳定后，连续风格和生理增量验证才有公平参照。
"""
    (REPORT_DIR / "stage03_vehicle_diagnostics_v0_3_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    all_per_sample: list[pd.DataFrame] = []
    all_info: list[pd.DataFrame] = []

    for window_id in base_eval.WINDOWS:
        y, y_mask, input_values, label_time, meta = load_window(window_id)
        for split_strategy in base_eval.SPLIT_STRATEGIES:
            per_sample, info, _ = evaluate_model_set(window_id, split_strategy, y, y_mask, input_values, label_time, meta)
            all_per_sample.append(per_sample)
            all_info.append(info)

    stronger_per_sample = pd.concat(all_per_sample, ignore_index=True)
    stronger_metrics = base_eval.aggregate_metrics(stronger_per_sample)
    stronger_info = pd.concat(all_info, ignore_index=True)
    stronger_per_sample.to_csv(TABLE_DIR / "stage03_stronger_vehicle_per_sample_v0_3.csv", index=False, encoding="utf-8-sig")
    stronger_metrics.to_csv(TABLE_DIR / "stage03_stronger_vehicle_metrics_v0_3.csv", index=False, encoding="utf-8-sig")
    stronger_info.to_csv(TABLE_DIR / "stage03_stronger_vehicle_model_info_v0_3.csv", index=False, encoding="utf-8-sig")

    comparison = build_comparison(stronger_metrics)
    comparison.to_csv(TABLE_DIR / "stage03_vehicle_model_comparison_v0_3.csv", index=False, encoding="utf-8-sig")

    overfit = overfit_subset_report()
    overfit.to_csv(TABLE_DIR / "stage03_small_overfit_report_v0_3.csv", index=False, encoding="utf-8-sig")

    fixed_diag, bad_diag, bucket = diagnostic_tables(stronger_per_sample)
    fixed_diag.to_csv(TABLE_DIR / "stage03_fixed_plot_diagnostics_v0_3.csv", index=False, encoding="utf-8-sig")
    bad_diag.to_csv(TABLE_DIR / "stage03_bad_sample_diagnostics_v0_3.csv", index=False, encoding="utf-8-sig")
    bucket.to_csv(TABLE_DIR / "stage03_error_bucket_summary_pre2_session_v0_3.csv", index=False, encoding="utf-8-sig")

    draw_bar_chart(comparison, FIG_DIR / "stage03_pre2_session_model_rmse_comparison_v0_3.png")
    draw_bad_sample_table(bad_diag, FIG_DIR / "stage03_pre2_session_bad_sample_diagnostic_v0_3.png")

    write_report(comparison, overfit, bucket)
    summary = {
        "feature_protocol_correction": "stage03 v0.2 ridge_vehicle_summary included subject one-hot; v0.3 no_subject models are the corrected pure-vehicle baselines",
        "windows": base_eval.WINDOWS,
        "split_strategies": base_eval.SPLIT_STRATEGIES,
        "new_models": ["ridge_vehicle_no_subject", "knn_vehicle_no_subject", "rbf_krr_vehicle_no_subject"],
        "stronger_metric_rows": int(len(stronger_metrics)),
        "stronger_per_sample_rows": int(len(stronger_per_sample)),
        "server_used": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "stage03_vehicle_diagnostics_summary_v0_3.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
