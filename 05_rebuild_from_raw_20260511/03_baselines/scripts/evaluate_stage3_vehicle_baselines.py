# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"F:/data_set_process/data_process/05_rebuild_from_raw_20260511")
DATASET_DIR = ROOT / "03_processed_datasets" / "vehicle_road_curvature_v0_2"
SAMPLES_PATH = DATASET_DIR / "tables" / "selected_samples_vehicle_road_v0_2.csv"
SPLIT_PATH = ROOT / "02_samples" / "tables" / "split_table.csv"
OUT_DIR = ROOT / "03_baselines" / "stage03_vehicle_baselines_v0_2"
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
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return float("nan")
    diff = y_pred[valid] - y_true[valid]
    return float(np.sqrt(np.mean(diff * diff)))


def first_crossing(arr: np.ndarray, thr: float) -> int:
    idx = np.where(np.abs(arr) >= thr)[0]
    return int(idx[0]) if idx.size else -1


def zero_crossing_has(arr: np.ndarray) -> bool:
    valid = arr[np.isfinite(arr)]
    if valid.size < 2:
        return False
    return bool((np.nanmin(valid) < 0.0) and (np.nanmax(valid) > 0.0))


def reversal_count(arr: np.ndarray) -> int:
    valid = arr[np.isfinite(arr)]
    if valid.size < 4:
        return 0
    deriv = np.diff(valid)
    cutoff = max(0.002, float(np.nanpercentile(np.abs(deriv), 70)) * 0.3) if deriv.size else 0.002
    sign = np.sign(deriv)
    sign[np.abs(deriv) < cutoff] = 0
    nonzero = sign[sign != 0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0))


def peak_stats(arr: np.ndarray, time_axis: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return {
            "peak_abs": float("nan"),
            "peak_signed": float("nan"),
            "peak_idx": -1,
            "peak_time_s": float("nan"),
            "peak_direction": 0,
        }
    vals = arr.copy()
    vals[~valid] = 0.0
    idx = int(np.nanargmax(np.abs(vals)))
    signed = float(vals[idx])
    return {
        "peak_abs": abs(signed),
        "peak_signed": signed,
        "peak_idx": idx,
        "peak_time_s": float(time_axis[idx]),
        "peak_direction": 1 if signed >= 0.0 else -1,
    }


def sample_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    time_axis: np.ndarray,
    sample_meta: pd.DataFrame,
    model_name: str,
    split_strategy: str,
    split_name: str,
    window_id: str,
    large_thr: float,
    difficult_thr: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    horizon_s = float(time_axis[-1] - time_axis[0])
    fallback_delay = horizon_s
    for i in range(y_true.shape[0]):
        valid = mask[i] & np.isfinite(y_true[i]) & np.isfinite(y_pred[i])
        if valid.sum() == 0:
            continue
        gt = np.where(valid, y_true[i], np.nan)
        pr = np.where(valid, y_pred[i], np.nan)
        gt_peak = peak_stats(gt, time_axis)
        pr_peak = peak_stats(pr, time_axis)
        onset_thr = max(0.015, 0.2 * max(gt_peak["peak_abs"], 1e-6))
        gt_onset = first_crossing(np.nan_to_num(gt, nan=0.0), onset_thr)
        pr_onset = first_crossing(np.nan_to_num(pr, nan=0.0), onset_thr)
        gt_delay = float(time_axis[gt_onset]) if gt_onset >= 0 else fallback_delay
        pr_delay = float(time_axis[pr_onset]) if pr_onset >= 0 else fallback_delay
        gt_rev = reversal_count(gt)
        pr_rev = reversal_count(pr)
        row = {
            "sample_id": sample_meta.iloc[i]["sample_id"],
            "event_uid": sample_meta.iloc[i]["event_uid"],
            "subject": sample_meta.iloc[i]["subject"],
            "session_stamp": sample_meta.iloc[i]["session_stamp"],
            "window_config_id": window_id,
            "split_strategy": split_strategy,
            "split": split_name,
            "model_name": model_name,
            "sample_rmse": rmse(gt[None, :], pr[None, :], valid[None, :]),
            "gt_peak_abs": gt_peak["peak_abs"],
            "pred_peak_abs": pr_peak["peak_abs"],
            "peak_amp_abs_error": abs(pr_peak["peak_abs"] - gt_peak["peak_abs"]),
            "peak_amp_ratio_pred_over_gt": pr_peak["peak_abs"] / max(gt_peak["peak_abs"], 1e-6),
            "peak_direction_match": int(gt_peak["peak_direction"] == pr_peak["peak_direction"]),
            "wrong_side": int(gt_peak["peak_direction"] != pr_peak["peak_direction"]),
            "peak_time_abs_error_s": abs(pr_peak["peak_time_s"] - gt_peak["peak_time_s"]),
            "onset_delay_abs_error_s": abs(pr_delay - gt_delay),
            "tail_abs_error": abs(float(pr[valid][-1]) - float(gt[valid][-1])),
            "tail_drift_risk": int(abs(float(pr[valid][-1]) - float(gt[valid][-1])) > 0.5 * max(gt_peak["peak_abs"], 1e-6)),
            "zero_crossing_mismatch": int(zero_crossing_has(gt) != zero_crossing_has(pr)),
            "gt_reversal_count": gt_rev,
            "pred_reversal_count": pr_rev,
            "reversal_count_exact": int(gt_rev == pr_rev),
            "gt_multi_segment": int(gt_rev >= 2),
            "pred_multi_segment": int(pr_rev >= 2),
            "is_large_response": int(gt_peak["peak_abs"] >= large_thr),
            "large_response_recalled": int((gt_peak["peak_abs"] >= large_thr) and (pr_peak["peak_abs"] >= 0.5 * gt_peak["peak_abs"])),
            "severe_amp_under": int(pr_peak["peak_abs"] < 0.5 * max(gt_peak["peak_abs"], 1e-6)),
            "is_difficult_peak_top20": int(gt_peak["peak_abs"] >= difficult_thr),
        }
        rows.append(row)
    return rows


def aggregate_metrics(per_sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["window_config_id", "split_strategy", "split", "model_name"]
    for key, grp in per_sample.groupby(group_cols):
        row = dict(zip(group_cols, key))
        row["n_samples"] = int(len(grp))
        row["rmse_steer"] = float(np.sqrt(np.mean(np.square(grp["sample_rmse"]))))
        row["peak_direction_accuracy"] = float(grp["peak_direction_match"].mean())
        row["wrong_side_rate"] = float(grp["wrong_side"].mean())
        row["large_response_recall"] = float(
            grp.loc[grp["is_large_response"] == 1, "large_response_recalled"].mean()
        ) if (grp["is_large_response"] == 1).any() else float("nan")
        row["peak_amp_mae"] = float(grp["peak_amp_abs_error"].mean())
        row["peak_amp_ratio_pred_over_gt_mean"] = float(grp["peak_amp_ratio_pred_over_gt"].replace([np.inf, -np.inf], np.nan).mean())
        row["severe_amp_under_rate"] = float(grp["severe_amp_under"].mean())
        row["peak_time_mae_s"] = float(grp["peak_time_abs_error_s"].mean())
        row["onset_delay_mae_s"] = float(grp["onset_delay_abs_error_s"].mean())
        row["tail_abs_error_mean"] = float(grp["tail_abs_error"].mean())
        row["tail_drift_risk_rate"] = float(grp["tail_drift_risk"].mean())
        row["zero_crossing_mismatch_rate"] = float(grp["zero_crossing_mismatch"].mean())
        row["reversal_count_exact_match_rate"] = float(grp["reversal_count_exact"].mean())
        row["multi_segment_gt_rate"] = float(grp["gt_multi_segment"].mean())
        row["multi_segment_pred_rate"] = float(grp["pred_multi_segment"].mean())
        hard = grp[grp["is_difficult_peak_top20"] == 1]
        row["difficult_top20_rmse"] = float(np.sqrt(np.mean(np.square(hard["sample_rmse"])))) if len(hard) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def make_baseline_predictions(
    y: np.ndarray,
    y_mask: np.ndarray,
    input_values: np.ndarray,
    input_time: np.ndarray,
    label_time: np.ndarray,
    meta: pd.DataFrame,
    split_col: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n, t_len = y.shape
    train_idx = np.where(meta[split_col].to_numpy() == "train")[0]
    val_idx = np.where(meta[split_col].to_numpy() == "val")[0]
    if train_idx.size == 0:
        train_idx = np.arange(n)
    preds: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}
    preds["zero_response"] = np.zeros_like(y, dtype=np.float32)
    preds["hold_current"] = np.zeros_like(y, dtype=np.float32)

    # Linear extrapolation from the last 250 ms of steering history.
    steer_hist = input_values[:, :, 0].astype(np.float64)
    hist_rel = steer_hist - steer_hist[:, [-1]]
    recent = input_time >= (float(input_time[-1]) - 0.25)
    if recent.sum() < 2:
        recent = np.ones_like(input_time, dtype=bool)
    x_recent = input_time[recent].astype(np.float64)
    x_center = x_recent - x_recent.mean()
    denom = float(np.sum(x_center * x_center)) or 1.0
    slopes = np.nansum((hist_rel[:, recent] - np.nanmean(hist_rel[:, recent], axis=1, keepdims=True)) * x_center[None, :], axis=1) / denom
    trend = slopes[:, None] * label_time[None, :].astype(np.float64)
    train_peak = np.nanpercentile(np.abs(y[train_idx]), 95) if train_idx.size else np.nanpercentile(np.abs(y), 95)
    clip = max(float(train_peak) * 1.5, 1.0)
    preds["history_trend_250ms"] = np.clip(trend, -clip, clip).astype(np.float32)

    train_mean = np.nanmean(np.where(y_mask[train_idx], y[train_idx], np.nan), axis=0)
    train_mean = np.nan_to_num(train_mean, nan=0.0).astype(np.float32)
    preds["train_mean_all"] = np.tile(train_mean[None, :], (n, 1))

    group_pred = np.zeros_like(y, dtype=np.float32)
    for group_key, group_df in meta.groupby(["event_type", "event_level"], dropna=False):
        idx = group_df.index.to_numpy()
        train_group = [i for i in idx if i in set(train_idx.tolist())]
        if len(train_group) < 3:
            group_pred[idx] = preds["train_mean_all"][idx]
            continue
        mean = np.nanmean(np.where(y_mask[train_group], y[train_group], np.nan), axis=0)
        group_pred[idx] = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    preds["train_mean_by_event_type"] = group_pred

    ridge_pred, ridge_info = ridge_vehicle_model(y, y_mask, input_values, meta, split_col, train_idx, val_idx)
    preds["ridge_vehicle_summary"] = ridge_pred
    info["ridge_vehicle_summary"] = ridge_info
    return preds, info


def extract_vehicle_features(input_values: np.ndarray, input_mask: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    names: list[str] = []
    x = input_values.astype(np.float64)
    x = np.where(input_mask, x, np.nan)
    for j in range(x.shape[2]):
        arr = x[:, :, j]
        for stat_name, vals in [
            ("last", arr[:, -1]),
            ("mean", np.nanmean(arr, axis=1)),
            ("std", np.nanstd(arr, axis=1)),
            ("min", np.nanmin(arr, axis=1)),
            ("max", np.nanmax(arr, axis=1)),
            ("delta", arr[:, -1] - arr[:, 0]),
        ]:
            features.append(vals)
            names.append(f"f{j}_{stat_name}")
    for col in ["anchor_time_rel_s", "curvature_anchor"]:
        if col in meta.columns:
            features.append(pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=np.float64))
            names.append(col)
    for col in ["event_type", "event_level", "subject"]:
        if col in meta.columns:
            values = meta[col].astype(str).fillna("NA")
            for val in sorted(values.unique()):
                features.append((values == val).to_numpy(dtype=np.float64))
                names.append(f"{col}={val}")
    X = np.vstack(features).T if features else np.zeros((len(meta), 0), dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def ridge_vehicle_model(
    y: np.ndarray,
    y_mask: np.ndarray,
    input_values: np.ndarray,
    meta: pd.DataFrame,
    split_col: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    input_mask = np.isfinite(input_values)
    X, feature_names = extract_vehicle_features(input_values, input_mask, meta)
    if train_idx.size < 5:
        return np.zeros_like(y, dtype=np.float32), {"status": "no_train_samples", "feature_count": len(feature_names)}
    mu = X[train_idx].mean(axis=0, keepdims=True)
    sigma = X[train_idx].std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    Xs = (X - mu) / sigma
    Xd = np.c_[np.ones((Xs.shape[0], 1)), Xs]
    Y = np.nan_to_num(np.where(y_mask, y, np.nan), nan=0.0).astype(np.float64)

    def fit_predict(alpha: float) -> np.ndarray:
        Xt = Xd[train_idx]
        Yt = Y[train_idx]
        reg = np.eye(Xt.shape[1], dtype=np.float64) * float(alpha)
        reg[0, 0] = 0.0
        coef = np.linalg.solve(Xt.T @ Xt + reg, Xt.T @ Yt)
        return (Xd @ coef).astype(np.float32)

    best_alpha = RIDGE_ALPHAS[0]
    best_score = float("inf")
    if val_idx.size == 0:
        val_idx = train_idx
    best_pred = None
    for alpha in RIDGE_ALPHAS:
        pred = fit_predict(alpha)
        score = rmse(y[val_idx], pred[val_idx], y_mask[val_idx])
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_pred = pred
    assert best_pred is not None
    train_score = rmse(y[train_idx], best_pred[train_idx], y_mask[train_idx])
    return best_pred, {
        "status": "ok",
        "selected_alpha": float(best_alpha),
        "val_rmse_for_alpha": float(best_score),
        "train_rmse_selected_alpha": float(train_score),
        "feature_count": len(feature_names),
    }


def draw_prediction_grid(
    path: Path,
    time_axis: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    meta: pd.DataFrame,
    sample_indices: list[int],
    title: str,
) -> None:
    width, height = 1400, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((30, 20), title, fill=(20, 20, 20), font=font)
    colors = {
        "gt": (0, 0, 0),
        "zero_response": (130, 130, 130),
        "history_trend_250ms": (204, 95, 39),
        "train_mean_by_event_type": (57, 106, 177),
        "ridge_vehicle_summary": (62, 150, 81),
    }
    plot_names = ["zero_response", "history_trend_250ms", "train_mean_by_event_type", "ridge_vehicle_summary"]
    cols, rows = 3, 4
    panel_w, panel_h = 420, 190
    left0, top0 = 45, 75
    for k, idx in enumerate(sample_indices[: cols * rows]):
        c = k % cols
        r = k // cols
        left = left0 + c * (panel_w + 25)
        top = top0 + r * (panel_h + 35)
        gt = y_true[idx]
        series = [gt] + [predictions[name][idx] for name in plot_names if name in predictions]
        y_min = min(float(np.nanpercentile(s, 2)) for s in series)
        y_max = max(float(np.nanpercentile(s, 98)) for s in series)
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        draw.rectangle((left, top, left + panel_w, top + panel_h), outline=(0, 0, 0))
        draw.line((left, top + panel_h // 2, left + panel_w, top + panel_h // 2), fill=(220, 220, 220))
        for name, arr in [("gt", gt)] + [(n, predictions[n][idx]) for n in plot_names if n in predictions]:
            pts = []
            for tx, val in zip(time_axis, arr):
                x = left + int((float(tx) - float(time_axis[0])) / (float(time_axis[-1] - time_axis[0])) * panel_w)
                y = top + panel_h - int((float(val) - y_min) / (y_max - y_min) * panel_h)
                pts.append((x, y))
            if len(pts) > 1:
                draw.line(pts, fill=colors.get(name, (120, 120, 120)), width=2)
        text = f"{meta.iloc[idx]['subject']} {meta.iloc[idx]['event_level']} peak={np.nanmax(np.abs(gt)):.2f}"
        draw.text((left, top + panel_h + 4), text[:58], fill=(0, 0, 0), font=font)
    legend_x, legend_y = 50, height - 45
    for i, (name, color) in enumerate(colors.items()):
        x = legend_x + i * 245
        draw.rectangle((x, legend_y, x + 18, legend_y + 12), fill=color)
        draw.text((x + 24, legend_y), name, fill=(0, 0, 0), font=font)
    img.save(path)


def main() -> None:
    ensure_dirs()
    all_samples = pd.read_csv(SAMPLES_PATH)
    split_table = pd.read_csv(SPLIT_PATH)
    split_cols = ["event_uid"] + SPLIT_STRATEGIES
    all_samples = all_samples.merge(split_table[split_cols], on="event_uid", how="left")
    all_metrics: list[pd.DataFrame] = []
    model_info_rows: list[dict[str, Any]] = []
    fixed_plot_records: list[dict[str, Any]] = []

    for window_id in WINDOWS:
        npz_path = DATASET_DIR / "arrays" / f"{window_id}.npz"
        index_path = DATASET_DIR / "tables" / f"sample_index_{window_id}.csv"
        z = np.load(npz_path, allow_pickle=True)
        y = z["label_steer_delta"].astype(np.float32)
        y_mask = z["label_valid_mask"].astype(bool)
        input_values = z["input_values"].astype(np.float32)
        label_time = z["label_time_rel_s"].astype(np.float32)
        input_time = z["input_time_rel_s"].astype(np.float32)
        idx_df = pd.read_csv(index_path)
        meta = idx_df.merge(all_samples, on=["sample_id", "event_uid", "subject", "session_stamp", "anchor_time_rel_s", "anchor_time_abs_s", "window_config_id"], how="left")
        gt_peak = np.nanmax(np.abs(np.where(y_mask, y, np.nan)), axis=1)

        for split_strategy in SPLIT_STRATEGIES:
            train_idx = np.where(meta[split_strategy].to_numpy() == "train")[0]
            large_thr = float(np.nanpercentile(gt_peak[train_idx], 75)) if train_idx.size else float(np.nanpercentile(gt_peak, 75))
            difficult_thr = float(np.nanpercentile(gt_peak[train_idx], 80)) if train_idx.size else float(np.nanpercentile(gt_peak, 80))
            preds, info = make_baseline_predictions(y, y_mask, input_values, input_time, label_time, meta, split_strategy)
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
                    rows = sample_metric_rows(
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
                    draw_prediction_grid(
                        FIG_DIR / "stage03_fixed_predictions_pre2_session_test.png",
                        label_time,
                        y,
                        preds,
                        meta,
                        fixed,
                        "Stage 3 fixed predictions: pre2, session-level test",
                    )
                    ridge = preds["ridge_vehicle_summary"]
                    sample_rmse = np.sqrt(np.nanmean(np.square(ridge[test_idx] - y[test_idx]), axis=1))
                    bad = test_idx[np.argsort(-sample_rmse)[:12]].tolist()
                    draw_prediction_grid(
                        FIG_DIR / "stage03_bad_samples_pre2_session_test_ridge.png",
                        label_time,
                        y,
                        preds,
                        meta,
                        bad,
                        "Stage 3 bad samples: pre2 session-level test ridge",
                    )

    per_sample = pd.concat(all_metrics, ignore_index=True)
    metrics = aggregate_metrics(per_sample)
    per_sample.to_csv(TABLE_DIR / "stage03_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(TABLE_DIR / "stage03_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(model_info_rows).to_csv(TABLE_DIR / "stage03_ridge_model_info.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fixed_plot_records).to_csv(TABLE_DIR / "stage03_fixed_plot_sample_set.csv", index=False, encoding="utf-8-sig")

    test_metrics = metrics[metrics["split"] == "test"].copy()
    best_rows = test_metrics.sort_values(["window_config_id", "split_strategy", "rmse_steer"]).groupby(["window_config_id", "split_strategy"]).head(1)
    best_rows.to_csv(TABLE_DIR / "stage03_best_test_by_window_split.csv", index=False, encoding="utf-8-sig")
    summary = {
        "windows": WINDOWS,
        "split_strategies": SPLIT_STRATEGIES,
        "models": sorted(per_sample["model_name"].unique().tolist()),
        "metric_rows": int(len(metrics)),
        "per_sample_rows": int(len(per_sample)),
        "best_test": best_rows[["window_config_id", "split_strategy", "model_name", "rmse_steer", "peak_direction_accuracy", "wrong_side_rate", "severe_amp_under_rate"]].to_dict(orient="records"),
        "server_used": False,
        "raw_files_modified": False,
    }
    (LOG_DIR / "stage03_baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(metrics, best_rows, model_info_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_reports(metrics: pd.DataFrame, best_rows: pd.DataFrame, model_info_rows: list[dict[str, Any]]) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    pre2_session = test[(test["window_config_id"] == "pre2_label2_old_main") & (test["split_strategy"] == "session_level_split")]
    pre2_table = pre2_session[
        [
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
    ].sort_values("rmse_steer")
    ridge_info = pd.DataFrame(model_info_rows)
    report = f"""# 阶段 3 基线总结：低泄漏道路曲率车辆窗口 v0.2

更新时间：2026-05-12

## 范围

本阶段只使用阶段 2 生成的低泄漏 `raw_road_curvature_onset` 车辆窗口，不使用生理、脑电、连续风格，也不使用 old v400 或 raw dynamic 锚点作为主线。

## 已完成

1. 无学习基线：`zero_response`、`hold_current`、`history_trend_250ms`、`train_mean_all`、`train_mean_by_event_type`。
2. 纯车辆强基线：`ridge_vehicle_summary`，只用车辆历史统计特征和事件元信息，标准化和 alpha 选择均只在 train/val 内完成。
3. 三个窗口、三种 split 均已评估：random event、session-level、subject-level。
4. 指标覆盖整体 RMSE、方向、错侧、大幅响应召回、峰值幅值、峰值时间、尾段、零线穿越、反向修正、多段修正和困难样本。
5. 固定预测图和坏样本图已经生成。

## pre2 + session-level test 关键表

{pre2_table.to_string(index=False)}

## 各窗口/切分测试集最优行

{best_rows[['window_config_id','split_strategy','model_name','rmse_steer','peak_direction_accuracy','wrong_side_rate','severe_amp_under_rate','difficult_top20_rmse']].to_string(index=False)}

## Ridge 训练信息

{ridge_info.to_string(index=False)}

## 当前判断

这一步已经建立了阶段 3 的无学习基线和一个纯车辆强基线。由于当前只覆盖道路曲率候选 359 个事件，结论只能说“低泄漏道路曲率子集上的车辆基线表现”，不能外推到全部旧 v400 事件，也不能用于判断连续风格或生理是否有效。下一步应检查固定图和坏样本，确认指标能解释可视化错误后，再决定是否扩展道路锚点或进入更强车辆模型。
"""
    (REPORT_DIR / "stage03_vehicle_baseline_summary_cn.md").write_text(report, encoding="utf-8")

    user = f"""# 阶段 3 用户查看版总结：无学习基线与纯车辆基线

更新时间：2026-05-12

## 这个阶段为什么做

在讨论连续风格、生理和脑电之前，必须先知道只靠车辆历史和道路事件信息能做到什么程度。否则后面即使模型变好，也说不清是生理有效，还是车辆信息本来就够用。

## 这个阶段检查了什么

- 用低泄漏道路曲率候选样本做基线，不使用旧 v400 响应锚点做主结论。
- 做了零响应、保持当前、历史趋势外推、训练集平均轨迹和同类事件平均轨迹。
- 做了一个纯车辆 ridge 基线，只使用车辆历史窗口统计特征。
- 在随机切分、按记录切分、按被试切分上都算了指标。
- 生成了固定预测图和坏样本图，不只看平均 RMSE。

## 目前发现了什么

pre2 窗口、session-level test 的结果如下：

{pre2_table.to_string(index=False)}

## 哪些结果可信

- 这些结果只依赖原始车辆数据派生出的低泄漏道路曲率候选。
- 车辆窗口处理没有改原始 CSV，没有用生理/脑电，没有用测试集统计做标准化。
- 指标和固定图可以作为阶段 3 继续调车辆模型的起点。

## 哪些结果还不能下结论

- 不能说风格有效或生理有效。
- 不能说全部事件都已经覆盖，因为当前主线只覆盖 359 个道路曲率候选。
- 不能把 old v400 和 raw dynamic 的结果混进无泄漏主结论。
- 还需要检查固定预测图和坏样本图，确认指标是否能解释具体物理错误。

## 下一阶段是否可以继续

可以继续阶段 3，但不是进入风格/生理阶段。下一步应先看纯车辆基线的固定图和坏样本，必要时扩展低泄漏道路锚点或改进车辆基线。

## 推荐优先查看

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_baseline_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_baseline_metrics.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_best_test_by_window_split.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_fixed_predictions_pre2_session_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_bad_samples_pre2_session_test_ridge.png`
"""
    (REPORT_DIR / "stage03_user_summary_cn.md").write_text(user, encoding="utf-8")


if __name__ == "__main__":
    main()
