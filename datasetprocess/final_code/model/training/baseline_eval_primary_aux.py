from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from future_steer_speed_subjectsplit_masked import FS, FUTURE_LEN, PRIMARY_HORIZON_LEN


EVAL_SCOPES = [
    ("overall_primary", 0, PRIMARY_HORIZON_LEN),
    ("0_0.5s", 0, 100),
    ("0.5_1.0s", 100, 200),
    ("1.0_1.5s", 200, 300),
    ("1.5_2.0s_aux", 300, FUTURE_LEN),
]
LATE_START = int(1.0 * FS)
LATE_END = PRIMARY_HORIZON_LEN
START_REF_IDX = 1
SMOOTH_WIN = 3
END_AVG_WIN = 20
DIR_EPS_RATIO = 0.002
AMP_EPS_RATIO = 0.15
RECENTER_RATIO = 0.75
OVERSHOOT_RATIO = 0.2

TRAJ_BLOCK = 10
TAIL_START_IDX = 300
PRE_TAIL_START_IDX = 200
TAIL_START_SEC = 1.5
TAIL_DIRECTION_EPS = 2e-3
GLOBAL_DIRECTION_EPS = 3e-3

TRAJECTORY_SUMMARY_METRICS = [
    "rmse_2s_abs_steer",
    "rmse_pre_tail_abs_steer",
    "rmse_tail_abs_steer",
    "tail_pre_gap_abs_steer",
    "tail_pre_ratio_abs_steer",
    "late_mean_abs_err_steer",
    "trend_corr",
    "tail_trend_corr",
    "shape_corr",
    "tail_shape_corr",
    "direction_match",
    "tail_direction_match",
    "tail_slope_abs_err",
    "boundary_slope_abs_err",
    "boundary_shift_abs_err",
    "turning_count_abs_err",
    "turning_has_reversal_match",
    "first_reversal_time_abs_err_s",
    "peak_time_abs_err_s",
    "peak_abs_amp_err",
    "range_abs_err",
    "extrema_count_abs_err",
]

TRAJECTORY_SELECTION_SCALES = {
    "rmse_tail_abs_steer": 0.40,
    "tail_pre_ratio_abs_steer": 1.25,
    "turning_count_abs_err": 2.0,
    "peak_time_abs_err_s": 0.60,
    "boundary_shift_abs_err": 0.80,
}


STRUCTURE_HEAVY_MORPH = {"reverse_correction", "multi_correction"}


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or win <= 1:
        return x.copy()
    win = int(min(max(win, 1), x.size))
    pad = win // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x_pad, kernel, mode="valid")[: x.size]


def compress_signs(diff_arr: np.ndarray, eps: float) -> list[int]:
    signs: list[int] = []
    for val in diff_arr:
        if abs(float(val)) <= float(eps):
            continue
        sign = 1 if val > 0 else -1
        if not signs or signs[-1] != sign:
            signs.append(sign)
    return signs


def classify_eval_morphology(y_steer: np.ndarray, valid_future_len: int) -> str:
    valid_future_len = int(valid_future_len)
    if valid_future_len <= 1:
        return "single_lobe"
    y = np.asarray(y_steer[:valid_future_len], dtype=np.float64)
    if y.size <= 1:
        return "single_lobe"
    late = y[min(LATE_START, y.size - 1) : min(LATE_END, y.size)]
    late_s = float(np.mean(late)) if late.size else float(y[-1])
    start_ref = float(y[min(START_REF_IDX, y.size - 1)])
    smooth = moving_average(y, SMOOTH_WIN)
    end_ref = float(np.mean(smooth[-min(END_AVG_WIN, smooth.size) :]))
    amp_ref = float(np.max(np.abs(smooth))) if smooth.size else float(np.max(np.abs(y)))
    diff_eps = max(DIR_EPS_RATIO, DIR_EPS_RATIO * max(1.0, amp_ref))
    amp_eps = max(1e-6, AMP_EPS_RATIO * max(amp_ref, 1e-6))
    diff_seq = np.diff(smooth)
    sign_seq = compress_signs(diff_seq, diff_eps)
    extrema_count = max(len(sign_seq) - 1, 0)
    if extrema_count >= 2:
        return "multi_correction"

    start_sign = 1.0 if start_ref >= 0 else -1.0
    end_sign = 1.0 if end_ref >= 0 else -1.0
    signed = smooth * start_sign
    cross_zero = bool(np.min(signed) < -amp_eps)
    toward_zero = bool(abs(end_ref) <= RECENTER_RATIO * max(abs(start_ref), amp_eps))
    overshoot = bool(np.min(signed) < -OVERSHOOT_RATIO * max(abs(start_ref), amp_eps))

    if cross_zero:
        return "reverse_correction" if overshoot or (start_sign != end_sign) else "recentering"
    if toward_zero and abs(late_s) <= RECENTER_RATIO * max(abs(start_ref), amp_eps):
        return "recentering"
    return "single_lobe"


def _metric_block_point(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict[str, dict[str, float]]:
    support_points = float(mask.sum())
    support_samples = float((mask.sum(axis=1) > 0).sum())
    out: dict[str, dict[str, float]] = {
        "support": {"support_samples": support_samples, "support_points": support_points}
    }
    for name, scale, col_idx in (("steer", 1.0, 0), ("speed_ms", 1.0, 1), ("speed_kmh", 3.6, 1)):
        err = (pred[:, :, col_idx] - true[:, :, col_idx]) * scale
        if support_points <= 0:
            out[name] = {"rmse": float("nan"), "mae": float("nan")}
            continue
        rmse = float(np.sqrt(np.sum((err**2) * mask) / support_points))
        mae = float(np.sum(np.abs(err) * mask) / support_points)
        out[name] = {"rmse": rmse, "mae": mae}
    return out


def _metric_block_sample(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict[str, dict[str, float]]:
    valid = mask.sum(axis=1) > 0
    support_samples = float(valid.sum())
    support_points = float(mask.sum())
    out: dict[str, dict[str, float]] = {
        "support": {"support_samples": support_samples, "support_points": support_points}
    }
    for name, scale, col_idx in (("steer", 1.0, 0), ("speed_ms", 1.0, 1), ("speed_kmh", 3.6, 1)):
        err = (pred[:, :, col_idx] - true[:, :, col_idx]) * scale
        rmse_vals = []
        mae_vals = []
        for idx in range(pred.shape[0]):
            count = mask[idx].sum()
            if count <= 0:
                continue
            err_i = err[idx][mask[idx] > 0]
            rmse_vals.append(float(np.sqrt(np.mean(err_i**2))))
            mae_vals.append(float(np.mean(np.abs(err_i))))
        out[name] = {
            "rmse": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
            "mae": float(np.mean(mae_vals)) if mae_vals else float("nan"),
        }
    return out


def compute_weighted_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    out = {"point_weighted": {}, "sample_weighted": {}}
    for scope_name, start, end in EVAL_SCOPES:
        pred_part = pred[:, start:end, :]
        true_part = true[:, start:end, :]
        mask_part = mask[:, start:end]
        out["point_weighted"][scope_name] = _metric_block_point(pred_part, true_part, mask_part)
        out["sample_weighted"][scope_name] = _metric_block_sample(pred_part, true_part, mask_part)
    return out


def flatten_weighted_metrics(metrics: dict[str, Any], split_name: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weighting_name, scopes in metrics.items():
        for scope_name, scope_metrics in scopes.items():
            for target_name, metric_block in scope_metrics.items():
                for metric_name, value in metric_block.items():
                    rows.append(
                        {
                            "seed": int(seed),
                            "split": split_name,
                            "weighting": weighting_name,
                            "scope": scope_name,
                            "target": target_name,
                            "metric": metric_name,
                            "value": float(value),
                        }
                    )
    return rows


def compute_group_metric_rows(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    meta_df: pd.DataFrame,
    group_col: str,
    split_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    if group_col not in meta_df.columns:
        return []
    work = meta_df.reset_index(drop=True).copy()
    work[group_col] = work[group_col].fillna("unknown").replace("", "unknown")
    rows: list[dict[str, Any]] = []
    for label, part in work.groupby(group_col):
        idx = part.index.to_numpy(dtype=np.int64)
        weighted = compute_weighted_metrics(pred[idx], true[idx], mask[idx])
        base_rows = flatten_weighted_metrics(weighted, split_name=split_name, seed=seed)
        for row in base_rows:
            row[group_col] = str(label)
            row["sample_count"] = int(len(idx))
            rows.append(row)
    return rows


def compute_morphology_labels(meta_df: pd.DataFrame, true: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = meta_df.reset_index(drop=True)
    valid_lens = mask.sum(axis=1).astype(np.int64)
    for idx, _row in work.iterrows():
        label = classify_eval_morphology(true[idx, :, 0], int(valid_lens[idx]))
        rows.append({"eval_morphology_label": label})
    return pd.DataFrame(rows)


def compute_morphology_metric_rows(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    meta_df: pd.DataFrame,
    split_name: str,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    label_df = compute_morphology_labels(meta_df, true, mask)
    rows: list[dict[str, Any]] = []
    for label, part in label_df.groupby("eval_morphology_label"):
        idx = part.index.to_numpy(dtype=np.int64)
        weighted = compute_weighted_metrics(pred[idx], true[idx], mask[idx])
        base_rows = flatten_weighted_metrics(weighted, split_name=split_name, seed=seed)
        for row in base_rows:
            row["eval_morphology_label"] = str(label)
            row["sample_count"] = int(len(idx))
            rows.append(row)
    return label_df, rows


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if float(np.std(a)) < 1e-10 or float(np.std(b)) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sign_eps(x: float, eps: float = 1e-3) -> int:
    if abs(x) < eps:
        return 0
    return 1 if x > 0 else -1


def compress_reversals(deriv: np.ndarray, threshold: float) -> tuple[int, float]:
    sign = np.sign(deriv).astype(np.int32)
    sign[np.abs(deriv) < threshold] = 0
    idx = np.where(sign != 0)[0]
    if idx.size < 2:
        return 0, float("nan")
    vals = sign[idx]
    changes = np.where(vals[1:] * vals[:-1] < 0)[0]
    if changes.size == 0:
        return 0, float("nan")
    first_original_idx = int(idx[changes[0] + 1])
    return int(changes.size), float(first_original_idx)


def local_extrema_count(arr: np.ndarray, threshold: float = 1e-4) -> int:
    d = np.diff(arr)
    sign = np.sign(d).astype(np.int32)
    sign[np.abs(d) < threshold] = 0
    idx = np.where(sign != 0)[0]
    if idx.size < 2:
        return 0
    vals = sign[idx]
    return int(np.sum(vals[1:] * vals[:-1] < 0))


def block_downsample(arr: np.ndarray, block: int = TRAJ_BLOCK) -> np.ndarray:
    n = (arr.size // block) * block
    if n < block:
        return arr.copy()
    return arr[:n].reshape(-1, block).mean(axis=1)


def compute_trajectory_sample_metrics(
    meta_df: pd.DataFrame,
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    ctx_raw: np.ndarray,
    split_name: str,
    seed: int,
) -> pd.DataFrame:
    work = meta_df.reset_index(drop=True).copy()
    rows: list[dict[str, object]] = []
    ds_dt = TRAJ_BLOCK / FS
    tail_start_ds = int(round(TAIL_START_SEC / ds_dt))
    steer_anchor = np.asarray(ctx_raw[:, 0], dtype=np.float32)
    abs_true = true[:, :, 0] + steer_anchor.reshape(-1, 1)
    abs_pred = pred[:, :, 0] + steer_anchor.reshape(-1, 1)

    for i in range(len(work)):
        valid_len = int(np.sum(mask[i] > 0))
        true_abs = abs_true[i, :valid_len]
        pred_abs = abs_pred[i, :valid_len]
        if true_abs.size < 20:
            continue
        err = pred_abs - true_abs

        rmse_2s = float(np.sqrt(np.mean(err**2)))
        rmse_pre_tail = float(np.sqrt(np.mean(err[PRE_TAIL_START_IDX:TAIL_START_IDX] ** 2))) if valid_len > TAIL_START_IDX else float("nan")
        rmse_tail = float(np.sqrt(np.mean(err[TAIL_START_IDX:valid_len] ** 2))) if valid_len > TAIL_START_IDX else float("nan")
        tail_pre_gap = rmse_tail - rmse_pre_tail if np.isfinite(rmse_tail) and np.isfinite(rmse_pre_tail) else float("nan")
        tail_pre_ratio = rmse_tail / (rmse_pre_tail + 1e-8) if np.isfinite(rmse_tail) and np.isfinite(rmse_pre_tail) else float("nan")

        late_mean_abs_err = float(np.mean(np.abs(err[TAIL_START_IDX:valid_len]))) if valid_len > TAIL_START_IDX else float("nan")
        tail_slope_true = float(np.mean(np.diff(true_abs[TAIL_START_IDX:valid_len])) * FS) if valid_len > TAIL_START_IDX + 3 else float("nan")
        tail_slope_pred = float(np.mean(np.diff(pred_abs[TAIL_START_IDX:valid_len])) * FS) if valid_len > TAIL_START_IDX + 3 else float("nan")
        tail_slope_abs_err = abs(tail_slope_pred - tail_slope_true) if np.isfinite(tail_slope_true) and np.isfinite(tail_slope_pred) else float("nan")

        true_ds = block_downsample(true_abs, block=TRAJ_BLOCK)
        pred_ds = block_downsample(pred_abs, block=TRAJ_BLOCK)
        d_true = np.diff(true_ds) / ds_dt
        d_pred = np.diff(pred_ds) / ds_dt

        trend_corr = safe_corr(d_true, d_pred)
        shape_corr = safe_corr(true_ds, pred_ds)
        tail_trend_corr = safe_corr(d_true[tail_start_ds:], d_pred[tail_start_ds:]) if d_true.size > tail_start_ds + 2 else float("nan")
        tail_shape_corr = safe_corr(true_ds[tail_start_ds:], pred_ds[tail_start_ds:]) if true_ds.size > tail_start_ds + 2 else float("nan")

        global_dir_true = sign_eps(float(true_ds[-1] - true_ds[0]), eps=GLOBAL_DIRECTION_EPS)
        global_dir_pred = sign_eps(float(pred_ds[-1] - pred_ds[0]), eps=GLOBAL_DIRECTION_EPS)
        tail_dir_true = sign_eps(float(true_ds[-1] - true_ds[tail_start_ds]), eps=TAIL_DIRECTION_EPS) if true_ds.size > tail_start_ds else 0
        tail_dir_pred = sign_eps(float(pred_ds[-1] - pred_ds[tail_start_ds]), eps=TAIL_DIRECTION_EPS) if pred_ds.size > tail_start_ds else 0

        direction_match = 1 if global_dir_true == global_dir_pred else 0
        tail_direction_match = 1 if tail_dir_true == tail_dir_pred else 0

        boundary_idx = tail_start_ds - 1
        if d_true.size > boundary_idx + 1 and d_pred.size > boundary_idx + 1:
            boundary_slope_abs_err = abs(float(d_pred[boundary_idx] - d_true[boundary_idx]))
            pre_slice_true = d_true[max(0, boundary_idx - 4) : boundary_idx + 1]
            post_slice_true = d_true[boundary_idx + 1 : min(d_true.size, boundary_idx + 6)]
            pre_slice_pred = d_pred[max(0, boundary_idx - 4) : boundary_idx + 1]
            post_slice_pred = d_pred[boundary_idx + 1 : min(d_pred.size, boundary_idx + 6)]
            if pre_slice_true.size > 0 and post_slice_true.size > 0 and pre_slice_pred.size > 0 and post_slice_pred.size > 0:
                shift_true = float(np.mean(post_slice_true) - np.mean(pre_slice_true))
                shift_pred = float(np.mean(post_slice_pred) - np.mean(pre_slice_pred))
                boundary_shift_abs_err = abs(shift_pred - shift_true)
            else:
                boundary_shift_abs_err = float("nan")
        else:
            boundary_slope_abs_err = float("nan")
            boundary_shift_abs_err = float("nan")

        turn_thr = max(0.02, 0.5 * float(np.percentile(np.abs(d_true), 30)))
        true_turn_cnt, true_first_rev = compress_reversals(d_true, threshold=turn_thr)
        pred_turn_cnt, pred_first_rev = compress_reversals(d_pred, threshold=turn_thr)
        turning_count_abs_err = abs(pred_turn_cnt - true_turn_cnt)
        turning_has_reversal_match = 1 if ((true_turn_cnt > 0) == (pred_turn_cnt > 0)) else 0
        if np.isfinite(true_first_rev) and np.isfinite(pred_first_rev):
            first_reversal_time_abs_err_s = abs(pred_first_rev - true_first_rev) * ds_dt
        else:
            first_reversal_time_abs_err_s = float("nan")

        peak_true = int(np.argmax(np.abs(true_ds)))
        peak_pred = int(np.argmax(np.abs(pred_ds)))
        peak_time_abs_err_s = abs(peak_pred - peak_true) * ds_dt
        peak_abs_amp_err = abs(float(np.max(np.abs(pred_ds)) - np.max(np.abs(true_ds))))
        range_abs_err = abs(float((np.max(pred_ds) - np.min(pred_ds)) - (np.max(true_ds) - np.min(true_ds))))
        extrema_true = local_extrema_count(true_ds)
        extrema_pred = local_extrema_count(pred_ds)
        extrema_count_abs_err = abs(extrema_pred - extrema_true)

        meta_row = work.iloc[i]
        rows.append(
            {
                "split": split_name,
                "seed": int(seed),
                "local_idx": int(i),
                "subj": str(meta_row.get("subj", "unknown")),
                "sample_key": str(meta_row.get("sample_key", i)),
                "phase_type": str(meta_row.get("phase_type", "unknown")),
                "road_type_anchor": str(meta_row.get("road_type_anchor", "unknown")),
                "mechanism_tag": str(meta_row.get("mechanism_tag", "unknown")),
                "is_curve": int(meta_row.get("is_curve", 0)),
                "structure_slice": str(meta_row.get("structure_slice", "unknown")),
                "structure_heavy": int(meta_row.get("structure_heavy", 0)),
                "valid_future_len": valid_len,
                "eval_morphology_label": classify_eval_morphology(true[i, :, 0], valid_len),
                "rmse_2s_abs_steer": rmse_2s,
                "rmse_pre_tail_abs_steer": rmse_pre_tail,
                "rmse_tail_abs_steer": rmse_tail,
                "tail_pre_gap_abs_steer": tail_pre_gap,
                "tail_pre_ratio_abs_steer": tail_pre_ratio,
                "late_mean_abs_err_steer": late_mean_abs_err,
                "trend_corr": trend_corr,
                "tail_trend_corr": tail_trend_corr,
                "shape_corr": shape_corr,
                "tail_shape_corr": tail_shape_corr,
                "direction_match": direction_match,
                "tail_direction_match": tail_direction_match,
                "tail_slope_abs_err": tail_slope_abs_err,
                "boundary_slope_abs_err": boundary_slope_abs_err,
                "boundary_shift_abs_err": boundary_shift_abs_err,
                "turning_count_abs_err": turning_count_abs_err,
                "turning_has_reversal_match": turning_has_reversal_match,
                "first_reversal_time_abs_err_s": first_reversal_time_abs_err_s,
                "peak_time_abs_err_s": peak_time_abs_err_s,
                "peak_abs_amp_err": peak_abs_amp_err,
                "range_abs_err": range_abs_err,
                "extrema_count_abs_err": extrema_count_abs_err,
            }
        )
    return pd.DataFrame(rows)


def _mean_metric(sample_df: pd.DataFrame, column: str, default: float) -> float:
    if column not in sample_df.columns or sample_df.empty:
        return float(default)
    value = float(sample_df[column].mean())
    if not np.isfinite(value):
        return float(default)
    return value


def summarize_trajectory_subset(
    sample_df: pd.DataFrame,
    split_name: str,
    seed: int,
    subset_family: str,
    subset_name: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split_name,
        "seed": int(seed),
        "subset_family": subset_family,
        "subset_name": subset_name,
        "sample_count": int(len(sample_df)),
    }
    for metric in TRAJECTORY_SUMMARY_METRICS:
        row[metric] = float(sample_df[metric].mean()) if metric in sample_df.columns and not sample_df.empty else float("nan")
    return row


def build_trajectory_subset_rows(sample_df: pd.DataFrame, split_name: str, seed: int) -> list[dict[str, Any]]:
    if sample_df.empty:
        return []
    rows = [summarize_trajectory_subset(sample_df, split_name, seed, "all", "all")]
    for family in ("phase_type", "road_type_anchor", "eval_morphology_label", "structure_slice"):
        if family not in sample_df.columns:
            continue
        for label, part in sample_df.groupby(family, dropna=False):
            rows.append(summarize_trajectory_subset(part.reset_index(drop=True), split_name, seed, family, str(label)))
    return rows


def build_trajectory_selection_summary(
    weighted_metrics: dict[str, Any],
    sample_df: pd.DataFrame,
    subset_name: str,
) -> dict[str, float]:
    rmse_tail = _mean_metric(sample_df, "rmse_tail_abs_steer", 10.0)
    tail_pre_ratio = _mean_metric(sample_df, "tail_pre_ratio_abs_steer", 10.0)
    tail_trend_corr = _mean_metric(sample_df, "tail_trend_corr", -1.0)
    tail_direction_match = _mean_metric(sample_df, "tail_direction_match", 0.0)
    turning_count_abs_err = _mean_metric(sample_df, "turning_count_abs_err", 10.0)
    peak_time_abs_err_s = _mean_metric(sample_df, "peak_time_abs_err_s", 10.0)
    boundary_shift_abs_err = _mean_metric(sample_df, "boundary_shift_abs_err", 10.0)
    overall_primary_rmse = float(weighted_metrics["sample_weighted"]["overall_primary"]["steer"]["rmse"])

    tail_score = 0.60 * (rmse_tail / TRAJECTORY_SELECTION_SCALES["rmse_tail_abs_steer"]) + 0.40 * (
        tail_pre_ratio / TRAJECTORY_SELECTION_SCALES["tail_pre_ratio_abs_steer"]
    )
    trend_score = 0.70 * (1.0 - tail_trend_corr) + 0.30 * (1.0 - tail_direction_match)
    turning_score = 0.65 * (
        turning_count_abs_err / TRAJECTORY_SELECTION_SCALES["turning_count_abs_err"]
    ) + 0.35 * (peak_time_abs_err_s / TRAJECTORY_SELECTION_SCALES["peak_time_abs_err_s"])
    continuity_score = boundary_shift_abs_err / TRAJECTORY_SELECTION_SCALES["boundary_shift_abs_err"]
    trajectory_score = 0.40 * tail_score + 0.35 * trend_score + 0.25 * continuity_score

    return {
        "selection_subset": subset_name,
        "rmse_tail_abs_steer": rmse_tail,
        "tail_pre_ratio_abs_steer": tail_pre_ratio,
        "tail_trend_corr": tail_trend_corr,
        "tail_direction_match": tail_direction_match,
        "turning_count_abs_err": turning_count_abs_err,
        "peak_time_abs_err_s": peak_time_abs_err_s,
        "boundary_shift_abs_err": boundary_shift_abs_err,
        "overall_primary_steer_rmse": overall_primary_rmse,
        "tail_score": float(tail_score),
        "trend_score": float(trend_score),
        "turning_score": float(turning_score),
        "continuity_score": float(continuity_score),
        "trajectory_score": float(trajectory_score),
    }


@torch.no_grad()
def evaluate_model_on_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    meta_df: pd.DataFrame,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: str,
    loss_fn,
    split_name: str,
    seed: int,
    epoch: int | None = None,
) -> dict[str, Any]:
    model.eval()
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)
    preds = []
    trues = []
    masks = []
    ctxs_raw = []
    total_loss = 0.0
    total_main = 0.0
    n_batch = 0

    for batch in loader:
        src = batch["src"].to(device=device, dtype=torch.float32)
        y_true = batch["y_true"].to(device=device, dtype=torch.float32)
        curve_norm = batch["curve_norm"].to(device=device, dtype=torch.float32)
        ctx = batch["ctx"].to(device=device, dtype=torch.float32)
        ctx_raw = batch.get("ctx_raw")
        if ctx_raw is not None:
            ctx_raw = ctx_raw.to(dtype=torch.float32)
        event_mask = batch["event_mask"].to(device=device, dtype=torch.float32)
        mechanism_id = batch.get("mechanism_id")
        if mechanism_id is not None:
            mechanism_id = mechanism_id.to(device=device)
        y_hat, extras = model(src, ctx, curve_norm, mechanism_id=mechanism_id)
        for key in (
            "first_turn_has",
            "first_turn_bin",
            "first_turn_dir",
            "first_reversal_has",
            "first_reversal_bin",
            "major_peak_bin",
        ):
            if key in batch:
                extras[key] = batch[key].to(device=device)
        loss, loss_main = loss_fn(y_hat, y_true, event_mask, y_mean_t, y_std_t, extras, epoch=epoch)
        total_loss += float(loss.item())
        total_main += float(loss_main.item())
        n_batch += 1
        pred_np = (y_hat * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)).cpu().numpy()
        true_np = (y_true * y_std_t.view(1, 1, -1) + y_mean_t.view(1, 1, -1)).cpu().numpy()
        preds.append(pred_np)
        trues.append(true_np)
        masks.append(event_mask.cpu().numpy())
        if ctx_raw is not None:
            ctxs_raw.append(ctx_raw.cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    mask = np.concatenate(masks, axis=0)
    weighted = compute_weighted_metrics(pred, true, mask)
    label_df, morph_rows = compute_morphology_metric_rows(pred, true, mask, meta_df, split_name, seed)

    trajectory_sample_df = pd.DataFrame()
    trajectory_subset_rows: list[dict[str, Any]] = []
    selection_summary = build_trajectory_selection_summary(weighted, pd.DataFrame(), subset_name="all")
    if ctxs_raw:
        ctx_raw_np = np.concatenate(ctxs_raw, axis=0)
        trajectory_sample_df = compute_trajectory_sample_metrics(
            meta_df=meta_df,
            pred=pred,
            true=true,
            mask=mask,
            ctx_raw=ctx_raw_np,
            split_name=split_name,
            seed=seed,
        )
        trajectory_subset_rows = build_trajectory_subset_rows(trajectory_sample_df, split_name, seed)
        primary_mask = trajectory_sample_df["phase_type"].astype(str).eq("primary")
        if primary_mask.any():
            meta_work = meta_df.reset_index(drop=True).copy()
            primary_idx = meta_work["phase_type"].astype(str).eq("primary").to_numpy()
            primary_weighted = compute_weighted_metrics(pred[primary_idx], true[primary_idx], mask[primary_idx])
            selection_summary = build_trajectory_selection_summary(
                weighted_metrics=primary_weighted,
                sample_df=trajectory_sample_df.loc[primary_mask].reset_index(drop=True),
                subset_name="primary",
            )
            structure_summary = build_trajectory_selection_summary(weighted, pd.DataFrame(), subset_name="structure_heavy")
            non_structure_summary = build_trajectory_selection_summary(weighted, pd.DataFrame(), subset_name="non_structure_heavy")
            if "structure_heavy" in meta_work.columns:
                structure_idx = primary_idx & meta_work["structure_heavy"].astype(int).to_numpy().astype(bool)
                non_structure_idx = primary_idx & (~meta_work["structure_heavy"].astype(int).to_numpy().astype(bool))
                if structure_idx.any():
                    structure_weighted = compute_weighted_metrics(pred[structure_idx], true[structure_idx], mask[structure_idx])
                    structure_summary = build_trajectory_selection_summary(
                        weighted_metrics=structure_weighted,
                        sample_df=trajectory_sample_df.loc[primary_mask & trajectory_sample_df["structure_heavy"].astype(int).eq(1)].reset_index(drop=True),
                        subset_name="structure_heavy",
                    )
                if non_structure_idx.any():
                    non_structure_weighted = compute_weighted_metrics(pred[non_structure_idx], true[non_structure_idx], mask[non_structure_idx])
                    non_structure_summary = build_trajectory_selection_summary(
                        weighted_metrics=non_structure_weighted,
                        sample_df=trajectory_sample_df.loc[primary_mask & trajectory_sample_df["structure_heavy"].astype(int).eq(0)].reset_index(drop=True),
                        subset_name="non_structure_heavy",
                    )
        else:
            selection_summary = build_trajectory_selection_summary(weighted, trajectory_sample_df, subset_name="all")
            structure_summary = build_trajectory_selection_summary(weighted, pd.DataFrame(), subset_name="structure_heavy")
            non_structure_summary = build_trajectory_selection_summary(weighted, pd.DataFrame(), subset_name="non_structure_heavy")
    else:
        structure_summary = build_trajectory_selection_summary(weighted, pd.DataFrame(), subset_name="structure_heavy")
        non_structure_summary = build_trajectory_selection_summary(weighted, pd.DataFrame(), subset_name="non_structure_heavy")

    return {
        "loss": total_loss / max(n_batch, 1),
        "main_loss": total_main / max(n_batch, 1),
        "metrics": weighted,
        "arrays": {"pred": pred, "true": true, "mask": mask},
        "label_df": label_df.reset_index(drop=True),
        "morph_rows": morph_rows,
        "meta_df": meta_df.reset_index(drop=True).copy(),
        "split": split_name,
        "seed": int(seed),
        "trajectory_sample_df": trajectory_sample_df,
        "trajectory_subset_rows": trajectory_subset_rows,
        "selection_summary": selection_summary,
        "selection_summary_structure_heavy": structure_summary,
        "selection_summary_non_structure_heavy": non_structure_summary,
    }
