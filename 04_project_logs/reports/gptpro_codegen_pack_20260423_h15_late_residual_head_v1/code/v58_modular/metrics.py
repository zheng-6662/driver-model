from .shared import *
from .modeling import unpack_model_output
from .losses import _avg_pool_seq_np

def _denorm_y(y_norm_np: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    # y_norm_np: (N,T,3)
    return y_norm_np * y_std.reshape(1, 1, 3) + y_mean.reshape(1, 1, 3)




def has_reversal_np(steer_seq_1d, eps=REV_EPS_WEAK):
    """Return 1.0 if the steering sequence crosses both +eps and -eps (sign reversal), else 0.0."""
    x = np.asarray(steer_seq_1d, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).any():
        return 0.0
    return 1.0 if (np.nanmax(x) > eps and np.nanmin(x) < -eps) else 0.0

def _binary_metrics(y_true, y_pred):
    """Precision/Recall/F1 for binary labels (0/1)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))
    f1   = 2 * prec * rec / max(1e-12, (prec + rec))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def _safe_mean_np(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return float(np.mean(x))


def _safe_median_np(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return float(np.median(x))


def _safe_rmse_np(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return float(np.sqrt(np.mean(x ** 2)))


def _safe_mae_np(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return float(np.mean(np.abs(x)))


def _safe_ratio_np(num, den, eps=1e-6):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if num.size == 0 or den.size == 0:
        return None
    return float(np.mean(num / np.maximum(den, eps)))


def _crossing_count_np(steer_seq_1d, eps=REV_EPS_WEAK):
    x = np.asarray(steer_seq_1d, dtype=np.float64)
    if x.size < 2 or not np.isfinite(x).any():
        return 0
    sign = np.zeros_like(x, dtype=np.int64)
    sign[x > eps] = 1
    sign[x < -eps] = -1
    nz = sign[sign != 0]
    if nz.size < 2:
        return 0
    return int(np.sum(nz[1:] != nz[:-1]))


def _first_reversal_time_np(steer_seq_1d, eps=REV_EPS_WEAK, fs=200):
    x = np.asarray(steer_seq_1d, dtype=np.float64)
    if x.size < 2 or not np.isfinite(x).any():
        return None
    sign = np.zeros_like(x, dtype=np.int64)
    sign[x > eps] = 1
    sign[x < -eps] = -1
    nz_idx = np.flatnonzero(sign != 0)
    if nz_idx.size < 2:
        return None
    nz_sign = sign[nz_idx]
    change_idx = np.flatnonzero(nz_sign[1:] != nz_sign[:-1])
    if change_idx.size == 0:
        return None
    first_idx = int(nz_idx[change_idx[0] + 1])
    return float(first_idx / max(1, fs))


def _first_threshold_crossing_idx_np(seq_1d, threshold, ref_value=None):
    x = np.asarray(seq_1d, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).any():
        return None
    if ref_value is None:
        ref_value = float(x[0])
    delta = np.abs(x - ref_value)
    idx = np.flatnonzero(delta >= float(threshold))
    if idx.size == 0:
        return None
    return int(idx[0])


def _head_metrics(pred, true, fs=200, head_frac=0.25, onset_thr_ratio=0.15, onset_thr_abs=STEER_ONSET_THR_ABS):
    t_len = int(pred.shape[1])
    head_len = max(1, int(round(t_len * head_frac)))
    pred_head = pred[:, :head_len, 0]
    true_head = true[:, :head_len, 0]
    err_head = pred_head - true_head

    pred_head_amp = np.ptp(pred_head, axis=1)
    true_head_amp = np.ptp(true_head, axis=1)
    pred_head_motion = np.mean(np.abs(pred_head - pred_head[:, :1]), axis=1)
    flat_thr = 0.10 * np.maximum(true_head_amp, 1e-6)

    if head_len > 1:
        pred_head_slope = np.mean(np.abs(np.diff(pred_head, axis=1)), axis=1)
        true_head_slope = np.mean(np.abs(np.diff(true_head, axis=1)), axis=1)
    else:
        pred_head_slope = np.zeros((pred_head.shape[0],), dtype=np.float64)
        true_head_slope = np.zeros((true_head.shape[0],), dtype=np.float64)

    pred_steer = np.asarray(pred[:, :, 0], dtype=np.float64)
    true_steer = np.asarray(true[:, :, 0], dtype=np.float64)
    onset_delay = []
    for i in range(pred_steer.shape[0]):
        true_seq = true_steer[i]
        pred_seq = pred_steer[i]
        true_base = float(true_seq[0])
        pred_base = float(pred_seq[0])
        true_peak_delta = float(np.max(np.abs(true_seq - true_base)))
        onset_thr = max(onset_thr_abs, onset_thr_ratio * true_peak_delta)
        true_idx = _first_threshold_crossing_idx_np(true_seq, threshold=onset_thr, ref_value=true_base)
        if true_idx is None:
            continue
        pred_idx = _first_threshold_crossing_idx_np(pred_seq, threshold=onset_thr, ref_value=pred_base)
        if pred_idx is None:
            pred_idx = t_len - 1
        onset_delay.append((pred_idx - true_idx) / max(1, fs))

    onset_delay = np.asarray(onset_delay, dtype=np.float64)
    return {
        "head_len": int(head_len),
        "head_end_idx": int(head_len - 1),
        "head_end_sec": float((head_len - 1) / max(1, fs)),
        "head_rmse_steer": _safe_rmse_np(err_head),
        "head_mae_steer": _safe_mae_np(err_head),
        "head_amp_pred": _safe_mean_np(pred_head_amp),
        "head_amp_gt": _safe_mean_np(true_head_amp),
        "head_amp_ratio_pred_over_gt": _safe_ratio_np(pred_head_amp, true_head_amp),
        "head_flatness_rate": float(np.mean(pred_head_motion <= flat_thr)),
        "early_slope_pred": _safe_mean_np(pred_head_slope),
        "early_slope_gt": _safe_mean_np(true_head_slope),
        "early_slope_ratio_pred_over_gt": _safe_ratio_np(pred_head_slope, true_head_slope),
        "response_onset_delay_sec": _safe_mean_np(onset_delay),
        "response_onset_delay_mae_sec": _safe_mae_np(onset_delay),
        "n_valid_onset": int(onset_delay.size),
        "response_onset_threshold_ratio": float(onset_thr_ratio),
        "response_onset_threshold_abs": float(onset_thr_abs),
        "steer_angle_unit": STEER_ANGLE_UNIT,
    }


def _tail_metrics(pred, true, fs=200, tail_frac=0.25):
    t_len = int(pred.shape[1])
    tail_len = max(1, int(round(t_len * tail_frac)))
    tail_start = max(0, t_len - tail_len)
    pred_tail = pred[:, tail_start:, 0]
    true_tail = true[:, tail_start:, 0]
    err_tail = pred_tail - true_tail
    pred_tail_std = pred_tail.std(axis=1)
    true_tail_std = true_tail.std(axis=1)
    pred_tail_amp = np.ptp(pred_tail, axis=1)
    true_tail_amp = np.ptp(true_tail, axis=1)
    pred_tail_slope = pred_tail[:, -1] - pred_tail[:, 0]
    true_tail_slope = true_tail[:, -1] - true_tail[:, 0]
    flat_thr = 0.10 * np.maximum(true_tail_amp, 1e-6)
    pred_tail_amp_mean = np.mean(np.abs(pred_tail - pred_tail.mean(axis=1, keepdims=True)), axis=1)
    return {
        "tail_start_idx": int(tail_start),
        "tail_len": int(tail_len),
        "tail_start_sec": float(tail_start / max(1, fs)),
        "tail_rmse_steer": _safe_rmse_np(err_tail),
        "tail_mae_steer": _safe_mae_np(err_tail),
        "tail_std_pred": _safe_mean_np(pred_tail_std),
        "tail_std_gt": _safe_mean_np(true_tail_std),
        "tail_std_ratio_pred_over_gt": _safe_ratio_np(pred_tail_std, true_tail_std),
        "tail_amp_pred": _safe_mean_np(pred_tail_amp),
        "tail_amp_gt": _safe_mean_np(true_tail_amp),
        "tail_amp_ratio_pred_over_gt": _safe_ratio_np(pred_tail_amp, true_tail_amp),
        "tail_slope_mae": _safe_mae_np(pred_tail_slope - true_tail_slope),
        "tail_flatness_rate": float(np.mean(pred_tail_amp_mean <= flat_thr)),
    }


def _peak_metrics(pred, true, fs=200):
    pred_steer = np.asarray(pred[:, :, 0], dtype=np.float64)
    true_steer = np.asarray(true[:, :, 0], dtype=np.float64)
    pred_peak_idx = np.argmax(np.abs(pred_steer), axis=1)
    true_peak_idx = np.argmax(np.abs(true_steer), axis=1)
    pred_peak_val = pred_steer[np.arange(pred_steer.shape[0]), pred_peak_idx]
    true_peak_val = true_steer[np.arange(true_steer.shape[0]), true_peak_idx]
    half_idx = pred_steer.shape[1] // 2
    mask_true_late = true_peak_idx >= half_idx
    return {
        "peak_time_mae_sec": _safe_mae_np((pred_peak_idx - true_peak_idx) / max(1, fs)),
        "peak_time_rmse_sec": _safe_rmse_np((pred_peak_idx - true_peak_idx) / max(1, fs)),
        "peak_mag_mae": _safe_mae_np(pred_peak_val - true_peak_val),
        "peak_mag_rmse": _safe_rmse_np(pred_peak_val - true_peak_val),
        "late_peak_rate_gt": float(np.mean(mask_true_late)),
        "late_peak_recall": float(np.mean(pred_peak_idx[mask_true_late] >= half_idx)) if np.any(mask_true_late) else None,
    }


def _safe_corrcoef_np(a, b, eps=1e-8):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size == 0 or bb.size == 0:
        return np.nan
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    aa_std = float(np.sqrt(np.mean(aa ** 2)))
    bb_std = float(np.sqrt(np.mean(bb ** 2)))
    if aa_std < eps and bb_std < eps:
        return 1.0 if float(np.mean(np.abs(np.asarray(a) - np.asarray(b)))) < eps else 0.0
    if aa_std < eps or bb_std < eps:
        return 0.0
    return float(np.mean((aa / aa_std) * (bb / bb_std)))


def _trend_metrics(pred, true, fs=200, pool_kernel=TREND_POOL_KERNEL, pool_stride=TREND_POOL_STRIDE, sign_eps=TREND_SIGN_EPS):
    pred_steer = np.asarray(pred[:, :, 0], dtype=np.float64)
    true_steer = np.asarray(true[:, :, 0], dtype=np.float64)
    pred_pool = _avg_pool_seq_np(pred_steer, pool_kernel, pool_stride)
    true_pool = _avg_pool_seq_np(true_steer, pool_kernel, pool_stride)

    corr_vec = []
    sign_match_vec = []
    for i in range(pred_pool.shape[0]):
        corr_vec.append(_safe_corrcoef_np(pred_pool[i], true_pool[i]))
        pred_delta = np.diff(pred_pool[i])
        true_delta = np.diff(true_pool[i])
        pred_sign = np.where(pred_delta > sign_eps, 1, np.where(pred_delta < -sign_eps, -1, 0))
        true_sign = np.where(true_delta > sign_eps, 1, np.where(true_delta < -sign_eps, -1, 0))
        sign_match_vec.append(float(np.mean(pred_sign == true_sign)) if pred_sign.size else np.nan)

    corr_vec = np.asarray(corr_vec, dtype=np.float64)
    sign_match_vec = np.asarray(sign_match_vec, dtype=np.float64)
    coarse_err = pred_pool - true_pool
    coarse_delta_err = np.diff(pred_pool, axis=1) - np.diff(true_pool, axis=1) if pred_pool.shape[1] > 1 else np.empty((pred_pool.shape[0], 0), dtype=np.float64)
    corr_valid = corr_vec[np.isfinite(corr_vec)]
    sign_valid = sign_match_vec[np.isfinite(sign_match_vec)]
    return {
        "trend_loss_mode": TREND_LOSS_MODE,
        "trend_pool_kernel": int(min(int(pool_kernel), pred_steer.shape[1])),
        "trend_pool_stride": int(pool_stride),
        "trend_segment_sec": float(min(int(pool_kernel), pred_steer.shape[1]) / max(1, fs)),
        "trend_pooled_len": int(pred_pool.shape[1]),
        "smooth_trend_corr_mean": _safe_mean_np(corr_valid),
        "smooth_trend_corr_median": _safe_median_np(corr_valid),
        "coarse_segment_sign_match_rate": _safe_mean_np(sign_valid),
        "coarse_segment_sign_match_median": _safe_median_np(sign_valid),
        "coarse_segment_mae": _safe_mae_np(coarse_err),
        "coarse_segment_rmse": _safe_rmse_np(coarse_err),
        "coarse_delta_mae": _safe_mae_np(coarse_delta_err),
        "coarse_delta_rmse": _safe_rmse_np(coarse_delta_err),
    }


def _structured_reversal_metrics(pred, true, rev_gt_weak_vec=None, rev_gt_strong_vec=None, fs=200):
    pred_steer = np.asarray(pred[:, :, 0], dtype=np.float64)
    true_steer = np.asarray(true[:, :, 0], dtype=np.float64)
    pred_rev_time = np.array([_first_reversal_time_np(x, eps=REV_EPS_WEAK, fs=fs) for x in pred_steer], dtype=np.float64)
    true_rev_time = np.array([_first_reversal_time_np(x, eps=REV_EPS_WEAK, fs=fs) for x in true_steer], dtype=np.float64)
    pred_rev_count = np.array([_crossing_count_np(x, eps=REV_EPS_WEAK) for x in pred_steer], dtype=np.int64)
    true_rev_count = np.array([_crossing_count_np(x, eps=REV_EPS_WEAK) for x in true_steer], dtype=np.int64)
    mask_both = np.isfinite(pred_rev_time) & np.isfinite(true_rev_time)

    def _bucket(mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0 or not np.any(mask):
            return None
        pred_b = pred[mask]
        true_b = true[mask]
        tail_b = _tail_metrics(pred_b, true_b, fs=fs)
        err_b = pred_b[:, :, 0] - true_b[:, :, 0]
        return {
            "n": int(mask.sum()),
            "rmse_steer": _safe_rmse_np(err_b),
            "tail_rmse_steer": tail_b["tail_rmse_steer"],
            "tail_amp_ratio_pred_over_gt": tail_b["tail_amp_ratio_pred_over_gt"],
            "tail_flatness_rate": tail_b["tail_flatness_rate"],
        }

    out = {
        "first_reversal_time_mae_sec": _safe_mae_np(pred_rev_time[mask_both] - true_rev_time[mask_both]),
        "first_reversal_time_rmse_sec": _safe_rmse_np(pred_rev_time[mask_both] - true_rev_time[mask_both]),
        "reversal_count_mae": _safe_mae_np(pred_rev_count - true_rev_count),
        "reversal_count_exact_match_rate": float(np.mean(pred_rev_count == true_rev_count)),
        "n_both_have_reversal": int(mask_both.sum()),
    }
    if rev_gt_weak_vec is not None:
        rev_gt_weak_vec = np.asarray(rev_gt_weak_vec).astype(np.int64)
        out["by_bucket"] = {
            "straight": _bucket(rev_gt_weak_vec == 0),
            "weak_pos": _bucket(rev_gt_weak_vec == 1),
        }
        if rev_gt_strong_vec is not None:
            rev_gt_strong_vec = np.asarray(rev_gt_strong_vec).astype(np.int64)
            out["by_bucket"]["strong_pos"] = _bucket(rev_gt_strong_vec == 1)
    return out


def _score_value_or_default(value, default):
    if value is None:
        return float(default)
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def compute_structured_score(structured_metrics: dict) -> float:
    return float(
        1.00 * _score_value_or_default(structured_metrics.get("rmse_steer"), 1e6)
        + 0.60 * _score_value_or_default(structured_metrics.get("tail_rmse_steer"), 1e6)
        + 0.80 * _score_value_or_default(structured_metrics.get("first_reversal_time_mae_sec"), 1e6)
        - 0.80 * _score_value_or_default(structured_metrics.get("late_peak_recall"), 0.0)
        - 0.40 * _score_value_or_default(structured_metrics.get("reversal_count_exact_match_rate"), 0.0)
    )


def collect_structured_metrics_from_loader(
    model: nn.Module,
    data_loader: DataLoader,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    fs: int = 200,
):
    model.eval()
    preds, trues = [], []
    rev_gt_weak_all, rev_gt_strong_all = [], []

    with torch.no_grad():
        for batch in data_loader:
            src = batch["src"].to(DEVICE, non_blocking=True)
            y_true_norm = batch["y_norm"].to(DEVICE, non_blocking=True)
            curve_norm = batch["curve_norm"].to(DEVICE, non_blocking=True)
            ctx = batch["ctx"].to(DEVICE, non_blocking=True)
            rev_gt_weak_b = batch.get("rev_gt_weak", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)
            rev_gt_strong_b = batch.get("rev_gt_strong", batch["rev_gt"]).to(DEVICE, non_blocking=True).squeeze(1)

            y_hat_norm, _, _, _ = unpack_model_output(model(src, ctx, curve_norm))
            preds.append(y_hat_norm.detach().cpu().numpy())
            trues.append(y_true_norm.detach().cpu().numpy())
            rev_gt_weak_all.append(rev_gt_weak_b.detach().cpu().numpy())
            rev_gt_strong_all.append(rev_gt_strong_b.detach().cpu().numpy())

    if len(preds) == 0:
        out = {
            "rmse_steer": None,
            "tail_rmse_steer": None,
            "late_peak_recall": None,
            "first_reversal_time_mae_sec": None,
            "reversal_count_exact_match_rate": None,
            "structured_score": float("inf"),
            "n_eval": 0,
        }
        return out

    pred_norm = np.concatenate(preds, axis=0)
    true_norm = np.concatenate(trues, axis=0)
    pred = _denorm_y(pred_norm, y_mean, y_std)
    true = _denorm_y(true_norm, y_mean, y_std)
    err = pred - true
    rmse_ch = np.sqrt(np.mean(err ** 2, axis=(0, 1))).astype(float)

    tail_metrics = _tail_metrics(pred, true, fs=fs)
    peak_metrics = _peak_metrics(pred, true, fs=fs)
    structured_rev_metrics = _structured_reversal_metrics(
        pred,
        true,
        rev_gt_weak_vec=np.concatenate(rev_gt_weak_all, axis=0).astype(np.int64),
        rev_gt_strong_vec=np.concatenate(rev_gt_strong_all, axis=0).astype(np.int64),
        fs=fs,
    )
    out = {
        "rmse_steer": float(rmse_ch[0]),
        "tail_rmse_steer": tail_metrics["tail_rmse_steer"],
        "late_peak_recall": peak_metrics["late_peak_recall"],
        "first_reversal_time_mae_sec": structured_rev_metrics["first_reversal_time_mae_sec"],
        "reversal_count_exact_match_rate": structured_rev_metrics["reversal_count_exact_match_rate"],
        "n_eval": int(pred.shape[0]),
    }
    out["structured_score"] = compute_structured_score(out)
    return out
