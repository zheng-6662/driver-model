from .shared import *

# =========================
# Multi-scale loss helpers
# =========================

def _diff1(x: torch.Tensor) -> torch.Tensor:
    # x: (B,T,C) or (B,T)
    if x.dim() == 2:
        return x[:, 1:] - x[:, :-1]
    return x[:, 1:, :] - x[:, :-1, :]


def _diff2(x: torch.Tensor) -> torch.Tensor:
    return _diff1(_diff1(x))


def weighted_l1_loss_per_sample(pred: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B, ...) tensors
    sample_weight: (B,) or (B,1)
    """
    per_sample = (pred - target).abs().reshape(pred.shape[0], -1).mean(dim=1)
    weight = sample_weight.reshape(sample_weight.shape[0]).to(per_sample.dtype)
    weight_sum = torch.clamp(weight.sum(), min=1e-6)
    return (per_sample * weight).sum() / weight_sum



def compute_amplitude_loss(y_hat: torch.Tensor, y_true: torch.Tensor, sample_weight=None) -> torch.Tensor:
    """
    steer-only 幅值损失（支持样本级加权）
    """
    pred = y_hat[:, :, 0]   # steer
    true = y_true[:, :, 0]

    pred_peak = pred.abs().amax(dim=1)
    true_peak = true.abs().amax(dim=1)

    pred_range = pred.amax(dim=1) - pred.amin(dim=1)
    true_range = true.amax(dim=1) - true.amin(dim=1)

    if sample_weight is None:
        loss_peak = F.l1_loss(pred_peak, true_peak)
        loss_range = F.l1_loss(pred_range, true_range)
    else:
        loss_peak = weighted_l1_loss_per_sample(pred_peak.unsqueeze(1), true_peak.unsqueeze(1), sample_weight)
        loss_range = weighted_l1_loss_per_sample(pred_range.unsqueeze(1), true_range.unsqueeze(1), sample_weight)

    return 0.7 * loss_peak + 0.3 * loss_range


def weighted_mean_per_sample(loss_per_sample: torch.Tensor, sample_weight=None) -> torch.Tensor:
    per_sample = loss_per_sample.reshape(loss_per_sample.shape[0])
    if sample_weight is None:
        return per_sample.mean()
    weight = sample_weight.reshape(sample_weight.shape[0]).to(per_sample.dtype)
    weight_sum = torch.clamp(weight.sum(), min=1e-6)
    return (per_sample * weight).sum() / weight_sum


def mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1)


def weighted_mse_loss_per_sample(pred: torch.Tensor, target: torch.Tensor, sample_weight=None) -> torch.Tensor:
    return weighted_mean_per_sample(mse_per_sample(pred, target), sample_weight)


def weighted_channel_task_loss(y_hat: torch.Tensor, y_true: torch.Tensor, sample_weight=None) -> torch.Tensor:
    loss_steer = mse_per_sample(y_hat[:, :, 0], y_true[:, :, 0])
    loss_yaw = mse_per_sample(y_hat[:, :, 1], y_true[:, :, 1])
    loss_ay = mse_per_sample(y_hat[:, :, 2], y_true[:, :, 2])
    total = (
        W_TASK_STEER * loss_steer
        + W_TASK_YAW * loss_yaw
        + W_TASK_AY * loss_ay
    ) / (W_TASK_STEER + W_TASK_YAW + W_TASK_AY)
    return weighted_mean_per_sample(total, sample_weight)


def weighted_steer_local_mse(steer_pred: torch.Tensor, steer_true: torch.Tensor, w_local: torch.Tensor, sample_weight=None) -> torch.Tensor:
    per_sample = (((steer_pred[:, 1:] - steer_true[:, 1:]) ** 2) * w_local).mean(dim=1)
    return weighted_mean_per_sample(per_sample, sample_weight)


def _avg_pool_seq_torch(seq: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    if seq.ndim != 2:
        raise ValueError(f"Expected (B,T), got {tuple(seq.shape)}")
    t_len = int(seq.shape[1])
    k = max(1, min(int(kernel_size), t_len))
    s = max(1, int(stride))
    pooled = F.avg_pool1d(seq.unsqueeze(1), kernel_size=k, stride=s).squeeze(1)
    if pooled.shape[1] == 0:
        pooled = seq.mean(dim=1, keepdim=True)
    return pooled


def _avg_pool_seq_np(seq, kernel_size: int, stride: int) -> np.ndarray:
    x = np.asarray(seq, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    t_len = int(x.shape[1])
    k = max(1, min(int(kernel_size), t_len))
    s = max(1, int(stride))
    pooled = [x[:, start:start + k].mean(axis=1) for start in range(0, t_len - k + 1, s)]
    if not pooled:
        pooled = [x.mean(axis=1)]
    return np.stack(pooled, axis=1)


def get_rev_aux_target(rev_gt_weak_b: torch.Tensor, rev_gt_strong_b: torch.Tensor) -> torch.Tensor:
    if REV_AUX_TARGET == "strong":
        return rev_gt_strong_b.float()
    if REV_AUX_TARGET == "weak":
        return rev_gt_weak_b.float()
    raise ValueError(f"Unsupported REV_AUX_TARGET={REV_AUX_TARGET!r}")


def build_reversal_sample_weight(
    rev_gt_used: torch.Tensor,
    rev_gt_weak: torch.Tensor | None = None,
    rev_gt_strong: torch.Tensor | None = None,
    weak_coef: float | None = None,
    strong_coef: float | None = None,
) -> torch.Tensor:
    if REV_SAMPLE_WEIGHT_MODE == "strong":
        base = rev_gt_used.float()
        return 1.0 + (REV_SAMPLE_WEIGHT - 1.0) * base
    if REV_SAMPLE_WEIGHT_MODE == "weak":
        base = rev_gt_weak if rev_gt_weak is not None else rev_gt_used
        return 1.0 + (REV_SAMPLE_WEIGHT - 1.0) * base.float()
    if REV_SAMPLE_WEIGHT_MODE == "hybrid":
        weak_base = rev_gt_weak if rev_gt_weak is not None else rev_gt_used
        strong_base = rev_gt_strong if rev_gt_strong is not None else rev_gt_used
        weak_coef = float(REV_HYBRID_WEAK_COEF if weak_coef is None else weak_coef)
        strong_coef = float(REV_HYBRID_STRONG_COEF if strong_coef is None else strong_coef)
        weak_term = weak_coef * (REV_SAMPLE_WEIGHT - 1.0) * weak_base.float()
        strong_term = strong_coef * (REV_SAMPLE_WEIGHT - 1.0) * strong_base.float()
        return 1.0 + weak_term + strong_term
    raise ValueError(f"Unsupported REV_SAMPLE_WEIGHT_MODE={REV_SAMPLE_WEIGHT_MODE!r}")

def _soft_reversal_prob(seq: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Soft reversal probability between consecutive timesteps.
    seq: (B,T)
    alpha: scalar tensor (>0) controlling soft sign.
    Return: (B,T-1) in [0,1]
    """
    a = torch.clamp(alpha, min=1e-6)
    p_pos = torch.sigmoid(seq / a)
    p_neg = 1.0 - p_pos
    return p_pos[:, :-1] * p_neg[:, 1:] + p_neg[:, :-1] * p_pos[:, 1:]

def _soft_peak_time(x: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """
    Soft-argmax expected peak time (normalized 0..1).
    x: (B,T) non-negative (e.g., abs(rate))
    temp: scalar tensor (>0)
    Return: (B,) in [0,1]
    """
    t = torch.linspace(0.0, 1.0, x.shape[1], device=x.device, dtype=x.dtype)
    tau = torch.clamp(temp, min=1e-6)
    w = torch.softmax(x / tau, dim=1)
    return (w * t.unsqueeze(0)).sum(dim=1)


def _first_reversal_idx_torch(steer_seq: torch.Tensor, eps: float) -> torch.Tensor:
    x = steer_seq.detach()
    sign = torch.zeros_like(x, dtype=torch.int64)
    sign = torch.where(x > float(eps), torch.ones_like(sign), sign)
    sign = torch.where(x < -float(eps), -torch.ones_like(sign), sign)
    out = torch.full((x.shape[0],), -1, device=x.device, dtype=torch.long)
    for b in range(x.shape[0]):
        nz_idx = torch.nonzero(sign[b] != 0, as_tuple=False).squeeze(1)
        if nz_idx.numel() < 2:
            continue
        nz_sign = sign[b].index_select(0, nz_idx)
        change_idx = torch.nonzero(nz_sign[1:] != nz_sign[:-1], as_tuple=False).squeeze(1)
        if change_idx.numel() == 0:
            continue
        out[b] = nz_idx[int(change_idx[0].item()) + 1]
    return out


def compute_first_reversal_local_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
    sample_weight=None,
    rev_gt_weak=None,
    radius: int = 16,
):
    if y_hat.shape[1] < 2:
        return torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)

    y_hat_den = y_hat * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
    y_true_den = y_true * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
    steer_pred = y_hat_den[:, :, 0]
    steer_true = y_true_den[:, :, 0]

    alpha = REVSEQ_ALPHA_FRAC * y_std_t[0]
    p_rev_pred = _soft_reversal_prob(steer_pred, alpha)
    target = torch.zeros_like(p_rev_pred)
    radius = max(1, int(radius))

    with torch.no_grad():
        first_idx = _first_reversal_idx_torch(steer_true, eps=REV_EPS_WEAK)
        for b in range(target.shape[0]):
            gt_idx = int(first_idx[b].item())
            if gt_idx < 0:
                continue
            center = max(0, min(int(target.shape[1] - 1), gt_idx - 1))
            left = max(0, center - radius)
            right = min(int(target.shape[1]), center + radius + 1)
            xs = torch.arange(left, right, device=target.device, dtype=target.dtype)
            tri = 1.0 - (xs - float(center)).abs() / float(radius)
            target[b, left:right] = torch.maximum(target[b, left:right], tri.clamp_min(0.0))

    per_sample = ((p_rev_pred - target) ** 2).mean(dim=1)
    if rev_gt_weak is not None:
        per_sample = per_sample * rev_gt_weak.float().reshape(-1)
    return weighted_mean_per_sample(per_sample, sample_weight)


def compute_active_task_losses(y_hat: torch.Tensor, y_true: torch.Tensor, sample_weight=None):
    loss_task = weighted_channel_task_loss(y_hat, y_true, sample_weight)
    loss_amp = compute_amplitude_loss(y_hat, y_true, sample_weight=sample_weight)
    loss_d1 = weighted_mse_loss_per_sample(_diff1(y_hat), _diff1(y_true), sample_weight)
    loss_d2 = weighted_mse_loss_per_sample(_diff2(y_hat), _diff2(y_true), sample_weight)
    loss_task = loss_task + W_DIFF1 * loss_d1 + W_DIFF2 * loss_d2 + W_AMP * loss_amp
    return loss_task, loss_amp, loss_d1, loss_d2


def compute_reversal_shape_losses(y_hat: torch.Tensor, y_true: torch.Tensor, y_mean_t: torch.Tensor, y_std_t: torch.Tensor, sample_weight=None, use_reversal_local_weight=True):
    y_hat_den = y_hat * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
    y_true_den = y_true * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
    steer_pred = y_hat_den[:, :, 0]
    steer_true = y_true_den[:, :, 0]

    alpha = REVSEQ_ALPHA_FRAC * y_std_t[0]
    p_rev_pred = _soft_reversal_prob(steer_pred, alpha)
    with torch.no_grad():
        p_rev_true = _soft_reversal_prob(steer_true, alpha)
    loss_revseq = weighted_mse_loss_per_sample(p_rev_pred, p_rev_true, sample_weight)

    steer_rate_pred = _diff1(steer_pred).abs()
    steer_rate_true = _diff1(steer_true).abs()
    temp = PEAK_TEMP_FRAC * (steer_rate_true.mean() + EPS)
    peak_pred = _soft_peak_time(steer_rate_pred, temp)
    with torch.no_grad():
        peak_true = _soft_peak_time(steer_rate_true, temp)
    loss_peaktime = weighted_mse_loss_per_sample(peak_pred, peak_true, sample_weight)

    with torch.no_grad():
        rate_norm = steer_rate_true / (steer_rate_true.mean(dim=1, keepdim=True) + EPS)
        rev_seq = p_rev_true if use_reversal_local_weight else torch.zeros_like(rate_norm)
        w_local = 1.0 + W_STEER_RATE * rate_norm + W_STEER_REV * rev_seq
        w_local = torch.clamp(w_local, max=STEER_WT_MAX)
    loss_steer_wt = weighted_steer_local_mse(steer_pred, steer_true, w_local, sample_weight)
    return loss_revseq, loss_peaktime, loss_steer_wt


def compute_trend_loss(y_hat: torch.Tensor, y_true: torch.Tensor, y_mean_t: torch.Tensor, y_std_t: torch.Tensor, sample_weight=None) -> torch.Tensor:
    y_hat_den = y_hat * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
    y_true_den = y_true * y_std_t.view(1, 1, 3) + y_mean_t.view(1, 1, 3)
    steer_pred = y_hat_den[:, :, 0]
    steer_true = y_true_den[:, :, 0]
    trend_pred = _avg_pool_seq_torch(steer_pred, TREND_POOL_KERNEL, TREND_POOL_STRIDE)
    trend_true = _avg_pool_seq_torch(steer_true, TREND_POOL_KERNEL, TREND_POOL_STRIDE)

    loss_level = weighted_mse_loss_per_sample(trend_pred, trend_true, sample_weight)
    if TREND_LOSS_MODE == "pooled_level_mse_v1":
        return loss_level
    if TREND_LOSS_MODE != "pooled_delta_direction_v1":
        raise ValueError(f"Unsupported TREND_LOSS_MODE={TREND_LOSS_MODE!r}")
    if trend_pred.shape[1] <= 1:
        return loss_level

    # Match coarse segment-to-segment movement directly instead of only pooled levels.
    trend_delta_pred = _diff1(trend_pred)
    trend_delta_true = _diff1(trend_true)
    loss_delta = weighted_mse_loss_per_sample(trend_delta_pred, trend_delta_true, sample_weight)

    delta_scale = torch.clamp(trend_delta_true.detach().abs().mean(dim=1, keepdim=True), min=TREND_SIGN_EPS)
    trend_dir_pred = torch.tanh(trend_delta_pred / delta_scale)
    with torch.no_grad():
        trend_dir_true = torch.tanh(trend_delta_true / delta_scale)
    loss_dir = weighted_mse_loss_per_sample(trend_dir_pred, trend_dir_true, sample_weight)

    return (
        TREND_LEVEL_WEIGHT * loss_level
        + TREND_DELTA_WEIGHT * loss_delta
        + TREND_DIR_WEIGHT * loss_dir
    )


def _sec_to_future_idx(sec: float, future_len: int) -> int:
    idx = int(round(float(sec) * float(FS)))
    return max(0, min(int(future_len), idx))


def _build_late_ramp(future_len: int, start_sec: float, device, dtype, power: float = 1.0) -> torch.Tensor:
    late_start_idx = _sec_to_future_idx(start_sec, future_len)
    ramp = torch.zeros((1, future_len), device=device, dtype=dtype)
    if late_start_idx >= future_len:
        return ramp
    weights = torch.linspace(0.0, 1.0, future_len - late_start_idx, device=device, dtype=dtype)
    if float(power) != 1.0:
        weights = weights.clamp_min(0.0).pow(float(power))
    ramp[:, late_start_idx:] = weights
    return ramp


def _build_late_binary_mask(future_len: int, start_sec: float, device, dtype) -> torch.Tensor:
    late_start_idx = _sec_to_future_idx(start_sec, future_len)
    mask = torch.zeros((1, future_len), device=device, dtype=dtype)
    if late_start_idx < future_len:
        mask[:, late_start_idx:] = 1.0
    return mask



def _build_hard_late_masks(steer_true_den: torch.Tensor, rev_gt_weak=None, rev_gt_strong=None):
    B, T = steer_true_den.shape
    hard_late_mask = torch.zeros_like(steer_true_den)
    late_start_idx = _sec_to_future_idx(HARD_LATE_START_SEC, T)
    tail_start_idx = _sec_to_future_idx(HARD_TAIL_START_SEC, T)
    if late_start_idx < T:
        hard_late_mask[:, late_start_idx:] = 1.0

    gt_peak = steer_true_den.detach().abs().amax(dim=1)
    if tail_start_idx < T:
        gt_tail = steer_true_den.detach()[:, tail_start_idx:].abs().amax(dim=1)
    else:
        gt_tail = gt_peak

    if gt_peak.numel() > 1:
        peak_thr = torch.quantile(gt_peak, HARD_PEAK_QUANTILE)
        tail_thr = torch.quantile(gt_tail, HARD_TAIL_QUANTILE)
    else:
        peak_thr = gt_peak[0]
        tail_thr = gt_tail[0]
    hard_pos_mask = (gt_peak >= peak_thr) & (gt_tail >= tail_thr)

    if rev_gt_strong is not None:
        hard_rev_mask = rev_gt_strong.view(-1) > 0.5
    else:
        hard_rev_mask = torch.zeros((B,), device=steer_true_den.device, dtype=torch.bool)
    if rev_gt_weak is not None:
        weak_rev_mask = rev_gt_weak.view(-1) > 0.5
    else:
        weak_rev_mask = torch.zeros((B,), device=steer_true_den.device, dtype=torch.bool)
    hard_mask = (hard_rev_mask | (weak_rev_mask & hard_pos_mask)).to(steer_true_den.dtype)
    return hard_mask, hard_late_mask


def _tail_guard_stats(steer_pred_den: torch.Tensor, steer_true_den: torch.Tensor, start_sec: float):
    if steer_pred_den.ndim != 2 or steer_true_den.ndim != 2:
        raise ValueError(
            f"Expected 2D steer tensors, got pred={tuple(steer_pred_den.shape)} true={tuple(steer_true_den.shape)}"
        )
    if steer_pred_den.shape != steer_true_den.shape:
        raise ValueError(
            f"Shape mismatch for tail guard stats: pred={tuple(steer_pred_den.shape)} true={tuple(steer_true_den.shape)}"
        )
    start_idx = _sec_to_future_idx(start_sec, steer_pred_den.shape[1])
    pred_tail = steer_pred_den[:, start_idx:] if start_idx < steer_pred_den.shape[1] else steer_pred_den[:, -1:]
    true_tail = steer_true_den[:, start_idx:] if start_idx < steer_true_den.shape[1] else steer_true_den[:, -1:]
    gt_tail_amp = true_tail.amax(dim=1) - true_tail.amin(dim=1)
    pred_tail_amp = pred_tail.amax(dim=1) - pred_tail.amin(dim=1)
    gt_tail_motion = (true_tail - true_tail.mean(dim=1, keepdim=True)).abs().mean(dim=1)
    pred_tail_motion = (pred_tail - pred_tail.mean(dim=1, keepdim=True)).abs().mean(dim=1)
    denom = gt_tail_amp.clamp_min(1e-6)
    amp_floor = STRONG_POS_TAIL_RATIO_FLOOR * gt_tail_amp
    flat_floor = STRONG_POS_TAIL_FLAT_FRAC * gt_tail_amp
    amp_ratio = pred_tail_amp / denom
    amp_deficit = (amp_floor - pred_tail_amp).clamp_min(0.0) / denom
    flat_deficit = (flat_floor - pred_tail_motion).clamp_min(0.0) / denom
    return {
        "tail_start_idx": int(start_idx),
        "gt_tail_amp": gt_tail_amp,
        "gt_tail_motion": gt_tail_motion,
        "pred_tail_amp": pred_tail_amp,
        "pred_tail_motion": pred_tail_motion,
        "amp_ratio": amp_ratio,
        "amp_deficit": amp_deficit,
        "flat_deficit": flat_deficit,
    }


def _expand_late_mask(mask: torch.Tensor, batch_size: int, target_dtype: torch.dtype):
    late_mask = mask.to(dtype=target_dtype)
    if late_mask.ndim != 2:
        raise ValueError(f"Expected 2D late mask, got shape={tuple(late_mask.shape)}")
    if late_mask.shape[0] == 1 and batch_size != 1:
        late_mask = late_mask.expand(batch_size, -1)
    elif late_mask.shape[0] != batch_size:
        raise ValueError(f"Late mask batch mismatch: mask={tuple(late_mask.shape)} batch_size={batch_size}")
    return late_mask


def _compute_underamp_targets(steer_base_den: torch.Tensor, steer_true_den: torch.Tensor):
    target_start_sec = max(float(LATE_UNDERAMP_DETECTOR_START_SEC), float(STRONG_POS_TAIL_GUARD_START_SEC))
    tail_stats = _tail_guard_stats(
        steer_base_den,
        steer_true_den,
        target_start_sec,
    )
    severity_target = tail_stats["amp_deficit"] + LATE_UNDERAMP_SEVERITY_FLATNESS_WEIGHT * tail_stats["flat_deficit"]
    risk_target = (severity_target >= float(LATE_UNDERAMP_RISK_BIN_THR)).to(steer_true_den.dtype)
    tail_stats["severity_target"] = severity_target
    tail_stats["risk_target"] = risk_target
    return tail_stats


def compute_underamp_detector_loss(
    forward_aux,
    y_true: torch.Tensor,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
    sample_weight=None,
    rev_gt_strong=None,
):
    steer_base_norm = None if forward_aux is None else forward_aux.get("steer_base_norm")
    underamp_risk_logit = None if forward_aux is None else forward_aux.get("underamp_risk_logit")
    underamp_severity_pred = None if forward_aux is None else forward_aux.get("underamp_severity_pred")
    if steer_base_norm is None or underamp_risk_logit is None or underamp_severity_pred is None:
        zero = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
        return zero, zero

    steer_true_den = y_true[:, :, 0] * y_std_t[0] + y_mean_t[0]
    steer_base_den = steer_base_norm.detach() * y_std_t[0] + y_mean_t[0]
    target_stats = _compute_underamp_targets(steer_base_den.detach(), steer_true_den.detach())
    severity_target = target_stats["severity_target"].detach()
    risk_target = target_stats["risk_target"].detach()
    focus = 1.0 + severity_target
    if rev_gt_strong is not None:
        focus = focus + 0.5 * (rev_gt_strong.view(-1) > 0.5).to(y_true.dtype)

    risk_per_sample = F.binary_cross_entropy_with_logits(
        underamp_risk_logit.view(-1),
        risk_target.view(-1),
        reduction="none",
    ) * focus.view(-1)
    severity_per_sample = F.smooth_l1_loss(
        underamp_severity_pred.view(-1),
        severity_target.view(-1),
        reduction="none",
    ) * focus.view(-1)
    return (
        weighted_mean_per_sample(risk_per_sample, sample_weight),
        weighted_mean_per_sample(severity_per_sample, sample_weight),
    )


def compute_head_prefix_onset_protection_losses(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
    sample_weight=None,
):
    steer_pred_den = y_hat[:, :, 0] * y_std_t[0] + y_mean_t[0]
    steer_true_den = y_true[:, :, 0] * y_std_t[0] + y_mean_t[0]
    t_len = int(steer_pred_den.shape[1])
    prefix_len = max(1, min(t_len, _sec_to_future_idx(1.0, t_len)))
    head_len = max(1, int(round(0.25 * t_len)))
    onset_len = max(2, prefix_len)

    loss_prefix = weighted_mse_loss_per_sample(
        steer_pred_den[:, :prefix_len],
        steer_true_den[:, :prefix_len],
        sample_weight=sample_weight,
    )
    loss_head = weighted_mse_loss_per_sample(
        steer_pred_den[:, :head_len],
        steer_true_den[:, :head_len],
        sample_weight=sample_weight,
    )

    pred_motion = torch.abs(steer_pred_den[:, :onset_len] - steer_pred_den[:, :1])
    true_motion = torch.abs(steer_true_den[:, :onset_len] - steer_true_den[:, :1])
    onset_weight = torch.linspace(
        1.5,
        0.5,
        onset_len,
        device=y_hat.device,
        dtype=y_hat.dtype,
    ).view(1, onset_len)
    onset_per_sample = (((pred_motion - true_motion) ** 2) * onset_weight).mean(dim=1)
    loss_onset = weighted_mean_per_sample(onset_per_sample, sample_weight)
    return loss_prefix, loss_head, loss_onset


def compute_mainline_tail_calibration_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
    sample_weight=None,
    rev_gt_strong=None,
):
    if (not ENABLE_MAINLINE_TAIL_CALIB) or W_MAINLINE_TAIL_CALIB <= 0.0:
        return torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)

    steer_pred_den = y_hat[:, :, 0] * y_std_t[0] + y_mean_t[0]
    steer_true_den = y_true[:, :, 0] * y_std_t[0] + y_mean_t[0]
    tail_stats = _tail_guard_stats(
        steer_pred_den,
        steer_true_den,
        MAINLINE_TAIL_CALIB_START_SEC,
    )

    gt_tail_amp = tail_stats["gt_tail_amp"].detach()
    gt_tail_motion = tail_stats["gt_tail_motion"].detach()
    amp_floor = MAINLINE_TAIL_AMP_RATIO_FLOOR * gt_tail_amp
    motion_floor = MAINLINE_TAIL_MOTION_RATIO_FLOOR * gt_tail_motion
    amp_denom = gt_tail_amp.clamp_min(1e-6)
    amp_deficit = (amp_floor - tail_stats["pred_tail_amp"]).clamp_min(0.0) / amp_denom
    flat_deficit = (motion_floor - tail_stats["pred_tail_motion"]).clamp_min(0.0) / amp_denom

    focus = gt_tail_amp / gt_tail_amp.mean().clamp_min(1e-6)
    focus = focus.clamp(0.5, MAINLINE_TAIL_FOCUS_MAX)
    if rev_gt_strong is not None:
        strong_mask = (rev_gt_strong.view(-1) > 0.5).to(y_true.dtype)
        focus = focus + MAINLINE_TAIL_STRONG_POS_EXTRA * strong_mask

    per_sample_loss = focus * (amp_deficit ** 2 + flat_deficit ** 2)
    return weighted_mean_per_sample(per_sample_loss, sample_weight)


def compute_coarse_fine_losses(forward_aux, y_true: torch.Tensor, y_mean_t: torch.Tensor, y_std_t: torch.Tensor, sample_weight=None, is_curve=None, rev_gt_weak=None, rev_gt_strong=None):
    steer_coarse_norm = None if forward_aux is None else forward_aux.get("steer_coarse_norm")
    steer_coarse_up_norm = None if forward_aux is None else forward_aux.get("steer_coarse_up_norm")
    steer_fine_raw_norm = None if forward_aux is None else forward_aux.get("steer_fine_raw_norm", forward_aux.get("steer_fine_norm"))
    steer_fine_out_norm = None if forward_aux is None else forward_aux.get("steer_fine_norm", steer_fine_raw_norm)
    if (
        steer_coarse_norm is None
        or steer_coarse_up_norm is None
        or steer_fine_raw_norm is None
        or steer_fine_out_norm is None
    ):
        zero = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
        return zero, zero, zero

    steer_true_den = y_true[:, :, 0] * y_std_t[0] + y_mean_t[0]
    steer_coarse_den = steer_coarse_norm * y_std_t[0] + y_mean_t[0]
    steer_coarse_up_den = steer_coarse_up_norm * y_std_t[0] + y_mean_t[0]
    steer_fine_raw_den = steer_fine_raw_norm * y_std_t[0]
    steer_fine_out_den = steer_fine_out_norm * y_std_t[0]

    trend_true = _avg_pool_seq_torch(steer_true_den, TREND_POOL_KERNEL, TREND_POOL_STRIDE)
    fine_pool = _avg_pool_seq_torch(steer_fine_raw_den, TREND_POOL_KERNEL, TREND_POOL_STRIDE)
    hard_mask, hard_late_mask = _build_hard_late_masks(steer_true_den, rev_gt_weak=rev_gt_weak, rev_gt_strong=rev_gt_strong)

    if ENABLE_PHASE_ADAPTIVE_TREND:
        seg_w = torch.ones_like(trend_true)
        t = torch.arange(trend_true.shape[1], device=trend_true.device, dtype=trend_true.dtype)
        early_mask = (t < float(TREND_EARLY_BINS)).to(trend_true.dtype).unsqueeze(0)
        late_mask = (t >= float(TREND_EARLY_BINS)).to(trend_true.dtype).unsqueeze(0)
        seg_w = seg_w + 0.25 * early_mask
        if is_curve is not None:
            straight = (1.0 - is_curve.float().clamp(0.0, 1.0)).view(-1, 1).to(trend_true.dtype)
            seg_w = seg_w - TREND_LATE_STRAIGHT_DOWN * late_mask * straight
        if rev_gt_strong is not None:
            strong = rev_gt_strong.float().view(-1, 1).to(trend_true.dtype)
            seg_w = seg_w - TREND_LATE_STRONGREV_DOWN * late_mask * strong
        if ENABLE_HARD_LATE_FINE:
            hard_late_bins = (hard_mask.view(-1, 1) > 0) & (late_mask > 0)
            seg_w = torch.where(hard_late_bins, torch.ones_like(seg_w), seg_w)
        seg_w = torch.clamp(seg_w, min=0.25)
        loss_coarse = weighted_mean_per_sample((((steer_coarse_den - trend_true) ** 2) * seg_w).mean(dim=1), sample_weight)
    else:
        loss_coarse = weighted_mse_loss_per_sample(steer_coarse_den, trend_true, sample_weight)
    loss_fine_dc = weighted_mse_loss_per_sample(fine_pool, torch.zeros_like(fine_pool), sample_weight)

    if ENABLE_HARD_LATE_FINE:
        res_gt = steer_true_den - steer_coarse_up_den.detach()
        hard_weight = hard_late_mask * hard_mask.view(-1, 1)
        per_sample_denom = hard_weight.sum(dim=1)
        per_sample_loss = torch.where(
            per_sample_denom > 0,
            (((steer_fine_out_den - res_gt) ** 2) * hard_weight).sum(dim=1) / per_sample_denom.clamp_min(1.0),
            torch.zeros_like(per_sample_denom),
        )
        loss_hard_late_fine = weighted_mean_per_sample(per_sample_loss, sample_weight)
    else:
        loss_hard_late_fine = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
    return loss_coarse, loss_fine_dc, loss_hard_late_fine


def compute_late_residual_head_loss(forward_aux, y_true: torch.Tensor, y_mean_t: torch.Tensor, y_std_t: torch.Tensor, sample_weight=None, rev_gt_weak=None, rev_gt_strong=None):
    steer_base_norm = None if forward_aux is None else forward_aux.get("steer_base_norm")
    late_residual_norm = None if forward_aux is None else forward_aux.get("steer_late_residual_norm")
    late_residual_mask = None if forward_aux is None else forward_aux.get("steer_late_residual_mask")
    if steer_base_norm is None or late_residual_norm is None or late_residual_mask is None:
        return torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)

    steer_true_den = y_true[:, :, 0] * y_std_t[0] + y_mean_t[0]
    steer_base_den = steer_base_norm.detach() * y_std_t[0] + y_mean_t[0]
    late_residual_den = late_residual_norm * y_std_t[0]
    late_target_den = steer_true_den - steer_base_den
    late_target_cap = float(LATE_RESIDUAL_MAX_MAG_NORM) * float(y_std_t[0].detach().cpu().item())
    if late_target_cap > 0.0:
        late_target_den = late_target_den.clamp(min=-late_target_cap, max=late_target_cap)

    late_mask = _expand_late_mask(late_residual_mask, steer_true_den.shape[0], y_true.dtype)
    target_stats = _compute_underamp_targets(steer_base_den.detach(), steer_true_den.detach())
    risk_target = target_stats["risk_target"].detach()
    severity_target = target_stats["severity_target"].detach()
    focus_boost = 1.0 + severity_target
    if rev_gt_strong is not None:
        strong_mask = (rev_gt_strong.view(-1) > 0.5).to(y_true.dtype)
        focus_signal = severity_target
        focus_boost = focus_boost + strong_mask * (
            0.25 * LATE_RESIDUAL_STRONG_BOOST + LATE_RESIDUAL_UNDERAMP_BOOST * focus_signal
        )
    focus_boost = torch.clamp(focus_boost, max=LATE_RESIDUAL_FOCUS_MAX)
    late_weight = late_mask * (0.25 + risk_target.view(-1, 1) + focus_boost.view(-1, 1))
    per_sample_denom = late_weight.sum(dim=1)
    per_sample_loss = torch.where(
        per_sample_denom > 0,
        (((late_residual_den - late_target_den) ** 2) * late_weight).sum(dim=1) / per_sample_denom.clamp_min(1.0),
        torch.zeros_like(per_sample_denom),
    )
    return weighted_mean_per_sample(per_sample_loss, sample_weight)


def compute_strong_pos_tail_guard_loss(y_hat: torch.Tensor, y_true: torch.Tensor, y_mean_t: torch.Tensor, y_std_t: torch.Tensor, sample_weight=None, rev_gt_strong=None):
    if rev_gt_strong is None or W_STRONG_POS_TAIL_GUARD <= 0.0:
        return torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
    steer_pred_den = y_hat[:, :, 0] * y_std_t[0] + y_mean_t[0]
    steer_true_den = y_true[:, :, 0] * y_std_t[0] + y_mean_t[0]
    tail_stats = _tail_guard_stats(
        steer_pred_den,
        steer_true_den,
        STRONG_POS_TAIL_GUARD_START_SEC,
    )
    strong_mask = (rev_gt_strong.view(-1) > 0.5).to(y_true.dtype)
    per_sample_loss = strong_mask * (
        tail_stats["amp_deficit"] ** 2
        + STRONG_POS_TAIL_GUARD_FLATNESS_WEIGHT * (tail_stats["flat_deficit"] ** 2)
    )
    return weighted_mean_per_sample(per_sample_loss, sample_weight)


def compute_total_task_loss(y_hat: torch.Tensor, y_true: torch.Tensor, y_mean_t: torch.Tensor, y_std_t: torch.Tensor, sample_weight=None, use_reversal_local_weight=True, forward_aux=None, is_curve=None, rev_gt_weak=None, rev_gt_strong=None):
    loss_task, loss_amp, loss_d1, loss_d2 = compute_active_task_losses(y_hat, y_true, sample_weight=sample_weight)
    loss_revseq, loss_peaktime, loss_steer_wt = compute_reversal_shape_losses(
        y_hat, y_true, y_mean_t, y_std_t, sample_weight=sample_weight, use_reversal_local_weight=use_reversal_local_weight
    )
    loss_firstrev_local = compute_first_reversal_local_loss(
        y_hat,
        y_true,
        y_mean_t,
        y_std_t,
        sample_weight=sample_weight,
        rev_gt_weak=rev_gt_weak,
        radius=FIRSTREV_LOCAL_RADIUS,
    )
    if ENABLE_STEER_COARSE_FINE:
        loss_trend = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
        loss_trend_coarse, loss_fine_dc, loss_hard_late_fine = compute_coarse_fine_losses(
            forward_aux, y_true, y_mean_t, y_std_t, sample_weight=sample_weight, is_curve=is_curve, rev_gt_weak=rev_gt_weak, rev_gt_strong=rev_gt_strong
        )
        loss_task = loss_task + W_TREND_COARSE * loss_trend_coarse + W_FINE_DC * loss_fine_dc + W_HARD_LATE_FINE * loss_hard_late_fine
    else:
        loss_trend = compute_trend_loss(y_hat, y_true, y_mean_t, y_std_t, sample_weight=sample_weight)
        loss_trend_coarse = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
        loss_fine_dc = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
        loss_hard_late_fine = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
        loss_task = loss_task + W_TREND * loss_trend
    if ENABLE_LATE_RESIDUAL_HEAD:
        loss_late_residual = compute_late_residual_head_loss(
            forward_aux,
            y_true,
            y_mean_t,
            y_std_t,
            sample_weight=sample_weight,
            rev_gt_weak=rev_gt_weak,
            rev_gt_strong=rev_gt_strong,
        )
        loss_task = loss_task + W_LATE_RESIDUAL * loss_late_residual
    else:
        loss_late_residual = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
    loss_underamp_risk, loss_underamp_severity = compute_underamp_detector_loss(
        forward_aux,
        y_true,
        y_mean_t,
        y_std_t,
        sample_weight=sample_weight,
        rev_gt_strong=rev_gt_strong,
    )
    loss_prefix_protect, loss_head_protect, loss_onset_protect = compute_head_prefix_onset_protection_losses(
        y_hat,
        y_true,
        y_mean_t,
        y_std_t,
        sample_weight=sample_weight,
    )
    loss_mainline_tail_calib = compute_mainline_tail_calibration_loss(
        y_hat,
        y_true,
        y_mean_t,
        y_std_t,
        sample_weight=sample_weight,
        rev_gt_strong=rev_gt_strong,
    )
    loss_strong_pos_tail_guard = compute_strong_pos_tail_guard_loss(
        y_hat,
        y_true,
        y_mean_t,
        y_std_t,
        sample_weight=sample_weight,
        rev_gt_strong=rev_gt_strong,
    )
    loss_task = (
        loss_task
        + W_LATE_UNDERAMP_RISK * loss_underamp_risk
        + W_LATE_UNDERAMP_SEVERITY * loss_underamp_severity
        + W_PREFIX_PROTECT * loss_prefix_protect
        + W_HEAD_PROTECT * loss_head_protect
        + W_ONSET_PROTECT * loss_onset_protect
        + W_MAINLINE_TAIL_CALIB * loss_mainline_tail_calib
        + W_STRONG_POS_TAIL_GUARD * loss_strong_pos_tail_guard
    )
    loss_task = (
        loss_task
        + W_REVSEQ * loss_revseq
        + W_PEAKTIME * loss_peaktime
        + W_STEER_WT * loss_steer_wt
        + W_FIRSTREV_LOCAL * loss_firstrev_local
    )
    return (
        loss_task,
        loss_amp,
        loss_d1,
        loss_d2,
        loss_revseq,
        loss_peaktime,
        loss_steer_wt,
        loss_trend,
        loss_trend_coarse,
        loss_fine_dc,
        loss_hard_late_fine,
        loss_late_residual,
        loss_underamp_risk,
        loss_underamp_severity,
        loss_prefix_protect,
        loss_head_protect,
        loss_onset_protect,
        loss_mainline_tail_calib,
        loss_firstrev_local,
        loss_strong_pos_tail_guard,
    )
