# Codex Handoff: Add Boundary-Aware Loss Term

## Goal

Add a differentiable boundary-shift penalty to the training loss in
`F:\data_set_process\data_process\datasetprocess\final_code\model\training\future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`

This prevents the model from smoothing predictions at the cost of
misaligning the onset of steering maneuvers.

---

## What to change

### 1. Add constants (after line 222, near existing `PEAK_TEMP_FRAC`)

```python
# Boundary-shift alignment loss: penalize temporal offset of first significant steering onset
W_BOUNDARY = 0.20           # boundary-shift loss weight (start moderate, tune later)
BOUNDARY_DECAY = 3.0        # exponential decay for early-weighting in soft boundary detection
```

### 2. Add helper function (after `_soft_peak_time` around line 995)

```python
def _soft_boundary_time(x: torch.Tensor, temp: torch.Tensor,
                        decay: float = BOUNDARY_DECAY) -> torch.Tensor:
    """
    Early-weighted soft-argmax: detect the *first* significant peak
    in a non-negative signal, not the global max.

    x: (B, T) non-negative (e.g., |steer_rate|)
    temp: scalar tensor (>0), controls softmax sharpness
    decay: exponential decay constant for temporal prior

    Returns: (B,) in [0, 1], normalized time of first significant peak.
    """
    T = x.shape[1]
    t = torch.linspace(0.0, 1.0, T, device=x.device, dtype=x.dtype)
    tau = torch.clamp(temp, min=1e-6)

    # Temporal prior: exponentially decay weight toward later timesteps
    # At t=0 weight=1.0, at t=1.0 weight=exp(-decay)
    time_prior = torch.exp(-decay * t)  # (T,)

    # Apply prior BEFORE softmax so the softmax competition favors early peaks
    x_weighted = x * time_prior.unsqueeze(0)  # (B, T)
    w = torch.softmax(x_weighted / tau, dim=1)  # (B, T)

    return (w * t.unsqueeze(0)).sum(dim=1)  # (B,)
```

### 3. Add boundary loss computation in TRAINING loop

Location: right after `loss_peaktime = F.mse_loss(peak_pred, peak_true)` (line 1589).

Insert:

```python
                # boundary-shift alignment: early-weighted soft-argmax on |steer_rate|
                boundary_pred = _soft_boundary_time(steer_rate_pred, temp)
                with torch.no_grad():
                    boundary_true = _soft_boundary_time(steer_rate_true, temp)
                loss_boundary = F.mse_loss(boundary_pred, boundary_true)
```

### 4. Add boundary loss to total loss in TRAINING loop

Location: line 1600, change:

```python
loss_task = loss_task + W_REVSEQ * loss_revseq + W_PEAKTIME * loss_peaktime + W_STEER_WT * loss_steer_wt
```

to:

```python
loss_task = loss_task + W_REVSEQ * loss_revseq + W_PEAKTIME * loss_peaktime + W_BOUNDARY * loss_boundary + W_STEER_WT * loss_steer_wt
```

### 5. Mirror in VALIDATION loop

Location: right after `loss_peaktime = F.mse_loss(peak_pred, peak_true)` (around line 1667).

Insert:

```python
                    # boundary-shift alignment
                    boundary_pred = _soft_boundary_time(steer_rate_pred, temp)
                    boundary_true = _soft_boundary_time(steer_rate_true, temp)
                    loss_boundary = F.mse_loss(boundary_pred, boundary_true)
```

And modify the total loss line (around line 1676):

```python
loss_task = loss_task + W_REVSEQ * loss_revseq + W_PEAKTIME * loss_peaktime + W_BOUNDARY * loss_boundary + W_STEER_WT * loss_steer_wt
```

Note: in validation loop, no `torch.no_grad()` wrapper needed for `boundary_true` because the entire val loop is already under `with torch.no_grad()`.

---

## Constraints

- Do NOT change any existing loss weights or logic
- Do NOT change model architecture
- Do NOT rename or remove any existing functions
- The new function `_soft_boundary_time` must follow the same pattern as `_soft_peak_time`
- Keep `BOUNDARY_DECAY` as a module-level constant (not computed from data)
- `temp` parameter for `_soft_boundary_time` reuses the same `temp` variable already computed from `steer_rate_true.mean()`
- `boundary_true` in training loop must be computed inside `torch.no_grad()` (same pattern as `peak_true`)

## Verification

After editing, the script should still be valid Python and importable. No runtime test needed — this is a loss-only change with no architecture impact.

Quick sanity check: `W_BOUNDARY` should appear exactly 2 times in loss computation lines (train + val), plus 1 time in constant definition = 3 total occurrences.
