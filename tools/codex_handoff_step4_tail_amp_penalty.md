# Codex Handoff: Step 4 — Tail Amplitude Penalty Experiment

## Task Type
Medium risk. Modifies the loss function in the training script. Does NOT change model architecture, protocol config, split, anchor, horizon, or any data pipeline.

## Background and Motivation

This is Step 4 of the conditioned v2 attribution → intervention pipeline. Three prior read-only analysis tasks (Tasks 1–3) have converged on the following mechanism:

**Q1_fast tail degradation**: conditioned v2 worsens tail-segment (step ≥ 200) RMSE on fast-reaction samples (`latency_proxy_bucket = Q1_fast`). The strongest predictor is `peak_abs_amp_err_conditioned` (r=0.72) and `shape_corr_conditioned` drop (|r|=0.62). Boundary drift and event timing are NOT the main drivers.

**boundary_shift worsening**: concentrated in `single_lobe` and `reverse_correction` morphologies; manifests as temporal shift (not slope flattening). The MSE loss provides no explicit amplitude or boundary structure constraint.

**Decision**: Add a single tail amplitude penalty loss term targeting steer channel amplitude in steps 200–399. This is the most directly evidence-backed, minimal, single-variable intervention.

---

## Objective

1. Add `W_TAIL_AMP` hyperparameter and tail amplitude penalty loss to the training script
2. Retrain conditioned v2 from the Task 3 reconstructed baseline checkpoint (same init as Task 3)
3. Save per-sample prediction sequences as `.npz` (same format as Task 3)
4. Report test metrics and compare against the Task 3 conditioned v2 baseline

---

## Files to Modify

**Main training script** (already restored from git):
```
F:\data_set_process\data_process\datasetprocess\final_code\model\training\run_event_conditioned_trajectory_baseline.py
```

**DO NOT modify**:
- `event_conditioned_baseline_model.py` (model architecture)
- `conditioned_trajectory_head.py` (conditioning module)
- Any `protocol_config.json`, `sample_manifest.csv`, split definitions
- Any existing run output folders

---

## Exact Code Change

### Step A: Add import and constant near top of training script

Find the section near the top where constants are defined (around `DEFAULT_BATCH_SIZE`, `DEFAULT_LR`). Add:

```python
# Tail amplitude penalty
TAIL_START = 200          # step index where tail begins (1.0 s at 200 Hz)
W_TAIL_AMP = 0.3          # penalty weight; adjust after seeing results
```

### Step B: Add tail amplitude penalty to the **train loop** loss computation

In the training loop, find:
```python
traj_loss = masked_mse(y_hat, batch["y_true"], traj_mask)
event_breakdown = compute_event_loss(batch, extras["event_logits"])
loss = traj_loss + float(args.event_loss_weight) * event_breakdown.total
```

Replace with:
```python
traj_loss = masked_mse(y_hat, batch["y_true"], traj_mask)
event_breakdown = compute_event_loss(batch, extras["event_logits"])

# Tail amplitude penalty (steer channel only, steps >= TAIL_START)
tail_mask = traj_mask[:, TAIL_START:, :]                          # (B, T_tail, 1)
pred_amp = y_hat[:, TAIL_START:, 0:1].abs()                       # predicted |steer|
true_amp = batch["y_true"][:, TAIL_START:, 0:1].abs()             # GT |steer|
tail_amp_loss = masked_mse(pred_amp, true_amp, tail_mask)

loss = traj_loss + float(args.event_loss_weight) * event_breakdown.total + W_TAIL_AMP * tail_amp_loss
```

> Note: `masked_mse` is already imported from `event_conditioned_baseline_model`. Do not redefine it.

**IMPORTANT**: There are TWO places where `traj_loss = masked_mse(...)` appears in the script — one in the validation loop and one in the training loop. Only add the tail amplitude penalty in the **training loop** (the one followed by `loss.backward()`). Leave the validation loop unchanged to keep validation metrics clean.

### Step C: Log the tail amp loss in training output (optional but helpful)

In the training loop logging section, if there is a per-batch or per-epoch loss print, optionally add `tail_amp_loss` to the output dict or print statement. This is optional — do not break existing logging if it is complex.

---

## Training Config

Use **EXACTLY** the same config as Task 3 conditioned v2, except:
- `run_prefix`: use `"EXP_EVENT_CONDITIONED_TRAJECTORY_V2_TAILAMP_STEP4"`
- `init_checkpoint`: use the reconstructed baseline checkpoint from Task 3:
  find it at `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_BASELINE_FORMAL_20260408_*/best_model.pt`
  (take the most recent baseline run if multiple exist)
- All other parameters identical:
  - seed: 2026
  - device: cuda
  - epochs: 3, min_epochs: 3, patience: 2
  - batch_size: 64, lr: 0.0003
  - event_loss_weight: 0.5
  - conditioning_mode: structured_v2
  - structure_width: 0.065, gate_temperature: 0.04, event_residual_scale: 1.2
  - selection_mode: structure_aware_primary

---

## Prediction Sequence Export

After training, run the eval/export step using the same modified `eval_event_conditioned_trajectory.py` from Task 3 (it already saves `.npz`).

Save to:
```
F:\data_set_process\data_process\reports\step4_tailamp_prediction_sequences.npz
```

Same format as Task 3: `pred (749, 400, 2)`, `true (749, 400, 2)`, `sample_keys (749,)`, `mask (749, 400)`, `channel_names = ['steer_rel', 'speed_delta']`.

---

## Comparison Metrics Required

After eval, compute and report the following **comparison table** (Step 4 vs Task 3 conditioned v2 baseline):

| Metric | Task 3 conditioned v2 | Step 4 (tail amp penalty) | Delta |
|---|---|---|---|
| test rmse_2s_abs_steer (overall) | 0.4973 | ? | ? |
| test rmse_tail_abs_steer (overall) | ? | ? | ? |
| Q1_fast: mean delta_rmse_tail_abs_steer | +0.0155 | ? | ? |
| non-Q1_fast: mean delta_rmse_tail_abs_steer | -0.0345 | ? | ? |
| single_lobe: mean delta_boundary_shift | +0.1821 | ? | ? |

To get Q1_fast slices: join step4 sample metrics with `reports/attribution_master_table.csv` on `sample_key` and filter by `latency_proxy_bucket == 'Q1_fast'`.

---

## Go / No-Go Assessment

After filling the comparison table, apply this assessment:

**Go (method A is working)**:
- Q1_fast `delta_rmse_tail_abs_steer` mean ≤ 0.00 (from baseline +0.0155)
- overall test `rmse_2s_abs_steer` ≤ 0.508 (Task 3 value + tolerance of 0.01)

**No-Go (method A insufficient)**:
- Q1_fast tail metric unchanged or worsened
- OR overall RMSE degraded by > 0.01 vs Task 3

State the Go/No-Go verdict explicitly in the progress log entry.

---

## Output Files

| File | Description |
|---|---|
| Modified training script (in-place edit) | Only loss function section changed |
| `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_V2_TAILAMP_STEP4_*/` | New run folder with checkpoint and metrics |
| `reports/step4_tailamp_prediction_sequences.npz` | Prediction sequences for downstream analysis |

---

## Constraints

- DO NOT change `protocol_config.json`, split, anchor, horizon, or manifest
- DO NOT change model architecture (`conditioned_trajectory_head.py`, `event_conditioned_baseline_model.py`)
- DO NOT enable W_PEAKTIME or W_REVSEQ or any other currently-disabled loss terms — single-variable change only
- Only modify the TRAINING loop loss computation; leave the VALIDATION loop unchanged
- Use GPU: `device: cuda`
- Python: `D:\ProgramData\anaconda3\envs\predict_2\python.exe` (same as Task 3)

---

## Progress Log Requirement

Before returning results, append a detailed progress entry to:
`F:\data_set_process\data_process\reports\project_progress_master.md`

Entry must include:
- Executor: Codex
- What was changed (exact lines / section of script modified)
- Training run path
- Comparison table with Go/No-Go verdict
- Recommended next step based on result
