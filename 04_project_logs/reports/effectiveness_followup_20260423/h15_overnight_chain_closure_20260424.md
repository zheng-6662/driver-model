# H15 Overnight Chain Closure

Date: 2026-04-24

## Goal

Find a protocol-safe, promotion-ready, reproducible mainline checkpoint for the modular V5.8 `H15` line without reopening optimizer / width / bridge / generic-loss sweeps.

## Executed Chain

1. Retrospective scan over saved checkpoint history:
   - `H15_LATE_RESIDUAL_HEAD_v1`
   - `H15_LATE_RESIDUAL_SELECTIVE_v1`
   - result:
     - no hidden promotable checkpoint
2. `H15_LATE_RESIDUAL_SELECTIVE_v2_UNDERAMP_ALIGN`
   - result:
     - no promotion
     - no continuation
     - selective late-residual route closed for this chain
3. `H15_MAINLINE_TAIL_CALIB_v1`
   - result:
     - no promotion
     - closest checkpoint of the whole chain
4. `H15_MAINLINE_TAIL_CALIB_v2`
   - only allowed conservative follow-up
   - result:
     - no promotion
     - did not improve on `v1`

## Strongest Mainline Evidence

- Nearest candidate in the whole chain:
  - `H15_MAINLINE_TAIL_CALIB_v1`
  - same-tool recalc `best_by_structured`
    - `rmse_steer=0.5192`
    - `abs_tail_last_0p5s.rmse_steer=0.6405`
    - `late_peak_recall=0.7559`
    - `prefix_1p0s.rmse_steer=0.4464`
    - `head_rmse_steer=0.2929`
    - `response_onset_delay_mae_sec=0.1129`
    - `first_reversal_time_mae_sec=0.3808`
    - `reversal_count_exact_match_rate=0.5455`
    - `strong_pos.tail_amp_ratio_pred_over_gt=0.6549`
    - `strong_pos.tail_flatness_rate=0.5000`
- Read:
  - eight hard-gate items passed
  - only the reversal pair failed
  - but it still did not satisfy the unified promotion rule, so no confirm run was allowed

## Why The Chain Closes

- Selective late-residual route:
  - produced mechanism evidence
  - did not produce a promotable mainline checkpoint
  - failed continuation, so `Task 2` was not allowed
- Mainline tail-calib route:
  - `v1` proved the problem can be pulled back into the main output objective
  - `v2` tested the one allowed small reversal-local correction
  - `v2` re-opened the old target-bucket collapse instead of improving the `v1` near-pass
- Therefore this chain has already exhausted the allowed local repair budget:
  - no more selective `v3/v4/...`
  - no more mainline tail-calib `v3/v4/...`

## Closure Statement

- Late residual / selective route has provided useful mechanism evidence:
  - separability exists
  - some detector alignment can be learned
  - but it is not the mainline solution
- Mainline tail calibration is the closest route in this chain:
  - it can recover broad fit / tail / prefix / head / onset balance
  - but it still does not produce a promotion-safe checkpoint under the unified gate
- Next step should be a higher-level re-audit of the task / supervision / structure tradeoff, not another local patch on:
  - strong-pos guardrails
  - late residual actuation
  - first-reversal local weighting

## Artifacts

- `H15_LATE_RESIDUAL_SELECTIVE_v2_UNDERAMP_ALIGN` summary:
  - `03_results/tmp/overnight_h15_20260423/h15_late_residual_selective_v2_underamp_align/TRAIN_V5_4_STATECOND_REV_20260423_235251/H15_LATE_RESIDUAL_SELECTIVE_v2_UNDERAMP_ALIGN_summary.md`
- `H15_MAINLINE_TAIL_CALIB_v1` summary:
  - `03_results/tmp/overnight_h15_20260423/h15_mainline_tail_calib_v1/TRAIN_V5_4_STATECOND_REV_20260424_003250/H15_MAINLINE_TAIL_CALIB_v1_summary.md`
- `H15_MAINLINE_TAIL_CALIB_v2` summary:
  - `03_results/tmp/overnight_h15_20260423/h15_mainline_tail_calib_v2/TRAIN_V5_4_STATECOND_REV_20260424_010417/H15_MAINLINE_TAIL_CALIB_v2_summary.md`
