# GPTPro Handoff: H15 Overnight Experiment Chain Closure

Date: 2026-04-24

This handoff summarizes the overnight `modular V5.8 / H15` chain and asks for a higher-level next-step analysis. Please do not interpret the request as permission to propose broad optimizer / width / bridge / generic loss sweeps. The purpose is to decide what the evidence means and whether a higher-level task / supervision / selector re-audit is needed.

## 1. Project Goal And Hard Gate

Goal:

- Find a protocol-safe, reproducible, defensible `mainline` checkpoint.
- The goal is not to keep locally repairing `strong_pos` forever.
- `strong_pos` is a guardrail, not the project itself.

Unified hard promotion gate:

- `rmse_steer <= 0.54`
- `abs_tail_last_0p5s.rmse_steer <= 0.68`
- `late_peak_recall >= 0.64`
- `prefix_1p0s.rmse_steer <= 0.50`
- `head_rmse_steer <= 0.50`
- `response_onset_delay_mae_sec <= 0.25`
- `first_reversal_time_mae_sec <= 0.35`
- `reversal_count_exact_match_rate >= 0.58`
- `strong_pos.tail_amp_ratio_pred_over_gt >= 0.60`
- `strong_pos.tail_flatness_rate <= 0.60`

Continuation gate:

- `rmse_steer <= 0.55`
- `abs_tail_last_0p5s.rmse_steer <= 0.69`
- `late_peak_recall >= 0.64`
- `prefix_1p0s.rmse_steer <= 0.52`
- `response_onset_delay_mae_sec <= 0.35`
- `strong_pos.tail_amp_ratio_pred_over_gt >= 0.55`
- `strong_pos.tail_flatness_rate <= 0.60`
- plus underamp correlation improvement when relevant.

## 2. Important Caveat: Actual Task 3 Config

The launched Task 3 environment attempted to keep the prior H15 coarse-fine / hard-late bundle, but `run_config.json` shows those flags were actually off:

- `ENABLE_STEER_COARSE_FINE=false`
- `ENABLE_MANUAL_COARSE_UPSAMPLE=false`
- `ENABLE_HARD_LATE_FINE=false`

Reason:

- The modular config reads:
  - `DRIVER_MODEL_STEER_COARSE_FINE`
  - `DRIVER_MODEL_MANUAL_COARSE_UPSAMPLE`
  - `DRIVER_MODEL_HARD_LATE_FINE`
- The inherited launch env used some older-style names:
  - `DRIVER_MODEL_ENABLE_STEER_COARSE_FINE`
  - `DRIVER_MODEL_ENABLE_MANUAL_COARSE_UPSAMPLE`
  - `DRIVER_MODEL_ENABLE_HARD_LATE_FINE`

So Task 3 should be interpreted as a simpler mainline-tail-calibration test, not as the intended full `H15_AC_CF_HLF` bundle plus mainline tail calibration.

This matters because the simpler Task 3 still produced the closest checkpoint of the chain. Please explicitly assess whether a corrected-flag rerun is a legitimate configuration-closure check or whether the current evidence should already stop local `H15` patching.

## 3. Chain Executed

### Task 0: Retrospective Scan

Scanned saved checkpoint history for:

- `H15_LATE_RESIDUAL_HEAD_v1`
- `H15_LATE_RESIDUAL_SELECTIVE_v1`

Result:

- No hidden promotable checkpoint found.
- Proceeded to Task 1.

### Task 1: `H15_LATE_RESIDUAL_SELECTIVE_v2_UNDERAMP_ALIGN`

Goal:

- Keep late residual route, but reframe it as:
  - underamp detector
  - bounded corrector
  - promotion-aware selector

Run:

- `03_results/tmp/overnight_h15_20260423/h15_late_residual_selective_v2_underamp_align/TRAIN_V5_4_STATECOND_REV_20260423_235251`

Outcome:

- No hard promotion.
- No continuation.
- Task 2 was skipped by rule.
- Selective late-residual route closed for this chain.

Same-tool recalc, `best_by_promotion`:

- `rmse_steer=0.5075`
- `abs_tail_last_0p5s.rmse_steer=0.6082`
- `late_peak_recall=0.5351`
- `prefix_1p0s.rmse_steer=0.4488`
- `head_rmse_steer=0.2867`
- `response_onset_delay_mae_sec=0.0996`
- `first_reversal_time_mae_sec=0.3153`
- `reversal_count_exact_match_rate=0.6553`
- `strong_pos.tail_amp_ratio_pred_over_gt=0.5207`
- `strong_pos.tail_flatness_rate=0.7500`
- `hard_pass=false`
- `continuation_pass=false`

Same-tool recalc, `best_by_structured`:

- `rmse_steer=0.4768`
- `abs_tail_last_0p5s.rmse_steer=0.5731`
- `late_peak_recall=0.6488`
- `prefix_1p0s.rmse_steer=0.4205`
- `head_rmse_steer=0.2797`
- `response_onset_delay_mae_sec=0.0936`
- `first_reversal_time_mae_sec=0.3057`
- `reversal_count_exact_match_rate=0.6193`
- `strong_pos.tail_amp_ratio_pred_over_gt=0.2637`
- `strong_pos.tail_flatness_rate=0.7500`
- `hard_pass=false`
- `continuation_pass=false`

Mechanism read:

- Detector-underamp alignment improved from weak / negative to modestly positive:
  - built-in promotion-root `corr(gate_prob, underamp_severity)=0.2127`
  - built-in structured `corr(gate_prob, underamp_severity)=0.2862`
- But actuation did not become useful repair:
  - `tail_amp_gain_on_strong_pos.mean_ratio_gain_over_gt=-0.0064` on promotion-root eval
  - `tail_amp_gain_on_strong_pos.mean_ratio_gain_over_gt=-0.0321` on structured eval
  - underamp gate probability on `strong_pos` remained below `non_strong`
- Separate `strong_pos` classifier remained bucket-selective, so missing separability was not the main issue.

Interpretation:

- Selective late residual gave mechanism evidence.
- It did not produce a promotion-safe mainline checkpoint.
- Failure is not simply "make the gate stronger"; it is actuation / objective alignment.

### Task 3: `H15_MAINLINE_TAIL_CALIB_v1`

Goal:

- Stop treating side late residual as the main solution.
- Pull the problem into the main output objective.
- Add direct late-tail amplitude / anti-flatness auxiliary on `y_hat`.
- Preserve prefix / head / onset.

Run:

- `03_results/tmp/overnight_h15_20260423/h15_mainline_tail_calib_v1/TRAIN_V5_4_STATECOND_REV_20260424_003250`

Actual route:

- late residual disabled
- selective gate disabled
- underamp detector disabled
- strong-pos gate disabled
- mainline tail calibration enabled
- strong-pos tail guard enabled
- prefix / head / onset protection enabled
- caveat: coarse-fine / hard-late flags were actually off, as noted above

Validation selector result:

- `best_by_promotion` and `best_by_mainline_balance` both selected epoch 16.
- Validation-side checkpoint did not pass promotion or continuation.

Same-tool recalc, `best_by_promotion`:

- `rmse_steer=0.4748`
- `abs_tail_last_0p5s.rmse_steer=0.5843`
- `late_peak_recall=0.6689`
- `prefix_1p0s.rmse_steer=0.4092`
- `head_rmse_steer=0.2709`
- `response_onset_delay_mae_sec=0.1080`
- `first_reversal_time_mae_sec=0.4953`
- `reversal_count_exact_match_rate=0.5492`
- `strong_pos.tail_amp_ratio_pred_over_gt=0.6860`
- `strong_pos.tail_flatness_rate=0.6250`
- `hard_pass=false`
- `continuation_pass=false`

Same-tool recalc, `best_by_structured`:

- `rmse_steer=0.5192`
- `abs_tail_last_0p5s.rmse_steer=0.6405`
- `late_peak_recall=0.7559`
- `prefix_1p0s.rmse_steer=0.4464`
- `head_rmse_steer=0.2929`
- `response_onset_delay_mae_sec=0.1130`
- `first_reversal_time_mae_sec=0.3808`
- `reversal_count_exact_match_rate=0.5455`
- `strong_pos.tail_amp_ratio_pred_over_gt=0.6549`
- `strong_pos.tail_flatness_rate=0.5000`
- `hard_pass=false`
- `continuation_pass=true`

Interpretation:

- This is the closest checkpoint in the whole chain.
- It passes eight of the ten hard-gate items on same-tool recalc.
- It fails only:
  - `first_reversal_time_mae_sec`
  - `reversal_count_exact_match_rate`
- It did not pass hard promotion, so no confirm run was allowed.
- Because it was close, one conservative follow-up was allowed.

### Task 3 Follow-Up: `H15_MAINLINE_TAIL_CALIB_v2`

Goal:

- Use the one allowed conservative follow-up to test whether a small reversal-local correction can convert the `v1` near-pass into promotion.

Run:

- `03_results/tmp/overnight_h15_20260423/h15_mainline_tail_calib_v2/TRAIN_V5_4_STATECOND_REV_20260424_010417`

Only material change from `v1`:

- `W_FIRSTREV_LOCAL=0.10`
- `FIRSTREV_LOCAL_RADIUS=12`

Same-tool recalc, `best_by_promotion`:

- `rmse_steer=0.5068`
- `abs_tail_last_0p5s.rmse_steer=0.6187`
- `late_peak_recall=0.6020`
- `prefix_1p0s.rmse_steer=0.4402`
- `head_rmse_steer=0.3197`
- `response_onset_delay_mae_sec=0.1435`
- `first_reversal_time_mae_sec=0.4633`
- `reversal_count_exact_match_rate=0.6061`
- `strong_pos.tail_amp_ratio_pred_over_gt=1.0128`
- `strong_pos.tail_flatness_rate=0.3750`
- `hard_pass=false`
- `continuation_pass=false`

Same-tool recalc, `best_by_structured`:

- `rmse_steer=0.4906`
- `abs_tail_last_0p5s.rmse_steer=0.6117`
- `late_peak_recall=0.7692`
- `prefix_1p0s.rmse_steer=0.4170`
- `head_rmse_steer=0.2725`
- `response_onset_delay_mae_sec=0.1130`
- `first_reversal_time_mae_sec=0.3535`
- `reversal_count_exact_match_rate=0.5833`
- `strong_pos.tail_amp_ratio_pred_over_gt=0.2720`
- `strong_pos.tail_flatness_rate=1.0000`
- `hard_pass=false`
- `continuation_pass=false`

Interpretation:

- `v2` did not improve on `v1`.
- It re-created a bracketed failure:
  - `best_by_promotion` keeps strong-pos amplitude and flatness, but loses late-peak and first-reversal timing.
  - `best_by_structured` almost fixes reversal, but collapses the exact strong-pos target bucket.
- Therefore no more local `tail_calib_v3/v4/...` is allowed in this chain.

## 4. Current Best Read

No promotable checkpoint was found.

Closest checkpoint:

- `H15_MAINLINE_TAIL_CALIB_v1`
- same-tool recalc `best_by_structured`
- It is not promotable, but it is the strongest mainline evidence from the chain.

Main conclusion:

- Selective late residual is mechanism evidence, not the mainline solution.
- Mainline tail calibration is a better direction than side residual repair.
- But the remaining failure is not solved by one more local reversal loss; the `v2` follow-up trades reversal recovery against strong-pos tail safety.

## 5. Questions For GPTPro

Please analyze the following, in order:

1. Given the actual Task 3 config caveat, should the next legal action be a corrected-flag rerun of `H15_MAINLINE_TAIL_CALIB_v1` with the intended `DRIVER_MODEL_STEER_COARSE_FINE / DRIVER_MODEL_MANUAL_COARSE_UPSAMPLE / DRIVER_MODEL_HARD_LATE_FINE` flags, or should the chain remain closed because this would count as another local variant?

2. If a corrected-flag rerun is justified, what exact single configuration should be used, and what stopping rule should apply?

3. If a corrected-flag rerun is not justified, what higher-level re-audit should come next?
   - supervision target?
   - reversal metric definition?
   - validation selector mismatch?
   - sparse `strong_pos` bucket robustness?
   - task framing / anchor logic?

4. How should we interpret the fact that `v1 best_by_structured` passes eight hard-gate items on same-tool recalc but is not selected as a promotion pass?
   - Is this a selector alignment issue?
   - Or is it just valid evidence that the remaining reversal pair is still mission-critical?

5. What is the next defensible mainline checkpoint policy?
   - Keep `baseline_fixed_input best_by_structured` as live keeper?
   - Keep `H15_MAINLINE_TAIL_CALIB_v1 best_by_structured` as a non-promoted near-pass reference?
   - Or promote nothing and restart from higher-level task analysis?

Please avoid recommending:

- optimizer sweep
- width sweep
- bridge sweep
- generic loss matrix
- more late-residual / gate-heavy variants
- more `strong_pos`-only repair variants

## 6. Pointers

Closure report:

- `04_project_logs/reports/effectiveness_followup_20260423/h15_overnight_chain_closure_20260424.md`

Version summaries:

- `03_results/tmp/overnight_h15_20260423/h15_late_residual_selective_v2_underamp_align/TRAIN_V5_4_STATECOND_REV_20260423_235251/H15_LATE_RESIDUAL_SELECTIVE_v2_UNDERAMP_ALIGN_summary.md`
- `03_results/tmp/overnight_h15_20260423/h15_mainline_tail_calib_v1/TRAIN_V5_4_STATECOND_REV_20260424_003250/H15_MAINLINE_TAIL_CALIB_v1_summary.md`
- `03_results/tmp/overnight_h15_20260423/h15_mainline_tail_calib_v2/TRAIN_V5_4_STATECOND_REV_20260424_010417/H15_MAINLINE_TAIL_CALIB_v2_summary.md`

Project state:

- `04_project_logs/references/current-state.md`
- `04_project_logs/reports/progress/daily/2026-04-23.md`
- `04_project_logs/reports/progress/experiment_registry.md`
