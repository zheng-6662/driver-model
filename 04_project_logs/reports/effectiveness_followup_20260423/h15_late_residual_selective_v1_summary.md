# `H15_LATE_RESIDUAL_SELECTIVE_v1` Summary

- Date:
  - `2026-04-23`
- Status:
  - `not a new keeper / strongest selectivity probe so far`
- Smoke preflight directory:
  - `03_results/tmp/effectiveness_followup_20260423/h15_late_residual_selective_v1_smoke/TRAIN_V5_4_STATECOND_REV_20260423_213621`
- Full run directory:
  - `03_results/tmp/effectiveness_followup_20260423/h15_late_residual_selective_v1/TRAIN_V5_4_STATECOND_REV_20260423_214210`

## Purpose

- Continue the late-residual line after `H15_LATE_RESIDUAL_HEAD_v1` showed that the late path is active but still not selective enough.
- Keep the base `H15_AC_CF_HLF_v1` bundle and add:
  - a selective late residual gate
  - an explicit `strong_pos` tail-guard loss
  - stronger structured-score protection against collapse checkpoints
- Test whether stronger late-residual selectivity can repair the dangerous `strong_pos` late-tail failure without reopening optimizer or width sweeps.

## Final env / config summary

- Inherited from the previous `1.5 s` anti-collapse line:
  - `DRIVER_MODEL_FUTURE_SEC=1.5`
  - `DRIVER_MODEL_STEER_COARSE_FINE=1`
  - `DRIVER_MODEL_HARD_LATE_FINE=1`
  - `DRIVER_MODEL_HARD_LATE_START_SEC=1.00`
  - `DRIVER_MODEL_HARD_TAIL_START_SEC=1.00`
  - `DRIVER_MODEL_W_HARD_LATE_FINE=0.10`
  - `DRIVER_MODEL_W_FINE_DC=0.01`
  - `DRIVER_MODEL_W_TREND_COARSE=0.10`
  - `DRIVER_MODEL_PHASE_ADAPTIVE_TREND=0`
  - `DRIVER_MODEL_STRONG_POS_GATE=0`
  - `DRIVER_MODEL_LATE_REV_GATE=0`
  - `DRIVER_MODEL_W_FIRSTREV_LOCAL=0.0`
- Late-residual selective branch enabled:
  - `ENABLE_LATE_RESIDUAL_HEAD=1`
  - `LATE_RESIDUAL_START_SEC=1.00`
  - `W_LATE_RESIDUAL=0.10`
  - `ENABLE_LATE_RESIDUAL_SELECTIVE_GATE=1`
  - `LATE_RESIDUAL_GATE_FLOOR=0.08`
  - `LATE_RESIDUAL_GATE_BOOST=1.35`
  - `LATE_RESIDUAL_GATE_PROB_CENTER=0.55`
  - `LATE_RESIDUAL_GATE_RAMP_POWER=1.5`
  - `LATE_RESIDUAL_STRONG_BOOST=2.0`
  - `LATE_RESIDUAL_UNDERAMP_BOOST=1.5`
  - `LATE_RESIDUAL_FOCUS_MAX=5.0`
  - `W_STRONG_POS_TAIL_GUARD=0.12`
  - `STRONG_POS_TAIL_GUARD_START_SEC=1.00`
  - `STRONG_POS_TAIL_RATIO_FLOOR=0.60`
  - `STRONG_POS_TAIL_FLAT_FRAC=0.12`
  - `STRONG_POS_TAIL_GUARD_FLATNESS_WEIGHT=0.75`
  - `ENABLE_STRONG_POS_STRUCTURED_GUARD=1`
  - `STRUCT_STRONG_POS_AMP_FLOOR=0.60`
  - `STRUCT_STRONG_POS_FLATNESS_MAX=0.60`
  - `STRUCT_STRONG_POS_AMP_WEIGHT=1.2`
  - `STRUCT_STRONG_POS_FLATNESS_WEIGHT=0.8`
- Unchanged full-run defaults from the `H15` line:
  - `BATCH_SIZE=64`
  - `EPOCHS=40`
  - `OPTIMIZER=adam`
  - `LR=1e-3`
  - `WEIGHT_DECAY=0`
  - `INPUT_PIPELINE_VERSION=fixed_v20260421`
  - protocol-safe split unchanged
  - stable GPU path unchanged
  - live script path used manual coarse upsample

## Minimal code changes and validation

- Integrated the user-provided GPT Pro patch into the current modular source tree:
  - `02_code/final_code/model/training/v58_modular/config.py`
  - `02_code/final_code/model/training/v58_modular/modeling.py`
  - `02_code/final_code/model/training/v58_modular/losses.py`
  - `02_code/final_code/model/training/v58_modular/metrics.py`
  - `02_code/final_code/model/training/v58_modular/evaluation.py`
  - `02_code/final_code/model/training/v58_modular/train.py`
  - `02_code/tools/recalc_v58_checkpoint_with_current_metrics.py`
- Validation before the real run:
  - `py_compile` passed for the wrapper, touched modular modules, and recalc tool
  - direct forward / loss preflight passed
  - aux outputs now include:
    - `late_residual_selective_scale`
    - `strong_pos_gate_prob`
  - `compute_total_task_loss()` now returns the extra strong-pos tail-guard term
  - a `2`-epoch smoke run completed end to end
  - smoke same-tool recalc also completed successfully
- Same-tool recalc still needed the known modular helper path:
  - set `PYTHONPATH=02_code/final_code/model/training`
  - use `tmp/recalc_v58_metrics_shim_20260423.py` as the metrics source
  - this is still a tooling-path closure issue, not a model-logic failure

## Run closure

- Smoke preflight:
  - completed successfully in `0.10 min`
- Full run:
  - completed on the first training attempt
  - `40` epochs
  - total training time `11.69 min`
- Built-in evaluation at the end of training completed successfully and exported:
  - `figures/test_metrics.json`
  - `figures/test_metrics_by_reversal.json`
  - `figures/test_metrics_reversal_structure.json`
  - `figures/test_late_residual_metrics.json`
  - `figures/test_state_dump.csv`
- Same-tool recalc closure completed for:
  - `best_by_loss`
  - `best_by_structured`
- Recalc outputs:
  - `figures/recalc_best_by_loss_summary.json`
  - `figures/recalc_best_by_loss_cases.csv`
  - `figures/recalc_best_by_structured_summary.json`
  - `figures/recalc_best_by_structured_cases.csv`

## Headline comparison

| Run | Selection | `rmse_steer` | `abs_tail_last_0p5s.rmse_steer` | `late_peak_recall` | `strong_pos.tail_amp_ratio_pred_over_gt` | `strong_pos.tail_flatness_rate` | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_fixed_input` | `best_by_structured` | 0.5559 | 0.7171 | 0.6496 | 1.3490 | 0.2105 | live fit/tail keeper |
| `H15` | `best_by_structured` | 0.4930 | 0.6022 | 0.6355 | 0.2687 | 1.0000 | original strong-pos collapse |
| `H15_AC_CF_HLF_v1` | `best_by_loss` | 0.5063 | 0.6323 | 0.5786 | 0.5141 | 0.3750 | partial anti-collapse, late-peak still weak |
| `H15_LATE_RESIDUAL_HEAD_v1` | `best_by_loss` | 0.4954 | 0.6284 | 0.6522 | 0.3163 | 1.0000 | fit/tail hold, strong-pos still collapses |
| `H15_LATE_RESIDUAL_HEAD_v1` | `best_by_structured` | 0.5474 | 0.6868 | 0.6656 | 0.4904 | 0.5000 | previous late-residual control checkpoint |
| `H15_LATE_RESIDUAL_SELECTIVE_v1` | `best_by_loss` | 0.5356 | 0.6745 | 0.6756 | 0.4947 | 0.7500 | partial repair, but still below guardrail |
| `H15_LATE_RESIDUAL_SELECTIVE_v1` | `best_by_structured` | 0.6379 | 0.7319 | 0.6187 | 1.4833 | 0.3750 | strong-pos repaired, but overall fit/head regress badly |

## Checkpoint interpretation

- `best_by_loss`
  - no longer behaves like the raw collapse checkpoint from `H15_LATE_RESIDUAL_HEAD_v1`
  - compared with `H15_LATE_RESIDUAL_HEAD_v1 best_by_loss`, it materially repairs the dangerous bucket:
    - `strong_pos.tail_amp_ratio_pred_over_gt: 0.3163 -> 0.4947`
    - `strong_pos.tail_flatness_rate: 1.0000 -> 0.7500`
    - `late_peak_recall: 0.6522 -> 0.6756`
  - but the repair still does not clear the required floor:
    - `strong_pos.tail_amp_ratio_pred_over_gt = 0.4947 < 0.60`
    - `strong_pos.tail_flatness_rate = 0.7500 > 0.60`
  - and it gives back too much fit / tail relative to the stronger late-residual and `H15` checkpoints:
    - `rmse_steer=0.5356`
    - `abs_tail_last_0p5s.rmse_steer=0.6745`
  - practical read:
    - the selective branch now does repair the target bucket on the fit-preserving side
    - but not strongly enough yet to become a keeper
- `best_by_structured`
  - proves that the stronger selective branch can clear the `strong_pos` amplitude floor on test:
    - `strong_pos.tail_amp_ratio_pred_over_gt=1.4833`
    - `strong_pos.tail_flatness_rate=0.3750`
  - however it gets there in a globally damaged regime:
    - `rmse_steer=0.6379`
    - `abs_tail_last_0p5s.rmse_steer=0.7319`
    - `prefix_1p0s.rmse_steer=0.5853`
    - `head_rmse_steer=0.5125`
    - `response_onset_delay_mae_sec=0.6270`
    - `smooth_trend_corr_mean=0.4561`
    - `coarse_segment_sign_match_rate=0.2967`
  - practical read:
    - this is a mechanism proof checkpoint, not a keeper candidate
    - it shows the branch can force strong-pos repair, but only by drifting too far on the main task

## Late residual diagnostics

- Built-in `test_late_residual_metrics.json` confirms that the new head is active and that the new gate is selective by bucket:
  - `mean_abs=0.0304`
  - `peak_abs_mean=0.1557`
  - `nonzero_rate=1.0`
  - `gate_prob_by_bucket.strong_pos=0.6908`
  - `gate_prob_by_bucket.non_strong=0.1483`
  - `strong_pos_vs_non_strong_ratio.gate_prob=4.6584`
  - `gate_mean_by_bucket.strong_pos=0.4124`
  - `gate_mean_by_bucket.non_strong=0.1233`
  - `strong_pos_vs_non_strong_ratio.gate_mean=3.3443`
- But the correction is still not well aligned to the true under-amplitude deficit:
  - `strong_pos_mean_abs=0.0250 < non_strong_mean_abs=0.0305`
  - `strong_pos_peak_abs=0.2264 > non_strong_peak_abs=0.1546`
  - `tail_amp_gain_on_strong_pos.mean_ratio_gain_over_gt=0.2207`
  - `correlation_with_tail_under_amp.late_residual_mean_abs=-0.1185`
  - `correlation_with_tail_under_amp.late_residual_peak_abs=-0.1251`
  - `correlation_with_tail_under_amp.gate_prob=-0.1335`
  - `correlation_with_tail_under_amp.gate_scale_mean=-0.1421`
- Practical read:
  - the selective gate is no longer missing
  - the remaining bottleneck is narrower:
    - bucket-level selectivity exists
    - failure-mechanism alignment is still weak
    - the branch is closer to `strong_pos` label selectivity than to true late-tail under-amplitude selectivity

## Success criteria check

| Criterion | Target | `best_by_loss` | `best_by_structured` | Result |
| --- | --- | ---: | ---: | --- |
| `strong_pos.tail_amp_ratio_pred_over_gt` | `>= 0.60` | 0.4947 | 1.4833 | mixed / fail as keeper |
| `strong_pos.tail_flatness_rate` | `<= 0.60` | 0.7500 | 0.3750 | mixed / fail as keeper |
| `abs_tail_last_0p5s.rmse_steer` | `<= 0.66` | 0.6745 | 0.7319 | fail |
| `rmse_steer` | `<= 0.53` | 0.5356 | 0.6379 | fail |
| `late_peak_recall` | `>= 0.62` | 0.6756 | 0.6187 | mixed |
| `prefix_1p0s.rmse_steer` | `<= 0.50` | 0.4504 | 0.5853 | mixed |
| `response_onset_delay_mae_sec` | `<= 0.25` | 0.1449 | 0.6270 | mixed |

## Final judgment

- `H15_LATE_RESIDUAL_SELECTIVE_v1` is not a new keeper.
- Compared with the earlier late-residual branch, it moves the story forward in a real way:
  - `best_by_loss` is no longer the old hard collapse shape
  - `best_by_structured` proves that stronger selectivity can repair `strong_pos`
- But the branch still splits into two bad endpoints:
  - `best_by_loss` keeps reasonable fit / tail, but under-repairs `strong_pos`
  - `best_by_structured` repairs `strong_pos`, but over-regresses the main task
- This means the next problem is no longer "does late residual work at all?"
- The next problem is:
  - can the selective late-residual path align to true late-tail under-amplitude cases strongly enough
  - while protecting fit / tail / prefix / onset from collateral damage

## Decision boundary

- Keep the live keeper split unchanged:
  - Run A full `best_by_structured` = response-structure anchor
  - `baseline_fixed_input` full `best_by_structured` = fit / tail keeper
- Do not promote `H15_LATE_RESIDUAL_SELECTIVE_v1` as-is.
- Do not reopen optimizer / width / bridge sweeps as the explanation for this branch.
- If the late-residual direction is continued later:
  - stay on the selective late-residual path
  - keep changes limited to `v58_modular/` and the recalc tool
  - treat the current run as a bracketed failure boundary:
    - `best_by_loss` = fit-preserving but under-repaired
    - `best_by_structured` = strong-pos-repaired but over-regressed
  - the next version must land between those two endpoints
