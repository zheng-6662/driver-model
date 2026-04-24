# `H15_LATE_RESIDUAL_HEAD_v1` Summary

- Date:
  - `2026-04-23`
- Status:
  - `not a new keeper / needs control if the late-residual direction continues`
- Full run directory:
  - `03_results/tmp/effectiveness_followup_20260423/h15_late_residual_head_v1/TRAIN_V5_4_STATECOND_REV_20260423_163844`
- Smoke preflight directory:
  - `03_results/tmp/effectiveness_followup_20260423/h15_late_residual_head_v1_smoke/TRAIN_V5_4_STATECOND_REV_20260423_163432`

## Purpose

- Execute the GPT Pro fallback branch after `H15_AC_CF_HLF_v1` closed as a no-go.
- Keep the real `1.5 s` fit / tail upside from `H15`.
- Add a minimal late residual steer head that is only active on `t >= 1.0 s`.
- Test whether extra late-slice capacity can repair the dangerous `strong_pos` late-tail collapse without reopening optimizer or width sweeps.

## Final env / config summary

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
- `DRIVER_MODEL_LATE_RESIDUAL_HEAD=1`
- `DRIVER_MODEL_LATE_RESIDUAL_START_SEC=1.00`
- `DRIVER_MODEL_W_LATE_RESIDUAL=0.10`
- Full-run defaults preserved from the `H15` line:
  - `BATCH_SIZE=64`
  - `EPOCHS=40`
  - `OPTIMIZER=adam`
  - `LR=1e-3`
  - `WEIGHT_DECAY=0`
  - `INPUT_PIPELINE_VERSION=fixed_v20260421`
  - protocol-safe split unchanged
  - stable GPU path unchanged
  - `ENABLE_MANUAL_COARSE_UPSAMPLE=1`

## Minimal code changes and validation

- Integrated the user-provided GPT Pro patch into the current modular source tree:
  - `02_code/final_code/model/training/v58_modular/config.py`
  - `02_code/final_code/model/training/v58_modular/modeling.py`
  - `02_code/final_code/model/training/v58_modular/losses.py`
  - `02_code/final_code/model/training/v58_modular/train.py`
  - `02_code/final_code/model/training/v58_modular/evaluation.py`
  - `02_code/tools/recalc_v58_checkpoint_with_current_metrics.py`
- Validation before the real run:
  - `py_compile` passed for the wrapper, touched modular modules, and recalc tool
  - a direct forward / loss preflight confirmed:
    - late residual aux fields are emitted
    - `compute_total_task_loss()` now returns the extra late-residual term
  - a `2`-epoch smoke run completed end to end and exported:
    - checkpoints
    - history
    - `test_late_residual_metrics.json`
- Same-tool recalc still needed the known modular helper path:
  - set `PYTHONPATH=02_code/final_code/model/training`
  - use `tmp/recalc_v58_metrics_shim_20260423.py` as the metric source
  - this was a tooling-path closure issue, not a model-logic failure

## Run closure

- Smoke preflight:
  - completed successfully in `0.10 min`
  - confirmed that training, checkpointing, built-in evaluation, and the new late-residual diagnostics all run end to end
- Full run:
  - completed on the first training attempt
  - `40` epochs
  - total training time `13.29 min`
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
| `H15_LATE_RESIDUAL_HEAD_v1` | `best_by_loss` | 0.4954 | 0.6284 | 0.6522 | 0.3163 | 1.0000 | fit/tail/late-peak hold, strong-pos still collapses |
| `H15_LATE_RESIDUAL_HEAD_v1` | `best_by_structured` | 0.5474 | 0.6868 | 0.6656 | 0.4904 | 0.5000 | collapse softened materially, but still below amplitude floor |

## Checkpoint interpretation

- `best_by_loss`
  - holds the attractive average metrics:
    - `rmse_steer=0.4954`
    - `abs_tail_last_0p5s.rmse_steer=0.6284`
    - `late_peak_recall=0.6522`
  - but it is still an explicit collapse checkpoint on the dangerous bucket:
    - `strong_pos.tail_amp_ratio_pred_over_gt=0.3163`
    - `strong_pos.tail_flatness_rate=1.0000`
  - conclusion:
    - the late residual head does not inherently destroy fit / tail / late-peak
    - but the current loss / routing still allows the model to win global metrics while leaving `strong_pos` under-served
- `best_by_structured`
  - is the only structurally meaningful checkpoint from this run
  - versus old `H15` `best_by_structured`, it improves the failure bucket materially:
    - `strong_pos.tail_amp_ratio_pred_over_gt: 0.2687 -> 0.4904`
    - `strong_pos.tail_flatness_rate: 1.0000 -> 0.5000`
    - `late_peak_recall: 0.6355 -> 0.6656`
  - however it still misses the explicit promotion floor:
    - amplitude target was `>= 0.60`
    - actual value is `0.4904`
  - and it gives back too much fit / tail versus the stronger `H15` / `H15_AC_CF_HLF_v1 best_by_loss` checkpoints:
    - `rmse_steer=0.5474`
    - `abs_tail_last_0p5s.rmse_steer=0.6868`

## Late residual diagnostics

- Built-in `test_late_residual_metrics.json` confirms that the new head is active and non-trivial on the late slice:
  - `mean_abs=0.0521`
  - `peak_abs_mean=0.1991`
  - `nonzero_rate=1.0`
- On the `best_by_loss` evaluation used by the built-in export, the head is only mildly more active on `strong_pos` than elsewhere:
  - `strong_pos_mean_abs=0.0629`
  - `non_strong_mean_abs=0.0520`
  - `strong_pos_peak_abs=0.3091`
  - `non_strong_peak_abs=0.1974`
- Interpretation:
  - the added late capacity is being used
  - but it is not yet selective enough to repair the rare `strong_pos` bucket robustly

## Success criteria check

| Criterion | Target | `best_by_loss` | `best_by_structured` | Result |
| --- | --- | ---: | ---: | --- |
| `strong_pos.tail_amp_ratio_pred_over_gt` | `>= 0.60` | 0.3163 | 0.4904 | fail |
| `strong_pos.tail_flatness_rate` | `<= 0.60` | 1.0000 | 0.5000 | mixed / fail |
| `abs_tail_last_0p5s.rmse_steer` | `<= 0.66` | 0.6284 | 0.6868 | mixed |
| `rmse_steer` | `<= 0.53` | 0.4954 | 0.5474 | mixed |
| `late_peak_recall` | `>= 0.62` | 0.6522 | 0.6656 | pass |

## Final judgment

- `H15_LATE_RESIDUAL_HEAD_v1` is not a new keeper.
- As a keeper candidate, the run fails because neither saved checkpoint simultaneously holds:
  - fit / tail
  - late peak
  - and the required `strong_pos` late-tail amplitude floor
- As a mechanism probe, the run is still informative:
  - `best_by_structured` is clearly less collapsed than old `H15`
  - this is the cleanest evidence so far that extra late-slice capacity is directionally useful
- But the current implementation still falls short of a promotable branch:
  - `best_by_loss` keeps global metrics while collapsing `strong_pos`
  - `best_by_structured` repairs the collapse partially, but not enough
- No manual strong-pos plot review was needed in this round because neither checkpoint met the promotion criteria directly.

## Decision boundary

- Keep the live keeper split unchanged:
  - Run A full `best_by_structured` = response-structure anchor
  - `baseline_fixed_input` full `best_by_structured` = fit / tail keeper
- Do not promote `H15_LATE_RESIDUAL_HEAD_v1` as-is.
- If the late-residual direction is continued later:
  - only `best_by_structured` from this run is worth treating as the control checkpoint
  - `best_by_loss` should be treated as a collapse case, not as the branch default
- Do not reopen optimizer / width sweeps as the explanation for this branch.

