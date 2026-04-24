# `H15_AC_CF_HLF_v1` Summary

- Date:
  - `2026-04-23`
- Status:
  - `fail / no-go after manual strong-pos review`
- Run directory:
  - `03_results/tmp/effectiveness_followup_20260423/h15_ac_cf_hlf_v1/TRAIN_V5_4_STATECOND_REV_20260423_131956`

## Purpose

- Test the single approved anti-collapse follow-up on top of `H15`.
- Keep the real `1.5 s` fit/tail gain from old `H15`.
- Repair the dangerous late `strong_pos` collapse by:
  - enabling steer coarse-fine decomposition
  - enabling hard-late fine residual supervision
  - moving the hard-late / hard-tail window to the true failing slice `1.0 s -> 1.5 s`

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
- `DRIVER_MODEL_W_FIRSTREV_LOCAL=0.0`
- Full-run defaults preserved from the H15 line:
  - `BATCH_SIZE=64`
  - `EPOCHS=40`
  - `OPTIMIZER=adam`
  - `LR=1e-3`
  - `WEIGHT_DECAY=0`
  - `INPUT_PIPELINE_VERSION=fixed_v20260421`
  - protocol-safe split unchanged
  - stable GPU path unchanged
  - `ENABLE_MANUAL_COARSE_UPSAMPLE=1`

## Minimal code changes

- No new hypothesis logic was needed; the requested switches were already wired in `v58_modular/`.
- Required blocker fixes:
  - `02_code/final_code/model/training/v58_modular/train.py`
    - import `has_reversal_np`
    - import `get_rev_aux_target`
  - `02_code/final_code/model/training/v58_modular/evaluation.py`
    - import `unpack_model_output`
    - import `make_state_column_names`
    - import `summarize_state_vector`
- Temporary helper for same-tool recalc:
  - `tmp/recalc_v58_metrics_shim_20260423.py`
  - Needed because the recalc tool loads the metrics module by file path, while `v58_modular/metrics.py` uses relative imports and the wrapper does not re-export underscore helper functions.

## Run closure

- Interrupted attempts kept for audit:
  - `TRAIN_V5_4_STATECOND_REV_20260423_130814`
    - blocked by missing `has_reversal_np` import
  - `TRAIN_V5_4_STATECOND_REV_20260423_131451`
    - blocked by missing `get_rev_aux_target` import
- Final completed attempt:
  - `TRAIN_V5_4_STATECOND_REV_20260423_131956`
  - 40 epochs
  - total training time `12.70 min`
- Built-in plot export failed after training with:
  - `NameError("name 'unpack_model_output' is not defined")`
- Checkpoints were already saved, so closure continued with:
  - minimal `evaluation.py` import fix
  - same-tool recalc for `best_by_loss`
  - same-tool recalc for `best_by_structured`

## Headline comparison

| Run | Selection | `rmse_steer` | `abs_tail_last_0p5s.rmse_steer` | `late_peak_recall` | `strong_pos.tail_amp_ratio_pred_over_gt` | `strong_pos.tail_flatness_rate` | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_fixed_input` | `best_by_structured` | 0.5559 | 0.7171 | 0.6496 | 1.3490 | 0.2105 | live fit/tail keeper |
| `H15` | `best_by_structured` | 0.4930 | 0.6022 | 0.6355 | 0.2687 | 1.0000 | clear strong-pos collapse |
| `H15_AC_CF_HLF_v1` | `best_by_loss` | 0.5063 | 0.6323 | 0.5786 | 0.5141 | 0.3750 | partial anti-collapse, still below target |
| `H15_AC_CF_HLF_v1` | `best_by_structured` | 0.5231 | 0.6472 | 0.5251 | 0.3304 | 0.7500 | explicit no-go |

## Checkpoint interpretation

- `best_by_loss`
  - Better than `baseline_fixed_input` on `rmse_steer` and absolute last-`0.5 s` RMSE.
  - Repairs old `H15` strong-pos flatness materially:
    - amplitude `0.2687 -> 0.5141`
    - flatness `1.0000 -> 0.3750`
  - Still misses the success targets:
    - `strong_pos.tail_amp_ratio_pred_over_gt >= 0.60`
    - `late_peak_recall >= 0.62`
  - Not enough to make `H15` promotable.
- `best_by_structured`
  - Still improves fit/tail versus the baseline keeper.
  - Fails the explicit failure condition:
    - `strong_pos.tail_amp_ratio_pred_over_gt = 0.3304 < 0.50`
  - Also regresses `late_peak_recall` heavily.

## Manual strong-pos review

- Because `best_by_loss` looked borderline on the aggregate strong-pos metrics, eight representative `strong_pos` plots were reviewed manually.
- Review directory:
  - `03_results/tmp/effectiveness_followup_20260423/h15_ac_cf_hlf_v1/TRAIN_V5_4_STATECOND_REV_20260423_131956/figures/strong_pos_review_best_by_loss`
- Review index:
  - `strong_pos_review_index.csv`
- Readout:
  - `3/8` cases remain severe final-tail under-amplitude:
    - `0.161`
    - `0.319`
    - `0.354`
  - `2/8` are only moderate repairs around `0.602`
  - `2/8` are closer but still capped / biased around `0.618`
  - `1/8` is the closest recovery at `0.837`
- Manual review conclusion:
  - the anti-collapse effect is real but not robust
  - the branch still fails as a practical no-go

## Success criteria check

| Criterion | Target | `best_by_loss` | `best_by_structured` | Result |
| --- | --- | ---: | ---: | --- |
| `strong_pos.tail_amp_ratio_pred_over_gt` | `>= 0.60` | 0.5141 | 0.3304 | fail |
| `strong_pos.tail_flatness_rate` | `<= 0.60` | 0.3750 | 0.7500 | mixed / fail |
| `abs_tail_last_0p5s.rmse_steer` | `<= 0.66` | 0.6323 | 0.6472 | pass |
| `rmse_steer` | `<= 0.53` | 0.5063 | 0.5231 | pass |
| `late_peak_recall` | `>= 0.62` | 0.5786 | 0.5251 | fail |

## Final judgment

- `H15_AC_CF_HLF_v1` is a failure / no-go after manual review.
- It improves old `H15` by reducing the pathological flatness of the strong-pos late tail.
- It does not repair enough strong-pos amplitude consistently enough to reopen `H15` as a promotable base.
- The current live keeper split stays unchanged:
  - Run A full `best_by_structured` = response-structure anchor
  - `baseline_fixed_input` full `best_by_structured` = fit / tail keeper

## Next step boundary

- Do not spend more budget on optimizer sweeps.
- Do not spend more budget on width sweeps.
- Do not keep iterating generic loss / gate / bridge micro-variants on this branch.
- If another run is approved, escalate directly to:
  - `H15_LATE_RESIDUAL_HEAD_v1`
