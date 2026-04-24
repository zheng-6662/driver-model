# Current State

Updated: 2026-04-22
Owner: Codex

## Read Order

1. `04_project_logs/reports/progress/daily/2026-04-22.md`
2. `04_project_logs/reports/progress/decision_log.md`
3. `04_project_logs/reports/progress/experiment_registry.md`
4. `04_project_logs/reports/effectiveness_followup_20260422/effectiveness_summary.md`
5. `02_code/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
6. `02_code/tools/recalc_v58_checkpoint_with_current_metrics.py`
7. `02_code/tools/run_effectiveness_followup.py`
8. `02_code/tools/summarize_effectiveness_followup.py`

## Handoff Priority

- The 2026-04-22 effectiveness plan is now the active closure for this round.
- Bridge / gate / loss are closed as the main frontier.
- When older 2026-04-21 bridge guidance conflicts with this file, follow the 2026-04-22 effectiveness outcome.

## Fixed Historical Anchors Still Kept

- Historical strongest allowed mainline:
  - `03_results/tmp/protocol_safe_runs/TRAIN_V5_4_STATECOND_REV_20260416_220918`
- Stable reproducible control:
  - `03_results/tmp/v220918_strict_repro_manualup_full_d/TRAIN_V5_4_STATECOND_REV_20260420_121314`
- Response-structure anchor:
  - Run A full `best_by_structured`
- Fit / tail numeric anchor and current best non-collapse base:
  - `baseline_fixed_input` full `best_by_structured`
- Historical frontier evidence kept only as non-promoted references:
  - `bridge_50_50`
  - `bridge_schedule_B_to_A`
  - `plus_pedals`

## Current Task Definition

- The project remains pooled post-trigger steering response prediction.
- The four scenarios remain elicitation protocols, not separate scene-specific models.
- `fixed_v20260421` stays as the active input version.
- protocol-safe split stays locked.
- stable GPU path stays locked.

## 2026-04-22 Effectiveness Round Completed

### Tooling Completed

- Active script now supports the effectiveness env interface for:
  - horizon
  - optimizer bundle
  - width / dropout
- Recalc now exports absolute-time `window_metrics`.
- Runner and summarizer for the effectiveness line are in place:
  - `02_code/tools/run_effectiveness_followup.py`
  - `02_code/tools/summarize_effectiveness_followup.py`

### D0 Diagnostic Completed

- `baseline_fixed_input` absolute-window anchor is locked at:
  - `prefix_1p0s.rmse_steer=0.4291`
  - `prefix_1p5s.rmse_steer=0.4906`
  - `full_horizon.rmse_steer=0.5559`
  - `abs_tail_last_0p5s.rmse_steer=0.7171`
- Run A still recalcates in `deg`, so it remains structure-only reference.
- Fraction-based tail metrics are now explicitly treated as horizon-biased; use `abs_tail_last_0p5s.rmse_steer` for cross-horizon comparison.

### Training Questions Answered

1. Is `1.5 s` viable as a promotable base?
   - Not yet. `H15` improves overall RMSE and absolute tail, but hard-collapses on `strong_pos`.
2. Can the original `2.0 s` task be rescued by the `OPT_A` bundle?
   - No. `OPT_A_20` slightly helps the first second but regresses overall RMSE, absolute tail, and late-peak recall.
3. Is there a shorter-horizon ceiling worth knowing?
   - Yes. `H10` shows a much easier `1.0 s` ceiling, but it remains diagnostic only and is not promoted.
4. Does mild capacity help on the current best base?
   - No. `CAP_192_BEST` regresses the main fit / tail metrics.
5. Does conditional regularization beat the baseline winner?
   - No. `OPT_C_BEST` improves some guardrail stability but does not beat the original baseline anchor on the selection order.
6. Does winner confirmation overturn the winner?
   - No. `WINNER_CONFIRM` stays close to the baseline route but does not outperform the original `baseline_fixed_input` keeper.

## Live Keepers After Effectiveness

- Response-structure keeper:
  - Run A full `best_by_structured`
- Fit / tail keeper and best current non-collapse base:
  - `baseline_fixed_input` full `best_by_structured`
- Diagnostic horizon ceiling only:
  - `H10` full `best_by_structured`

## Closed Non-Promoted Effectiveness Branches

- `H15`
- `OPT_A_20`
- `OPT_C_BEST`
- `CAP_192_BEST`
- `WINNER_CONFIRM`
- `H10` as a mainline candidate

## Next Execution Order

1. Use Run A and `baseline_fixed_input` as the live keeper split for any future comparison.
2. Do not run `OPT_A_H15` or `OPT_B_H15`; `H15` failed the hard guardrail.
3. Do not promote `H10` into the default mainline; keep it as ceiling evidence only.
4. Do not reopen bridge / gate / loss as the forward path.
5. If a new follow-up round is approved, it should target:
   - explicit anti-collapse work for the `1.5 s` branch, or
   - a new architecture direction justified by the `H10` ceiling gap,
   rather than repeating the same optimization / width sweeps.

## Logging Rules For This State

- Every meaningful result in this round is logged in:
  - `04_project_logs/reports/progress/daily/2026-04-22.md`
- Every named run or analysis has a registry row in:
  - `04_project_logs/reports/progress/experiment_registry.md`
- The consolidated effectiveness outputs live in:
  - `04_project_logs/reports/effectiveness_followup_20260422/`

## Do Not Reopen

- Do not split into scene-specific models.
- Do not reopen bridge / gate / loss as the main line.
- Do not treat `H10` as the default next training route.
- Do not spend budget on `CAP_256` or deeper stacks from the current evidence.
- Do not change split, input pipeline, anchor policy, or the stable GPU path in the name of this round.
