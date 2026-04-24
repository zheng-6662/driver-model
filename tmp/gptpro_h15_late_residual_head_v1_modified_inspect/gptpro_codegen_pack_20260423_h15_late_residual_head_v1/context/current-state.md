# Current State

Updated: 2026-04-23
Owner: Codex

## Read Order

1. `04_project_logs/reports/progress/daily/2026-04-23.md`
2. `04_project_logs/reports/progress/decision_log.md`
3. `04_project_logs/reports/progress/experiment_registry.md`
4. `04_project_logs/reports/gptpro_effectiveness_review_20260423.md`
5. `04_project_logs/reports/effectiveness_followup_20260422/effectiveness_summary.md`
6. `02_code/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
7. `02_code/final_code/model/training/v58_modular/README.md`
8. `02_code/final_code/model/training/v58_modular/train.py`
9. `02_code/tools/recalc_v58_checkpoint_with_current_metrics.py`
10. `02_code/tools/run_effectiveness_followup.py`
11. `02_code/tools/summarize_effectiveness_followup.py`

## Handoff Priority

- The 2026-04-22 effectiveness plan is now the active closure for this round.
- Bridge / gate / loss are closed as the main frontier.
- When older 2026-04-21 bridge guidance conflicts with this file, follow the 2026-04-22 effectiveness outcome.
- The active V5.8 implementation now lives under `02_code/final_code/model/training/v58_modular/`; the old script remains the stable entrypoint only.
- The 2026-04-23 anti-collapse follow-up is now closed: `H15_AC_CF_HLF_v1` partially repairs the old `H15` collapse but still fails the guardrail review, so the next approved direction is a minimal late-residual-head slice if another run is authorized.

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

## Active Code Organization

- Primary wrapper entrypoint:
  - `02_code/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
- Active implementation package:
  - `02_code/final_code/model/training/v58_modular/`
- Current package split:
  - `paths.py`
  - `utils.py`
  - `config.py`
  - `data.py`
  - `modeling.py`
  - `losses.py`
  - `metrics.py`
  - `evaluation.py`
  - `train.py`
- Compatibility layers kept intentionally:
  - `shared.py`
  - `losses_metrics.py`
- Editing rule for future work:
  - do not add substantial new logic back into the wrapper
  - place future changes in the matching module under `v58_modular/`
  - keep compatibility layers thin rather than letting them become a second monolith

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
- `H15_AC_CF_HLF_v1`
- `OPT_A_20`
- `OPT_C_BEST`
- `CAP_192_BEST`
- `WINNER_CONFIRM`
- `H10` as a mainline candidate

## 2026-04-23 GPT Pro Review Outcome

- GPT Pro agrees that the first bottleneck is not optimizer tuning and not mild width/capacity.
- The preferred next approved run is:
  - `H15_AC_CF_HLF_v1`
- Meaning:
  - keep `FUTURE_SEC=1.5`
  - enable coarse-fine steer decomposition
  - enable hard-late fine supervision
  - align the hard-late window to the actual failing absolute slice:
    - `HARD_LATE_START_SEC = 1.00`
    - `HARD_TAIL_START_SEC = 1.00`
  - keep these disabled for the first anti-collapse attempt:
    - `PHASE_ADAPTIVE_TREND`
    - `STRONG_POS_GATE`
    - `W_FIRSTREV_LOCAL`
- Rationale locked by the review:
  - `H15` gains are real under D0 absolute windows.
  - `H15` fails because those gains are bought by flattening the `strong_pos` late tail.
  - `OPT_A_20` and `CAP_192_BEST` lower the EV of optimizer / width sweeps as the next move.
- Fallback if the anti-collapse run fails:
  - stop optimizer / width / loss micro-sweeps
  - escalate to a minimal late residual head on the decoder output for `t >= 1.0 s`
- Stored review artifact:
  - `04_project_logs/reports/gptpro_effectiveness_review_20260423.md`

## 2026-04-23 Anti-Collapse Follow-up Completed

- Executed run:
  - `H15_AC_CF_HLF_v1`
- Final run directory:
  - `03_results/tmp/effectiveness_followup_20260423/h15_ac_cf_hlf_v1/TRAIN_V5_4_STATECOND_REV_20260423_131956`
- Outcome:
  - failed / no-go after manual strong-pos review
- Why it still fails:
  - `best_by_loss` does improve old `H15` strong-pos collapse:
    - `strong_pos.tail_amp_ratio_pred_over_gt: 0.2687 -> 0.5141`
    - `strong_pos.tail_flatness_rate: 1.0000 -> 0.3750`
  - but it still misses the required amplitude floor and late-peak recall:
    - `late_peak_recall = 0.5786`
    - target was `>= 0.62`
  - `best_by_structured` explicitly fails:
    - `strong_pos.tail_amp_ratio_pred_over_gt = 0.3304 < 0.50`
  - manual review of eight representative `strong_pos` plots still finds three severe final-tail under-amplitude cases
- Decision boundary after the run:
  - keep Run A as the response-structure anchor
  - keep `baseline_fixed_input` full `best_by_structured` as the fit / tail keeper
  - do not reopen optimizer / width / bridge / gate / generic loss sweeps on this line
  - if another run is approved, escalate directly to:
    - `H15_LATE_RESIDUAL_HEAD_v1`
- Stored follow-up artifact:
  - `04_project_logs/reports/effectiveness_followup_20260423/h15_ac_cf_hlf_v1_summary.md`

## Next Execution Order

1. Use Run A and `baseline_fixed_input` as the live keeper split for any future comparison.
2. Do not run `OPT_A_H15` or `OPT_B_H15`; `H15` failed the hard guardrail.
3. Do not promote `H10` into the default mainline; keep it as ceiling evidence only.
4. Do not reopen bridge / gate / loss as the forward path.
5. Do not rerun `H15_AC_CF_HLF_v1`; the anti-collapse attempt is already closed as a no-go.
6. Do not spend more budget on optimizer or width sweeps for the `H15` branch.
7. If a new follow-up round is approved, go directly to `H15_LATE_RESIDUAL_HEAD_v1`.

## Logging Rules For This State

- Every meaningful result in this round is logged in:
  - `04_project_logs/reports/progress/daily/2026-04-23.md`
- Every named run or analysis has a registry row in:
  - `04_project_logs/reports/progress/experiment_registry.md`
- The consolidated effectiveness outputs live in:
  - `04_project_logs/reports/effectiveness_followup_20260422/`
- The anti-collapse follow-up closure lives in:
  - `04_project_logs/reports/effectiveness_followup_20260423/`
- The normalized external review for the next-step recommendation lives in:
  - `04_project_logs/reports/gptpro_effectiveness_review_20260423.md`

## Do Not Reopen

- Do not split into scene-specific models.
- Do not reopen bridge / gate / loss as the main line.
- Do not treat `H10` as the default next training route.
- Do not spend budget on `CAP_256` or deeper stacks from the current evidence.
- Do not change split, input pipeline, anchor policy, or the stable GPU path in the name of this round.
- Do not pile new substantial logic back into the monolithic wrapper script; continue from the `v58_modular/` package structure.
