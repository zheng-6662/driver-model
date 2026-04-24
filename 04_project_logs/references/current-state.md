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
- The 2026-04-23 late-residual follow-up has now also completed: `H15_LATE_RESIDUAL_HEAD_v1` is not a new keeper, but it gives the clearest late-capacity signal so far; if this direction is continued later, only its `best_by_structured` checkpoint is worth treating as the control.
- The 2026-04-23 selective late-residual follow-up has now also completed: `H15_LATE_RESIDUAL_SELECTIVE_v1` is not a new keeper, but it is the strongest selectivity probe so far; the remaining bottleneck is no longer missing late capacity, but aligning selective correction to true under-amplitude cases without giving back fit / tail / prefix / onset.
- The 2026-04-24 overnight chain is now closed:
  - `H15_LATE_RESIDUAL_SELECTIVE_v2_UNDERAMP_ALIGN` improved detector alignment but failed continuation, so the selective late-residual route is closed for this chain.
  - `H15_MAINLINE_TAIL_CALIB_v1` produced the closest mainline near-pass of the chain, but still missed the reversal pair.
  - the one allowed conservative follow-up `H15_MAINLINE_TAIL_CALIB_v2` did not improve on `v1` and re-opened a strong-pos target-bucket collapse.
  - next approved work should be a higher-level re-audit, not more local late-residual / gate / tail-calib patching on this chain.

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
  - `H15_MAINLINE_TAIL_CALIB_v1` as the closest non-promoted mainline checkpoint from the overnight chain

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
- `H15_LATE_RESIDUAL_HEAD_v1`
- `H15_LATE_RESIDUAL_SELECTIVE_v1`
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

## 2026-04-23 Late Residual Follow-up Completed

- Executed run:
  - `H15_LATE_RESIDUAL_HEAD_v1`
- Final run directory:
  - `03_results/tmp/effectiveness_followup_20260423/h15_late_residual_head_v1/TRAIN_V5_4_STATECOND_REV_20260423_163844`
- Outcome:
  - not promotable as a new keeper
  - but still informative as a mechanism probe
- Why it still fails promotion:
  - `best_by_loss` keeps the attractive average metrics:
    - `rmse_steer=0.4954`
    - `abs_tail_last_0p5s.rmse_steer=0.6284`
    - `late_peak_recall=0.6522`
  - but `best_by_loss` still hard-collapses on `strong_pos`:
    - `strong_pos.tail_amp_ratio_pred_over_gt=0.3163`
    - `strong_pos.tail_flatness_rate=1.0000`
  - `best_by_structured` repairs the old `H15` collapse materially:
    - `strong_pos.tail_amp_ratio_pred_over_gt: 0.2687 -> 0.4904`
    - `strong_pos.tail_flatness_rate: 1.0000 -> 0.5000`
    - `late_peak_recall: 0.6355 -> 0.6656`
  - but `best_by_structured` still misses the explicit amplitude floor:
    - `strong_pos.tail_amp_ratio_pred_over_gt = 0.4904 < 0.60`
  - and gives back too much fit / tail:
    - `rmse_steer=0.5474`
    - `abs_tail_last_0p5s.rmse_steer=0.6868`
- Late residual path diagnostic:
  - the new head is active and non-trivial on the late slice
  - on the built-in evaluated checkpoint it is only mildly more active on `strong_pos` than on non-strong cases, so selectivity is still weak
- Decision boundary after the run:
  - keep Run A as the response-structure anchor
  - keep `baseline_fixed_input` full `best_by_structured` as the fit / tail keeper
  - do not promote `H15_LATE_RESIDUAL_HEAD_v1` as-is
  - if another follow-up is ever approved on this line, only `best_by_structured` from this run is worth treating as the control checkpoint
  - keep optimizer / width / broad bridge / gate sweeps closed
- Stored follow-up artifact:
  - `04_project_logs/reports/effectiveness_followup_20260423/h15_late_residual_head_v1_summary.md`

## 2026-04-23 Selective Late Residual Follow-up Completed

- Executed run:
  - `H15_LATE_RESIDUAL_SELECTIVE_v1`
- Final run directory:
  - `03_results/tmp/effectiveness_followup_20260423/h15_late_residual_selective_v1/TRAIN_V5_4_STATECOND_REV_20260423_214210`
- Outcome:
  - not promotable as a new keeper
  - but the strongest selectivity probe so far on the late-residual line
- Why it still fails promotion:
  - `best_by_loss` is no longer the old hard-collapse shape:
    - `rmse_steer=0.5356`
    - `abs_tail_last_0p5s.rmse_steer=0.6745`
    - `late_peak_recall=0.6756`
    - `strong_pos.tail_amp_ratio_pred_over_gt=0.4947`
    - `strong_pos.tail_flatness_rate=0.7500`
  - but `best_by_loss` still misses the explicit strong-pos guardrail:
    - `strong_pos.tail_amp_ratio_pred_over_gt = 0.4947 < 0.60`
    - `strong_pos.tail_flatness_rate = 0.7500 > 0.60`
  - `best_by_structured` proves the selective path can push strong-pos repair much harder:
    - `strong_pos.tail_amp_ratio_pred_over_gt=1.4833`
    - `strong_pos.tail_flatness_rate=0.3750`
  - but `best_by_structured` does so in a globally damaged regime:
    - `rmse_steer=0.6379`
    - `abs_tail_last_0p5s.rmse_steer=0.7319`
    - `prefix_1p0s.rmse_steer=0.5853`
    - `response_onset_delay_mae_sec=0.6270`
- Late residual path diagnostic:
  - the selective gate is now clearly active and bucket-selective:
    - `strong_pos_vs_non_strong_ratio.gate_prob=4.6584`
    - `strong_pos_vs_non_strong_ratio.gate_mean=3.3443`
  - but correlation with actual tail under-amplitude remains weak / slightly negative, so failure-mechanism alignment is still insufficient
- Decision boundary after the run:
  - keep Run A as the response-structure anchor
  - keep `baseline_fixed_input` full `best_by_structured` as the fit / tail keeper
  - do not promote `H15_LATE_RESIDUAL_SELECTIVE_v1` as-is
  - do not reopen optimizer / width / broad bridge sweeps for this line
  - if another follow-up is ever approved on this line, stay on the selective late-residual path and treat the current run as the bracketed failure boundary:
    - `best_by_loss` = fit-preserving but under-repaired
    - `best_by_structured` = strong-pos-repaired but over-regressed
- Stored follow-up artifact:
  - `04_project_logs/reports/effectiveness_followup_20260423/h15_late_residual_selective_v1_summary.md`

## Next Execution Order

1. Use Run A and `baseline_fixed_input` as the live keeper split for any future comparison.
2. Do not run `OPT_A_H15` or `OPT_B_H15`; `H15` failed the hard guardrail.
3. Do not promote `H10` into the default mainline; keep it as ceiling evidence only.
4. Do not reopen bridge / gate / loss as the forward path.
5. Do not rerun `H15_AC_CF_HLF_v1`; the anti-collapse attempt is already closed as a no-go.
6. Do not spend more budget on optimizer or width sweeps for the `H15` branch.
7. Do not rerun raw `H15_LATE_RESIDUAL_HEAD_v1` as-is; it is not promotable in its current form.
8. Do not rerun `H15_LATE_RESIDUAL_SELECTIVE_v1` as-is; it is not promotable in its current form.
9. If a later follow-up is approved on this line, stay on the selective late-residual path and focus only on stronger under-amplitude-aligned selectivity with fit / tail / prefix protection.
10. Treat the current selective run as the bracketed failure boundary:
  - `best_by_loss` = fit-preserving but under-repaired
  - `best_by_structured` = strong-pos-repaired but over-regressed
11. Do not continue the 2026-04-24 overnight `H15` chain with more local `tail_calib` or selective late-residual variants.
12. If work resumes after this closure, start from the closure report:
  - `04_project_logs/reports/effectiveness_followup_20260423/h15_overnight_chain_closure_20260424.md`

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
