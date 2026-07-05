# Current State

Updated: 2026-06-23
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

- 2026-06-23 v230 failure-case manual review / paper-case evidence pack is complete. GPTPro accepted v229 and explicitly kept model work stopped; the only allowed local step was audit-only casebook packaging. Codex implemented `stage03_v230_failure_case_manual_review_casebook_20260623.py` using only v225/v226/v228/v229 outputs. Output: `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/`; ZIP `v230_failure_case_manual_review_casebook_pack.zip`; validation passed with `py_compile`, full script run, ZIP `testzip=None`, selected case count `46` (`23` per pool), required files `[]`, guardrail `pass=True`, consistency `pass=True`, forbidden hits `[]`, copied figures `85`, logged missing case figures `13`, and all manual review fields left blank.
- v230 current boundary: this is not a new experiment and not model progress. Formal headline remains `loose_main_pool=avg_joint_focus` and `strict_main_pool=peak_floor_090`. Do not train models, generate predictions, tune tau/threshold, create gate/router/selector, run v222b/v223, or change formal headline. The only reasonable next step after v230 is manual reading of the casebook, filling `v230_manual_review_template.csv`, and drafting/refining the paper failure-case section.
- 2026-06-23 v228 final paper artifact freeze is complete. The user identified the earlier GPTPro handoff as mojibake; Codex switched to the local ChatGPT Desktop app, used a clean ASCII retry, obtained a valid GPTPro reply, and executed `stage03_v228_final_paper_artifact_freeze_20260623.py`. Output: `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/`; ZIP `v228_final_paper_artifact_freeze_pack.zip`; validation passed with ZIP `testzip=None`, required files `[]`, exact formal lock, zero main-metric diffs, v226 CI row count match, forbidden hits `0`, and guardrail/consistency pass.
- The previous 2026-06-23 goal-level blocked diagnosis is superseded for current-state purposes: it was incomplete because the prompt itself was unreadable in GPTPro. Keep the blocked archive as process history, but treat v228 as the current latest completed step.
- 2026-06-23 prompt encoding correction: the earlier Desktop GPTPro handoff was observed by the user as mojibake / garbled Chinese, so the prior "no valid GPTPro reply" diagnosis is incomplete. Use the self-contained ASCII-only prompt `gptpro_reviews/20260623_v227_clean_ascii_handoff_prompt.md` for the next GPTPro handoff.
- 2026-06-23 goal-level blocked audit: the Codex-GPTPro loop cannot continue automatically because the same GPTPro channel blocker repeated across consecutive goal turns. Desktop still lacks a valid bounded GPTPro reply, and Chrome bridge still cannot verify Pro/进阶 mode before sending. Final blocked archive: `gptpro_reviews/20260623_goal_blocked_gptpro_channel_*`.
- 2026-06-22 Gold-V2/v226 formal robustness / confidence-interval audit is complete. The authoritative live note layer is under `05_rebuild_from_raw_20260511/00_project_notes/`.
- GPTPro's latest instruction was accepted locally: build an audit-only robustness/CI pack from the v225 locked formal outputs, with no training, no new tau/threshold, no router/gate, and no v222b/v223.
- Formal headline remains locked as `loose_main_pool=avg_joint_focus` and `strict_main_pool=peak_floor_090`; `v222a_bounded_residual`, `v222a_noharm_gate`, `oracle_safe_gate`, and other diagnostic variants are excluded from formal tables.
- v225 reproduced the locked test metrics within `1e-5`: loose `avg_joint_focus` RMSE/tail `0.544884/0.629752`; strict `peak_floor_090` RMSE/tail `0.571770/0.658306`.
- v226 reproduced the same locked test metrics within `1e-5` and produced robustness CIs:
  - sample bootstrap test CI: loose RMSE `0.496066-0.593811`, loose tail `0.564811-0.693788`; strict RMSE `0.511036-0.635521`, strict tail `0.581652-0.736696`.
  - subject-block test CI: loose RMSE `0.428783-0.599684`, loose tail `0.515881-0.687686`; strict RMSE `0.473689-0.615000`, strict tail `0.539479-0.706505`.
  - tail error concentration remains non-uniform: test top-20% tail-SSE share is `0.659320` for loose and `0.672493` for strict.
- v226 verification passed: `py_compile`, full script run, ZIP `bad_file=None`, required files `[]`, formal lock, metric reproduction, leakage guard, forbidden scan, table alignment, and required figure counts.
- 2026-06-22 Gold-V2/v227 paper / claim readiness pack is complete as a reporting-only fallback. It was created only because the v226-to-GPTPro handoff channel was blocked: Desktop returned empty stopped-thinking outputs and Chrome required login. It does not unlock any new model work.
- v227 outputs live under `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/`; the next GPTPro prompt is `reports/v227_next_gptpro_prompt_ascii.md`.
- v227 verification passed: `py_compile`, full script run, ZIP `bad_file=None`, required files `[]`, `no_model_change_guard.pass=True`, `source_artifact_checks.pass=True`, and formal lock unchanged.
- A second GPTPro handoff attempt with the v227 prompt was blocked by Chrome bridge before sending: `Could not verify Pro/进阶 mode. Refusing to send.` The blocked v227 archive is under `gptpro_reviews/20260622_v227_result_gptpro_*_blocked.md`.
- 2026-06-23 heartbeat retry remains blocked. Desktop ChatGPT still shows the v226/v227 handoff prompt plus `已停止思考` and no valid six-item bounded GPTPro reply. Chrome bridge again failed before sending because it could not verify Pro/进阶 mode. The heartbeat blocked archive is under `gptpro_reviews/20260623_v227_heartbeat_gptpro_*_blocked.md`.
- Current next step when GPTPro is reachable: report v226+v227 results and ask only for a bounded writing/claim/reporting instruction. Do not start v222b/v223, a new tau, gate/router, selector, or test-based retuning.
- Closeout diagnosis still stands: locked test `selector_failed_rate` is about 0.41 combined, `candidate_missing_rate` is about 0.028 combined, and high-tail `candidate_missing_rate` is about 0.127 combined. This does not unlock v222b/v223.
- Do not enter v222b/v223, a new tau, a gate_v2, or a multi-candidate router from this state. The active next step is to report the v226 robustness/CI pack to GPTPro for the next bounded instruction.
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

- Current active 2026-06-23 task state: v230 failure-case manual review casebook has completed and passed validation. If work continues, do not ask for or run another model. The next useful action is human review: open the casebook figures and fill `tables/v230_manual_review_template.csv`, then use `reports/v230_paper_failure_case_section_draft_cn.md` as the starting point for the paper failure-case section.
- Current active v230 outputs:
  - `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v230_failure_case_manual_review_casebook_20260623.py`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/reports/v230_failure_case_manual_review_casebook_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/reports/v230_advisor_discussion_notes_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/reports/v230_paper_failure_case_section_draft_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/tables/v230_case_selection_index.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/tables/v230_manual_review_template.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/tables/v230_failure_casebook_table.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/tables/v230_bucket_to_claim_mapping.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/logs/guardrail_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/logs/consistency_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/logs/figure_copy_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v230_failure_case_manual_review_casebook_20260623/v230_failure_case_manual_review_casebook_pack.zip`
- Current active 2026-06-23 task state: v228 final paper artifact freeze has completed and passed validation. If the loop continues, report v228 results back to GPTPro and request exactly one bounded next step; do not start new model, threshold, gate/router/selector, or test-retuning work locally.
- Current active v228 outputs:
  - `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v228_final_paper_artifact_freeze_20260623.py`
  - `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/reports/v228_final_paper_artifact_freeze_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/reports/manuscript_results_section_draft_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/reports/manuscript_claim_boundary_notes_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/tables/final_main_result_table.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/logs/guardrail_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/logs/consistency_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/v228_final_paper_artifact_freeze_pack.zip`
- Current active 2026-06-22 task: GPTPro handoff is temporarily blocked after reporting the completed v226 robustness / confidence-interval audit. v227 has therefore been completed only as a reporting-only paper/claim readiness fallback from v225+v226 evidence. The next active action is to report v226+v227 to GPTPro once the GPTPro channel is reachable and wait for the next bounded instruction.
- Current active outputs:
  - `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v227_paper_claim_readiness_pack_20260622.py`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/reports/v227_paper_claim_readiness_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/reports/v227_next_gptpro_prompt_ascii.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/tables/paper_main_result_table.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/tables/paper_claim_support_matrix.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/tables/paper_limitation_table.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/logs/no_model_change_guard.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/logs/source_artifact_checks.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/logs/file_inventory.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/v227_paper_claim_readiness_pack.zip`
  - `05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v226_formal_robustness_ci_audit_20260622.py`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/reports/v226_formal_robustness_ci_audit_cn.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/tables/formal_model_lock_recheck.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/tables/formal_metric_ci_sample_bootstrap.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/tables/formal_metric_ci_subject_block_bootstrap.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/tables/formal_subject_level_metrics.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/tables/formal_tail_error_concentration.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/tables/formal_readiness_decision.csv`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/logs/run_manifest.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/logs/metric_reproduction_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/logs/leakage_guard_report.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/logs/forbidden_scan_report.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/logs/table_alignment_check.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/logs/file_inventory.json`
  - `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/v226_formal_robustness_ci_audit_pack.zip`
  - Prior v225 input remains at `05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/`.
  - `05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/v225_formal_route_reconstruction_evidence_pack.zip`
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
