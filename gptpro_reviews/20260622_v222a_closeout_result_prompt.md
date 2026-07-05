# GPTPro Review Request: v222a closeout candidate gap audit result

当前需要你给下一轮 bounded 指令。请不要给 broad brainstorming，也不要要求 test-set retuning。

## Local Decision So Far

- decision: `STOP v222a bounded residual / no-harm gate formal line`
- formal headline locked:
  - `loose_main_pool = avg_joint_focus`
  - `strict_main_pool = peak_floor_090`
- diagnostic only:
  - `v222a_bounded_residual`
  - `v222a_noharm_gate`
  - `oracle_safe_gate`
  - best allowed candidate oracle
- output dir: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622`
- pack: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\v222a_closeout_candidate_gap_audit_pack.zip`
- report: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_closeout_candidate_gap_audit_20260622\reports\v222a_closeout_candidate_gap_audit_cn.md`

## What Codex Implemented

Script:

`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_closeout_candidate_gap_audit_20260622.py`

It reads only existing artifacts:

- `v221_formal_model_leaderboard_20260622`
- `v222a_candidate_curve_cache_20260622`
- `v222a_light_fusion_residual_20260622`
- `v222a_noharm_gate_diagnostic_20260622`

It does not train a new model, does not tune a new tau, does not add a router, and does not change test-based config.

Generated required deliverables:

- `tables/formal_headline_decision.csv`
- `tables/v222a_stop_evidence.csv`
- `tables/oracle_vs_learned_gap.csv`
- `tables/candidate_gap_audit.csv`
- `tables/per_sample_failure_taxonomy.csv`
- `tables/bucket_failure_summary.csv`
- `tables/future_route_decision.csv`
- `figures/top_selector_failed_cases/`
- `figures/top_candidate_missing_cases/`
- `figures/top_safe_under_fix_cases/`
- `figures/top_baseline_sufficient_cases/`
- `reports/v222a_closeout_candidate_gap_audit_cn.md`
- `logs/closeout_manifest.json`
- `logs/sha256_manifest.csv`
- `logs/zip_verify.json`
- `v222a_closeout_candidate_gap_audit_pack.zip`

## Exact Evidence

v222a stop evidence:

| pool | validation pass | locked test pass | test RMSE delta | test tail delta | test under reduction | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `loose_main_pool` | True | False | `+0.010559` | `+0.027764` | `+0.043478` | under improves but RMSE/tail harmed |
| `strict_main_pool` | True | False | `-0.008975` | `-0.005264` | `-0.017241` | RMSE/tail safe but under worsens |

Oracle vs learned gate on locked test:

| pool | learned tail gain | oracle tail gain | oracle-minus-learned tail gap | selector_failed_rate | candidate_missing_rate |
|---|---:|---:|---:|---:|---:|
| `loose_main_pool` | `-0.027764` | `+0.105286` | `+0.133050` | `0.407609` | `0.027174` |
| `strict_main_pool` | `+0.005264` | `+0.106719` | `+0.101455` | `0.413793` | `0.028736` |
| `combined` | not directly pooled in table | not directly pooled in table | not directly pooled in table | `0.410615` | `0.027933` |

Future route decision from `future_route_decision.csv`:

| scope | high_tail_error_n | high_tail_candidate_missing_rate | high_tail_oracle_clear_gain_rate | v222b_allowed | v223_allowed |
|---|---:|---:|---:|---:|---:|
| `loose_main_pool` | `42` | `0.119048` | `0.928571` | False | False |
| `strict_main_pool` | `37` | `0.135135` | `0.891892` | False | False |
| `combined` | `79` | `0.126582` | `0.911392` | False | False |

Interpretation:

- The allowed candidate pool often still contains a better diagnostic candidate, especially in high-tail cases.
- The dominant actionable failure is learned selector/gate generalization, not candidate_missing.
- Since `candidate_missing_rate` in high-tail samples is only `0.126582` combined, the closeout audit does not unlock v223.
- Since learned gate passes validation but fails locked test, v222b/neural gate is also not unlocked.

## Verification

Commands / checks passed:

- `python -m py_compile F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_closeout_candidate_gap_audit_20260622.py`
- full script run completed
- ZIP `testzip()` result: `bad_file=None`
- ZIP file count: `74`
- required files missing: `[]`
- `leakage_guard_result.csv`: 6/6 pass
- formal headline forbidden identity check: pass
- forbidden string scan over closeout tables/reports/logs for `W3_B4_original_soft|oracle_model|true_label|fallback`: no matches
- case figure visual inspection: pass; 61 PNGs generated

## Guardrails For Your Next Instruction

Please keep the next step inside these constraints:

- No v222b training unless you give a new subject-group repeated validation protocol and a concrete reason why it is not just a larger overfit gate.
- No v223/new candidate generator unless you challenge the closeout conclusion with a specific alternative interpretation of the low high-tail `candidate_missing_rate`.
- No new tau, no `v222a_gate_v2`, no multi-candidate router, no test-based config.
- No use of `W3_B4_original_soft`, oracle rows, true-label fallback, or diagnostic-only rows in formal leaderboard/gate/usage/selected configs.
- Train-only fitting, validation-only selection, test reporting only.

## Questions For GPTPro

1. Do you accept the closeout diagnosis that the current failure is mainly selector/gate generalization rather than candidate pool missing?
2. Given `v222b_allowed=False` and `v223_allowed=False`, what is the next bounded local step that can still move the project closer to the final goal?
3. Should Codex next do:
   - A. stop and write a final project-facing v221/v222a conclusion report,
   - B. run a non-training subject-group repeated validation audit over the existing gate/candidate decisions,
   - C. build a compact failure atlas / paper-style analysis from the closeout taxonomy,
   - D. return to upstream data/anchor/sample audit,
   - E. something else strictly bounded?
4. For your chosen next step, specify exact required files, pass/fail criteria, and stop conditions.
5. What should Codex explicitly avoid doing next?
