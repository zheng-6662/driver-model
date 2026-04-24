# GPT Pro Codegen Pack

Date: `2026-04-23`

This pack is for the next step after `H15_AC_CF_HLF_v1` closed as a no-go.
Unlike the earlier effectiveness handoff, this pack is meant for direct code generation against the current modular source tree.

## One-line status

The live keepers are still:

- fit / tail keeper:
  - `baseline_fixed_input` full `best_by_structured`
- response-structure anchor:
  - Run A full `best_by_structured`

`H15` had real upside but collapsed on `strong_pos`.
`H15_AC_CF_HLF_v1` partially repaired that collapse but still failed after manual review.
The next approved direction, if code is to be generated, is a minimal late residual head:

- `H15_LATE_RESIDUAL_HEAD_v1`

## Recommended read order

1. `PROJECT_STATUS_AND_CODEGEN_BRIEF_CN.md`
2. `GPTPRO_PROMPT_CN.md`
3. `context/current-state.md`
4. `context/daily_2026-04-23.md`
5. `context/decision_log.md`
6. `context/experiment_registry.md`
7. `evidence/gptpro_effectiveness_review_20260423.md`
8. `evidence/h15_ac_cf_hlf_v1_summary.md`
9. `evidence/baseline_fixed_input_recalc_best_by_structured_summary.json`
10. `evidence/h15_recalc_best_by_structured_summary.json`
11. `evidence/h15_ac_cf_hlf_v1_recalc_best_by_loss_summary.json`
12. `evidence/h15_ac_cf_hlf_v1_recalc_best_by_structured_summary.json`
13. `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
14. `code/v58_modular/README.md`
15. `code/v58_modular/modeling.py`
16. `code/v58_modular/losses.py`
17. `code/v58_modular/train.py`
18. `code/v58_modular/evaluation.py`
19. `code/recalc_v58_checkpoint_with_current_metrics.py`

## Package contents

- `PROJECT_STATUS_AND_CODEGEN_BRIEF_CN.md`
  - concise Chinese briefing of the current state, the failed anti-collapse follow-up, and the exact code-generation target
- `GPTPRO_PROMPT_CN.md`
  - direct prompt for GPT Pro to generate the next minimal code slice
- `context/`
  - current-state, latest daily log, decision log, experiment registry
- `evidence/`
  - latest review, closed-run summary, comparison summaries, and the manual `strong_pos` review artifacts
- `configs/`
  - exact `run_config.json` snapshots for:
    - `baseline_fixed_input`
    - old `H15`
    - new `H15_AC_CF_HLF_v1`
- `protocol/`
  - protocol config and frozen subject split
- `code/`
  - current wrapper script
  - full `v58_modular/` source tree
  - current recalc and follow-up tools

## What GPT Pro should do in this round

- Do not reopen optimizer / width / bridge / gate / generic loss sweeps.
- Use the current modular source tree as the implementation base.
- Generate the smallest code change that makes `H15_LATE_RESIDUAL_HEAD_v1` real and runnable.
- Keep the wrapper thin and put new logic in `v58_modular/`.
