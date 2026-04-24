# GPT Pro Effectiveness Handoff Pack

Date: `2026-04-23`

This pack is meant to let GPT Pro judge the next step quickly without re-reading the whole repository history.

## One-line status

The current official keeper is still `baseline_fixed_input` full `best_by_structured` on the `2.0s` line.
`H15` showed real upside, but it hard-collapsed on `strong_pos`.
`OPT_A_20`, `OPT_C_BEST`, `CAP_192_BEST`, and `WINNER_CONFIRM` did not replace the current keeper.
`H10` only serves as diagnostic ceiling evidence and is not promoted.

## Recommended read order

1. `PROGRESS_AND_RESULTS_SUMMARY_CN.md`
2. `GPTPRO_PROMPT_CN.md`
3. `context/current-state.md`
4. `context/daily_2026-04-22.md`
5. `context/decision_log.md`
6. `context/experiment_registry.md`
7. `evidence/effectiveness_summary.md`
8. `evidence/effectiveness_comparison_table.csv`
9. `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
10. `code/recalc_v58_checkpoint_with_current_metrics.py`
11. `code/run_effectiveness_followup.py`
12. `code/summarize_effectiveness_followup.py`

## Package contents

- `PROGRESS_AND_RESULTS_SUMMARY_CN.md`
  - Chinese briefing of the current project state, latest round results, and what is still open.
- `GPTPRO_PROMPT_CN.md`
  - A direct prompt that asks GPT Pro to judge the next move with concrete output requirements.
- `context/`
  - `current-state`, daily log, decision log, and experiment registry.
- `code/`
  - Active training script, recalc tool, runner, and summarizer used in the closed effectiveness round.
- `configs/`
  - Exact `run_config.json` snapshots for the keeper and the main effectiveness runs.
- `evidence/`
  - The consolidated effectiveness summary, tables, manifest, and per-run summary JSON evidence.

## What GPT Pro should focus on

- Decide whether the next budget should go to:
  - explicit anti-collapse work for the `1.5s` branch, or
  - a new architecture direction justified by the `H10` ceiling gap.
- Avoid reopening already closed frontier directions:
  - bridge / gate / loss as the main line
  - plain optimizer sweeps on the current `2.0s` baseline
  - default promotion of `H10`
  - deeper / wider stacks without a sharper hypothesis
