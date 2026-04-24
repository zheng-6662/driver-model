# Source Manifest

## Proposal

- `proposal/opening_report_zhengxin_final.docx`
  - original opening report provided by the user
- `proposal/opening_report_zhengxin_final.pdf`
  - PDF export when available
- `proposal/OPENING_REPORT_PAPER_RELEVANT_SUMMARY_CN.md`
  - manual summary of the opening report for paper positioning
- `proposal/OPENING_REPORT_KEY_SECTIONS_EXTRACT_CN.md`
  - direct section extract for paper-relevant chapters

## Project context

- `context/current-state.md`
  - current official state and read order
- `context/decision_log.md`
  - fixed decisions and closed directions
- `context/experiment_registry.md`
  - run-by-run project history
- `context/daily_2026-04-20.md`
- `context/daily_2026-04-21.md`
- `context/daily_2026-04-22.md`
- `context/daily_2026-04-23.md`
  - active rounds most relevant to the current paper story
- `context/INPUT_ROLE_AUDIT.md`
  - claim boundary for style, road preview, and teacher-side signals
- `context/THESIS_DEFENSE_TABLES.md`
  - already contains paper-like tables and claim-limitation pairing
- `context/NEXT_DECISION_RULES.md`
  - transition logic from project evidence to narrative choices

## Active code

- `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
  - wrapper entrypoint
- `code/v58_modular/*.py`
  - active modular implementation package
- `tools/recalc_v58_checkpoint_with_current_metrics.py`
  - common recalc and metric export tool
- `tools/run_input_group_ablation.py`
- `tools/summarize_input_group_ablation.py`
- `tools/run_effectiveness_followup.py`
- `tools/summarize_effectiveness_followup.py`
  - active evidence-generation tools
- `tools/audit_vehicle_feature_columns.py`
- `tools/analyze_trigger_response_lag.py`
- `tools/select_checkpoint_pareto.py`
- `tools/analyze_spike_positions.py`
  - boundary and diagnostic tools directly relevant to paper claims

## Evidence

- `evidence/feature_presence_report.md`
- `evidence/TASK_DEFINITION_AND_EVENT_LOGIC.md`
- `evidence/pareto_summary.json`
- `evidence/spike_position_summary.md`
  - boundary and interpretation evidence
- `evidence/input_ablation_summary.md`
- `evidence/bridge_summary.md`
- `evidence/effectiveness_summary.md`
  - major result summaries
- `evidence/*comparison_table.csv`
  - consolidated comparison tables
- `evidence/run_summaries/*.json`
  - representative per-run structured summaries

## Configs

- `configs/runA_run_config.json`
- `configs/baseline_fixed_input_run_config.json`
- `configs/plus_pedals_run_config.json`
- `configs/bridge_50_50_run_config.json`
- `configs/bridge_schedule_B_to_A_run_config.json`
- `configs/h15_run_config.json`
- `configs/h10_run_config.json`
- `configs/opt_a_20_run_config.json`
- `configs/opt_c_best_run_config.json`
- `configs/winner_confirm_run_config.json`
  - representative settings for the key keepers and branches

## Assets

- `assets/driver_style_cluster_result.xlsx`
- `assets/driver_style_per_subject.csv`
  - style-prior source artifacts used by the active mainline
