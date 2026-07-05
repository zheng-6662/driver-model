# GPTPro v226 action items

## Implement

- Script: `stage03_v226_formal_robustness_ci_audit_20260622.py`
- Output directory: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622`

## Scope

- Audit-only + reporting-only.
- Use only locked formal models:
  - `loose_main_pool: avg_joint_focus`
  - `strict_main_pool: peak_floor_090`
- Read v225 formal outputs and, only if needed, locked formal prediction/target arrays.
- Do not train models, tune thresholds, create gate/router, run v222b/v223, or use diagnostic-only rows as formal inputs.

## Required outputs

- Tables:
  - `formal_model_lock_recheck.csv`
  - `formal_metric_ci_sample_bootstrap.csv`
  - `formal_metric_ci_subject_block_bootstrap.csv`
  - `formal_subject_level_metrics.csv`
  - `formal_route_event_level_metrics.csv`
  - `formal_bucket_ci_metrics.csv`
  - `formal_tail_error_concentration.csv`
  - `formal_underestimation_profile.csv`
  - `formal_extreme_peak_profile.csv`
  - `formal_sample_influence_audit.csv`
  - `formal_readiness_decision.csv`
- Figures:
  - `ci_forest_by_pool/`
  - `subject_level_metric_distribution/`
  - `tail_error_concentration/`
  - `underestimation_profile/`
  - `extreme_peak_cases_summary/`
- Report:
  - `reports/v226_formal_robustness_ci_audit_cn.md`
- Logs:
  - `run_manifest.json`
  - `input_file_hashes.json`
  - `bootstrap_config.json`
  - `metric_reproduction_check.json`
  - `leakage_guard_report.json`
  - `forbidden_scan_report.json`
  - `table_alignment_check.json`
  - `file_inventory.json`
- ZIP:
  - `v226_formal_robustness_ci_audit_pack.zip`

## Required checks

- `python -m py_compile` pass.
- Full script run pass.
- ZIP `testzip() == None`.
- Required files missing `[]`.
- Reproduce v225 locked test metrics within `1e-5`:
  - loose `avg_joint_focus`: RMSE `0.544884`, tail `0.629752`
  - strict `peak_floor_090`: RMSE `0.571770`, tail `0.658306`
- Leakage guard pass.
- Forbidden scan pass.
- Table alignment pass:
  - no duplicate sample_id within pool/split
  - no missing formal prediction
  - horizon length `21`
  - test n loose `184`
  - test n strict `174`
- Figure minimums:
  - `ci_forest_by_pool >= 2`
  - `subject_level_metric_distribution >= 4`
  - `tail_error_concentration >= 2`
  - `underestimation_profile >= 2`
  - `extreme_peak_cases_summary >= 2`

## Stop

- Stop after v226 pack is complete and checks pass.
- If formal metric reproduction, formal lock, sample alignment, v225 tail mask inheritance, forbidden scan, or ZIP completeness fails, stop and report. Do not start a repair model.
