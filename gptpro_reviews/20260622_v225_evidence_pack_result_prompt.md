# v225 execution result for GPTPro review

You gave the bounded instruction to build `stage03_v225_formal_route_reconstruction_evidence_pack_20260622.py` as a one-shot formal route reconstruction evidence pack. I executed it locally and need your next bounded instruction.

## Executed boundary

- No training.
- No new threshold/tau search.
- No router/gate creation.
- No v222b/v223 run.
- No formal headline change.
- Locked formal models only:
  - `loose_main_pool = avg_joint_focus`
  - `strict_main_pool = peak_floor_090`
- Diagnostic-only models/rows stayed excluded from formal tables:
  - `v222a_bounded_residual`
  - `v222a_noharm_gate`
  - `oracle_safe_gate`
  - `ridge_residual_peakfloor`
  - `W3_B4_original_soft`
  - oracle/fallback/true-label rows

## Local outputs

- Script:
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v225_formal_route_reconstruction_evidence_pack_20260622.py`
- Output directory:
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622`
- Zip:
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v225_formal_route_reconstruction_evidence_pack_20260622\v225_formal_route_reconstruction_evidence_pack.zip`
- Report:
  - `reports/v225_formal_route_reconstruction_evidence_cn.md`
- Key tables:
  - `tables/formal_model_lock.csv`
  - `tables/formal_reconstruction_metrics_overall.csv`
  - `tables/formal_reconstruction_metrics_by_pool.csv`
  - `tables/formal_reconstruction_metrics_by_bucket.csv`
  - `tables/formal_reconstruction_metrics_by_route_event.csv`
  - `tables/per_sample_formal_reconstruction_eval.csv`
  - `tables/formal_failure_case_index.csv`
  - `tables/diagnostic_only_v222a_closeout_summary.csv`
  - `tables/excluded_diagnostic_models_audit.csv`
- Logs:
  - `logs/run_manifest.json`
  - `logs/leakage_guard_report.json`
  - `logs/forbidden_scan_report.json`
  - `logs/metric_reproduction_check.json`
  - `logs/table_alignment_check.json`
  - `logs/file_inventory.json`

## Required checks and results

- `python -m py_compile` passed.
- Full script run passed.
- ZIP `testzip()` returned `None`.
- Required files missing: `[]`.
- Figure counts:
  - `formal_examples = 12`
  - `worst_tail_cases = 12`
  - `strong_under_cases = 8`
  - `baseline_sufficient_cases = 8`
- Sample figure visual spot-check passed: nonblank plot with title containing pool, sample_id, formal_model, RMSE, tail RMSE, under flag.
- Metric reproduction passed within abs tolerance `1e-5`:
  - `loose_main_pool / avg_joint_focus / test / rmse`: actual `0.5448840970647589`, expected `0.544884`, diff `9.71e-08`
  - `loose_main_pool / avg_joint_focus / test / tail_rmse`: actual `0.6297521592665997`, expected `0.629752`, diff `1.59e-07`
  - `strict_main_pool / peak_floor_090 / test / rmse`: actual `0.571769914574812`, expected `0.571770`, diff `8.54e-08`
  - `strict_main_pool / peak_floor_090 / test / tail_rmse`: actual `0.6583063251135349`, expected `0.658306`, diff `3.25e-07`
- Leakage guard passed all checks:
  - `formal_model_lock_exact`
  - `no_training_executed`
  - `no_new_tau_created`
  - `no_test_retuning`
  - `no_router_created`
  - `no_v222b_or_v223`
  - `no_oracle_in_formal`
  - `no_true_label_in_formal`
  - `sample_id_alignment_pass`
  - `pool_filter_pass`
  - `split_filter_pass`
- Forbidden scan passed on formal files:
  - scanned formal tables only
  - no hits for `W3_B4_original_soft`, `oracle`, `oracle_model`, `true_label`, `fallback`, `v222a_noharm_gate`, `v222a_bounded_residual`, `oracle_safe_gate`
  - diagnostic mentions are only in `diagnostic_only_v222a_closeout_summary.csv`, `excluded_diagnostic_models_audit.csv`, and report appendix.
- Table alignment passed:
  - `per_sample_rows = 2130`
  - `route_event_rows = 2130`
  - `failure_case_rows = 2130`
  - key sets match
  - duplicate sample_id within pool/split: `0`
  - missing formal prediction rows: `0`
  - bad horizon rows: `0`
  - prediction shape: `N x 21`
  - horizon length: `21`

## Formal test metrics in `formal_reconstruction_metrics_by_pool.csv`

- `loose_main_pool / avg_joint_focus / test`
  - `n = 184`
  - `rmse = 0.544884`
  - `tail_rmse = 0.629752`
  - `mean_sample_rmse = 0.468061`
  - `median_sample_rmse = 0.394778`
  - `p90_sample_rmse = 0.856631`
  - `under_rate = 0.163043`
  - `direction_acc = 0.967391`
  - `strong_steer_rate = 0.434783`
  - `extreme_peak_rate = 0.032609`
- `strict_main_pool / peak_floor_090 / test`
  - `n = 174`
  - `rmse = 0.571770`
  - `tail_rmse = 0.658306`
  - `mean_sample_rmse = 0.485644`
  - `median_sample_rmse = 0.406667`
  - `p90_sample_rmse = 0.832101`
  - `under_rate = 0.137931`
  - `direction_acc = 0.948276`
  - `strong_steer_rate = 0.459770`
  - `extreme_peak_rate = 0.034483`

## Note-layer sync completed

I updated:

- `05_rebuild_from_raw_20260511\00_project_notes\PROJECT_STATUS_CN.md`
- `05_rebuild_from_raw_20260511\00_project_notes\TASK_QUEUE_CN.md`
- `05_rebuild_from_raw_20260511\00_project_notes\ARTIFACT_INDEX_CN.md`
- `05_rebuild_from_raw_20260511\00_project_notes\daily_logs\2026-06-22.md`
- `04_project_logs\references\current-state.md`
- `04_project_logs\reports\progress\decision_log.md`

## Request

Please analyze the v225 result and give the next bounded instruction.

Hard requirements for your next instruction:

1. State whether v225 is accepted as complete.
2. State the next exact script name and output directory.
3. State allowed inputs and forbidden inputs.
4. State whether the next step is audit-only, packaging/reporting, or any model/training step.
5. If you propose any model/training/router/gate/tau/threshold step, specify why it is unlocked despite the current closeout diagnosis and provide a strict stop condition.
6. Do not request test-based retuning.
7. Do not request v222b/v223 unless you explicitly override the current locked condition and provide formal guardrails.
8. Provide required output files and pass/fail checks.

The goal is to keep moving toward the project objective without local overfitting or getting stuck in repeated selector tuning.
