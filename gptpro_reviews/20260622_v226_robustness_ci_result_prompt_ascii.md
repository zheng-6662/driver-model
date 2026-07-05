# GPTPro review request: v226 formal robustness / CI audit completed

Please ignore the immediately previous message if it displayed mojibake or
garbled Chinese. This is the same v226 result report, rewritten in ASCII-only
English so it can be parsed reliably.

I need your bounded next instruction for Codex. Please answer with:

1. Whether v226 is accepted as complete or rejected, and the exact reason.
2. The next local task name/version.
3. The allowed input files.
4. The required output directory and required files.
5. The exact stop condition.
6. The validation commands/checks required before reporting back to you.

Current local decision boundary:

- v225 formal closeout is complete.
- v226 formal robustness / CI audit is complete.
- v226 was audit-only and reporting-only.
- No model was trained.
- No threshold was tuned.
- No gate/router was created.
- v222b/v223 were not run.
- Diagnostic-only rows were not used as formal inputs.
- Formal locked models only:
  - loose_main_pool: avg_joint_focus
  - strict_main_pool: peak_floor_090

v226 implementation:

- Script:
  - 05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v226_formal_robustness_ci_audit_20260622.py
- Output directory:
  - 05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622
- Report:
  - reports/v226_formal_robustness_ci_audit_cn.md
- ZIP:
  - v226_formal_robustness_ci_audit_pack.zip

Required tables generated:

- formal_model_lock_recheck.csv
- formal_metric_ci_sample_bootstrap.csv
- formal_metric_ci_subject_block_bootstrap.csv
- formal_subject_level_metrics.csv
- formal_route_event_level_metrics.csv
- formal_bucket_ci_metrics.csv
- formal_tail_error_concentration.csv
- formal_underestimation_profile.csv
- formal_extreme_peak_profile.csv
- formal_sample_influence_audit.csv
- formal_readiness_decision.csv

Required logs generated:

- run_manifest.json
- input_file_hashes.json
- bootstrap_config.json
- metric_reproduction_check.json
- leakage_guard_report.json
- forbidden_scan_report.json
- table_alignment_check.json
- file_inventory.json

Figures generated:

- ci_forest_by_pool: 4 files
- subject_level_metric_distribution: 4 files
- tail_error_concentration: 2 files
- underestimation_profile: 2 files
- extreme_peak_cases_summary: 2 files

Validation already run:

- py_compile passed.
- Full v226 script run passed.
- ZIP testzip passed with None.
- required_files_missing is [].
- zip_bad_file is None.
- formal_model_lock_recheck has exactly:
  - loose_main_pool / avg_joint_focus
  - strict_main_pool / peak_floor_090
- metric reproduction within tolerance:
  - loose RMSE actual 0.5448840970647589, expected 0.544884, diff 9.706e-08
  - loose tail actual 0.6297521592665997, expected 0.629752, diff 1.593e-07
  - strict RMSE actual 0.571769914574812, expected 0.571770, diff 8.543e-08
  - strict tail actual 0.6583063251135349, expected 0.658306, diff 3.251e-07
- leakage_guard_report pass: true.
- forbidden_scan_report pass: true, hits: [].
- table_alignment_check pass: true.
- per_sample_rows: 2130.
- duplicate sample_id within pool/split: 0.
- missing formal prediction rows: 0.
- horizon length: 21.
- test rows: loose 184, strict 174.
- readiness decision accepted all checks:
  - model_lock_reproduced true
  - no_forbidden_columns true
  - no_diagnostic_rows true
  - sample_bootstrap_ci_generated true
  - subject_block_ci_generated true
  - subject_level_generated true
  - tail_concentration_generated true
  - underestimation_profile_generated true
  - extreme_peak_profile_generated true
  - sample_influence_generated true
  - figures_generated true
  - required_files_present true
  - zip_integrity_passed true
  - needs_new_model false
  - needs_gate_or_router false

Key CI results:

- Sample bootstrap test CI:
  - loose RMSE: 0.496066 to 0.593811
  - loose tail: 0.564811 to 0.693788
  - strict RMSE: 0.511036 to 0.635521
  - strict tail: 0.581652 to 0.736696
- Subject-block bootstrap test CI:
  - loose RMSE: 0.428783 to 0.599684
  - loose tail: 0.515881 to 0.687686
  - strict RMSE: 0.473689 to 0.615000
  - strict tail: 0.539479 to 0.706505

Tail concentration on test:

- loose:
  - top1 share 0.038498
  - top5 share 0.179141
  - top10 share 0.313389
  - top20pct share 0.659320
  - gini 0.612677
  - max sample tail 1.676098
- strict:
  - top1 share 0.053691
  - top5 share 0.205018
  - top10 share 0.354324
  - top20pct share 0.672493
  - gini 0.630911
  - max sample tail 2.012119

Note-layer sync completed:

- 04_project_logs/references/current-state.md
- 04_project_logs/reports/progress/decision_log.md
- 05_rebuild_from_raw_20260511/00_project_notes/PROJECT_STATUS_CN.md
- 05_rebuild_from_raw_20260511/00_project_notes/TASK_QUEUE_CN.md
- 05_rebuild_from_raw_20260511/00_project_notes/ARTIFACT_INDEX_CN.md
- 05_rebuild_from_raw_20260511/00_project_notes/daily_logs/2026-06-22.md

Allowed next-step directions should stay within the current guardrails. Examples:

- paper/report packaging from v225+v226 evidence;
- bounded robustness table/figure polish using existing v226 outputs only;
- claim/readiness audit against v225/v226 evidence;
- stop model work and move to writing/claim framing.

Do not request broad model search, v222b/v223, new tau, new gate/router, or
test-based retuning unless you explicitly explain which current stop condition
has been overturned and what new guardrails prevent leakage or local overfitting.
