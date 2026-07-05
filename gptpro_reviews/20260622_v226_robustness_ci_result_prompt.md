# GPTPro review request: v226 formal robustness / CI audit completed

请你复核本地 v226 执行结果，并给出下一轮 bounded 指令。注意：你的回复只作为外部审阅输入；本地仍以 guardrail、exact evaluator、leakage audit、ZIP/required-file 验证为准。

## Current local decision boundary

- v225 已被你接受为 complete。
- v226 按你的指令完成：`formal robustness / confidence-interval audit`。
- v226 是 audit-only + reporting-only。
- 本轮没有训练新模型，没有调 threshold/tau，没有创建 gate/router，没有运行 v222b/v223，没有使用 diagnostic-only rows 作为 formal 输入。
- Formal headline 仍锁定：
  - `loose_main_pool = avg_joint_focus`
  - `strict_main_pool = peak_floor_090`
- v222b_allowed=False、v223_allowed=False 仍保持。

## Local outputs

- Script:
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v226_formal_robustness_ci_audit_20260622.py`
- Output dir:
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622`
- ZIP:
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622\v226_formal_robustness_ci_audit_pack.zip`
- Report:
  - `reports/v226_formal_robustness_ci_audit_cn.md`

## Required files generated

Tables:

- `tables/formal_model_lock_recheck.csv`
- `tables/formal_metric_ci_sample_bootstrap.csv`
- `tables/formal_metric_ci_subject_block_bootstrap.csv`
- `tables/formal_subject_level_metrics.csv`
- `tables/formal_route_event_level_metrics.csv`
- `tables/formal_bucket_ci_metrics.csv`
- `tables/formal_tail_error_concentration.csv`
- `tables/formal_underestimation_profile.csv`
- `tables/formal_extreme_peak_profile.csv`
- `tables/formal_sample_influence_audit.csv`
- `tables/formal_readiness_decision.csv`

Figures:

- `figures/ci_forest_by_pool`
- `figures/subject_level_metric_distribution`
- `figures/tail_error_concentration`
- `figures/underestimation_profile`
- `figures/extreme_peak_cases_summary`

Logs:

- `logs/run_manifest.json`
- `logs/input_file_hashes.json`
- `logs/bootstrap_config.json`
- `logs/metric_reproduction_check.json`
- `logs/leakage_guard_report.json`
- `logs/forbidden_scan_report.json`
- `logs/table_alignment_check.json`
- `logs/file_inventory.json`

## Exact metric reproduction

All locked test metric reproduction checks pass within `1e-5`.

- `loose_main_pool / avg_joint_focus`
  - RMSE actual `0.5448840970647589`, expected `0.544884`, abs diff `9.706e-08`
  - tail RMSE actual `0.6297521592665997`, expected `0.629752`, abs diff `1.593e-07`
- `strict_main_pool / peak_floor_090`
  - RMSE actual `0.571769914574812`, expected `0.571770`, abs diff `8.543e-08`
  - tail RMSE actual `0.6583063251135349`, expected `0.658306`, abs diff `3.251e-07`

## Bootstrap CI results

Bootstrap config:

- random_seed = `20260622`
- n_bootstrap = `2000`
- ci_level = `0.95`
- sample bootstrap unit = sample row within pool/split
- subject-block bootstrap unit = subject block within pool/split

Sample bootstrap CI on test:

| pool | model | metric | point | ci_lower | ci_upper | n |
|---|---|---:|---:|---:|---:|---:|
| loose_main_pool | avg_joint_focus | rmse | 0.544884 | 0.496066 | 0.593811 | 184 |
| loose_main_pool | avg_joint_focus | tail_rmse | 0.629752 | 0.564811 | 0.693788 | 184 |
| strict_main_pool | peak_floor_090 | rmse | 0.571770 | 0.511036 | 0.635521 | 174 |
| strict_main_pool | peak_floor_090 | tail_rmse | 0.658306 | 0.581652 | 0.736696 | 174 |

Subject-block bootstrap CI on test:

| pool | model | metric | point | ci_lower | ci_upper | n | n_subjects |
|---|---|---:|---:|---:|---:|---:|---:|
| loose_main_pool | avg_joint_focus | rmse | 0.544884 | 0.428783 | 0.599684 | 184 | 4 |
| loose_main_pool | avg_joint_focus | tail_rmse | 0.629752 | 0.515881 | 0.687686 | 184 | 4 |
| strict_main_pool | peak_floor_090 | rmse | 0.571770 | 0.473689 | 0.615000 | 174 | 4 |
| strict_main_pool | peak_floor_090 | tail_rmse | 0.658306 | 0.539479 | 0.706505 | 174 | 4 |

Tail error concentration on test:

| pool | model | top1_share | top5_share | top10_share | top20pct_share | gini_tail_sse | max_sample_tail_rmse |
|---|---|---:|---:|---:|---:|---:|---:|
| loose_main_pool | avg_joint_focus | 0.038498 | 0.179141 | 0.313389 | 0.659320 | 0.612677 | 1.676098 |
| strict_main_pool | peak_floor_090 | 0.053691 | 0.205018 | 0.354324 | 0.672493 | 0.630911 | 2.012119 |

Readiness decision:

- total accepted_for_paper_main_result=True
- loose accepted_for_paper_main_result=True
- strict accepted_for_paper_main_result=True
- needs_new_model=False
- needs_gate_or_router=False
- needs_more_diagnostic_only=False

## Independent verification already run

- `python -m py_compile` passed.
- Full script run passed.
- ZIP `testzip() == None`.
- Required files missing `[]`.
- `formal_model_lock_recheck.csv` contains only:
  - `loose_main_pool, avg_joint_focus`
  - `strict_main_pool, peak_floor_090`
- `metric_reproduction_check.json`: pass.
- `leakage_guard_report.json`: pass.
- `forbidden_scan_report.json`: pass, hits `[]`.
- `table_alignment_check.json`: pass.
  - per_sample_rows `2130`
  - duplicate sample_id within pool/split `0`
  - missing formal prediction rows `0`
  - horizon length `21`
  - test n loose `184`
  - test n strict `174`
- `file_inventory.json`:
  - required files missing `[]`
  - zip_bad_file `None`
  - figure counts:
    - ci_forest_by_pool `4`
    - subject_level_metric_distribution `4`
    - tail_error_concentration `2`
    - underestimation_profile `2`
    - extreme_peak_cases_summary `2`

## What I need from you

Please give one bounded next instruction only.

Your reply should include:

1. Whether v226 is accepted as complete or rejected, and exact reason.
2. The next local task name/version.
3. The allowed input files.
4. The required output directory and required files.
5. The exact stop condition.
6. The validation commands/checks required before reporting back to you.

Allowed next-step directions should stay within the current guardrails. Examples:

- paper/report packaging from v225+v226 evidence;
- a bounded robustness table/figure polish pass using existing v226 outputs only;
- a claim/readiness audit against the v225/v226 evidence;
- stop model work and move to writing/claim framing.

Do not request broad model search, v222b/v223, new tau, new gate/router, or test-based retuning unless you explicitly explain which current stop condition has been overturned and what new guardrails prevent leakage or local overfitting.
