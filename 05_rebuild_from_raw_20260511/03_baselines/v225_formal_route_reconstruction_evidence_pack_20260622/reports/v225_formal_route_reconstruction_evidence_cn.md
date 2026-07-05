# v225 formal route reconstruction evidence pack

## 结论

本包按 GPTPro 指令只固化 formal baseline 证据：`loose_main_pool=avg_joint_focus`，`strict_main_pool=peak_floor_090`。
未训练新模型，未调 threshold/tau，未创建 router/gate，未运行 v222b/v223。

## Formal model lock

- `loose_main_pool`: `avg_joint_focus`
- `strict_main_pool`: `peak_floor_090`

## Locked test reproduction

- `loose_main_pool` / `avg_joint_focus`: RMSE=0.544884, tail RMSE=0.629752, under_rate=0.163043, n=184
- `strict_main_pool` / `peak_floor_090`: RMSE=0.571770, tail RMSE=0.658306, under_rate=0.137931, n=174

- metric reproduction pass: `True` (tolerance <= 1e-05)
- leakage guard pass: `True`
- forbidden scan pass: `True`

## Bucket summary (test split)

- `loose_main_pool` `extreme_peak=True`: n=6, RMSE=1.093467, tail=1.205023, under_rate=0.166667
- `loose_main_pool` `high_tail_error=True`: n=42, RMSE=0.905668, tail=1.095403, under_rate=0.428571
- `loose_main_pool` `multi_correction=True`: n=37, RMSE=0.447468, tail=0.469699, under_rate=0.135135
- `loose_main_pool` `normal_curve=True`: n=104, RMSE=0.400188, tail=0.454722, under_rate=0.134615
- `loose_main_pool` `reverse=True`: n=34, RMSE=0.777398, tail=0.918378, under_rate=0.294118
- `loose_main_pool` `scene_type=下坡弯道事件`: n=49, RMSE=0.574973, tail=0.700393, under_rate=0.183673
- `loose_main_pool` `scene_type=普通弯道事件`: n=16, RMSE=0.443934, tail=0.525139, under_rate=0.312500
- `loose_main_pool` `scene_type=直道事件`: n=119, RMSE=0.544468, tail=0.611671, under_rate=0.134454
- `loose_main_pool` `strong_steer=True`: n=80, RMSE=0.688964, tail=0.802090, under_rate=0.200000
- `loose_main_pool` `under_flag=True`: n=30, RMSE=0.731877, tail=0.938923, under_rate=1.000000
- `loose_main_pool` `vehicle_strong=True`: n=66, RMSE=0.566716, tail=0.655221, under_rate=0.212121
- `loose_main_pool` `zero_cross=True`: n=140, RMSE=0.537397, tail=0.627030, under_rate=0.178571
- `strict_main_pool` `extreme_peak=True`: n=6, RMSE=1.136487, tail=1.256405, under_rate=0.333333
- `strict_main_pool` `high_tail_error=True`: n=37, RMSE=0.976089, tail=1.183820, under_rate=0.486486
- `strict_main_pool` `multi_correction=True`: n=35, RMSE=0.470415, tail=0.502633, under_rate=0.114286
- `strict_main_pool` `normal_curve=True`: n=94, RMSE=0.430384, tail=0.495292, under_rate=0.063830
- `strict_main_pool` `reverse=True`: n=34, RMSE=0.767903, tail=0.890830, under_rate=0.294118
- `strict_main_pool` `scene_type=下坡弯道事件`: n=47, RMSE=0.613018, tail=0.743355, under_rate=0.191489
- `strict_main_pool` `scene_type=普通弯道事件`: n=13, RMSE=0.412622, tail=0.431767, under_rate=0.076923
- `strict_main_pool` `scene_type=直道事件`: n=114, RMSE=0.569770, tail=0.642168, under_rate=0.122807
- `strict_main_pool` `strong_steer=True`: n=80, RMSE=0.702429, tail=0.808907, under_rate=0.225000
- `strict_main_pool` `under_flag=True`: n=24, RMSE=0.892126, tail=1.097561, under_rate=1.000000
- `strict_main_pool` `vehicle_strong=True`: n=59, RMSE=0.637123, tail=0.743327, under_rate=0.203390
- `strict_main_pool` `zero_cross=True`: n=135, RMSE=0.542400, tail=0.624150, under_rate=0.148148

## Figure inventory

- `baseline_sufficient_cases/loose_main_pool`: 4 PNG
- `baseline_sufficient_cases/strict_main_pool`: 4 PNG
- `formal_examples/loose_main_pool`: 6 PNG
- `formal_examples/strict_main_pool`: 6 PNG
- `strong_under_cases/loose_main_pool`: 4 PNG
- `strong_under_cases/strict_main_pool`: 4 PNG
- `worst_tail_cases/loose_main_pool`: 6 PNG
- `worst_tail_cases/strict_main_pool`: 6 PNG

## Diagnostic appendix boundary

下列内容仅用于 diagnostic appendix，不进入 formal usage / formal selected config / formal leaderboard：
- `future_route_decision`
- `oracle_safe_gate`
- `ridge_residual_peakfloor`
- `v222a_bounded_residual`
- `v222a_noharm_gate`

## Required files

- `tables/formal_model_lock.csv`
- `tables/formal_reconstruction_metrics_overall.csv`
- `tables/formal_reconstruction_metrics_by_pool.csv`
- `tables/formal_reconstruction_metrics_by_bucket.csv`
- `tables/formal_reconstruction_metrics_by_route_event.csv`
- `tables/per_sample_formal_reconstruction_eval.csv`
- `tables/formal_failure_case_index.csv`
- `tables/diagnostic_only_v222a_closeout_summary.csv`
- `tables/excluded_diagnostic_models_audit.csv`
- `reports/v225_formal_route_reconstruction_evidence_cn.md`
- `logs/run_manifest.json`
- `logs/leakage_guard_report.json`
- `logs/forbidden_scan_report.json`
- `logs/metric_reproduction_check.json`
- `logs/file_inventory.json`
- `v225_formal_route_reconstruction_evidence_pack.zip`

