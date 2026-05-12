# 阶段产物索引

更新时间：2026-05-12 11:43:16

## 阶段 0：旧流程冻结与重建准则

- 阶段 0 规则说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/stage00_old_flow_freeze_and_rules_cn.md`
- 阶段 0 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage00_user_summary_cn.md`
- 旧流程参考定位：`04_project_logs/reports/progress/experiment_registry.md`、`04_project_logs/reports/physio_to_g14_progress_review_20260511/`
- GPTPro 原始数据重建建议：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_replies/20260512_rebuild_steering_reply_summary_cn.md`

## 阶段 1：原始数据审计

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage01_user_summary_cn.md`
- 审计中文总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/raw_data_audit_summary_cn.md`
- 文件清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/raw_file_inventory.csv`
- 字段报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/raw_schema_report.csv`
- 被试/记录/模态矩阵：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/subject_session_modality_matrix.csv`
- 时间连续性报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/timestamp_continuity_report.csv`
- 采样率报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/sampling_rate_report.csv`
- 模态重叠报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/modality_overlap_report.csv`
- 信号质量报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/signal_quality_report.csv`
- EEG 初审报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/eeg_artifact_report.csv`
- 泄漏风险报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/leakage_risk_report.csv`
- 审计脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/scripts/raw_csv_audit.py`
- 审计图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/figures/audit`
- 审计运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/logs/raw_csv_audit.log`
- 阶段 0/1 完成审计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/STAGE00_01_COMPLETION_AUDIT_CN.md`

## 阶段 2：事件锚点与样本清单重建

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_user_summary_cn.md`
- 事件锚点重建总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/event_anchor_rebuild_summary_cn.md`
- 数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_v0_2_cn.md`
- 处理后车辆窗口说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/processed_vehicle_windows_v0_2_cn.md`
- 候选事件总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/candidate_events_master.csv`
- 样本总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/samples_master.csv`
- 样本 JSONL：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/samples_master.jsonl`
- split 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/split_table.csv`
- split 可行性报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/split_feasibility_report.csv`
- 锚点来源统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/anchor_source_inventory.csv`
- 道路设计清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/road_design_inventory.csv`
- 窗口配置对比：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/window_config_comparison.csv`
- 锚点来源近邻对照：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/anchor_source_comparison.csv`
- 车辆重采样状态：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/vehicle_resample_status.csv`
- 阶段 2 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_stage2_samples.py`
- 处理后车辆窗口脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_processed_vehicle_windows.py`
- 阶段 2 图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/figures`
- 处理后车辆窗口数组：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/arrays`
- 处理后车辆窗口索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/tables`

## 阶段 3：无学习基线与纯车辆基线

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_user_summary_cn.md`
- 阶段 3 基线总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_baseline_summary_cn.md`
- 阶段 3 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/evaluate_stage3_vehicle_baselines.py`
- 汇总指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_baseline_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_per_sample_metrics.csv`
- 各窗口/切分测试集最好行：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_best_test_by_window_split.csv`
- ridge 训练信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_ridge_model_info.csv`
- 固定画图样本集：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/tables/stage03_fixed_plot_sample_set.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_fixed_predictions_pre2_session_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_bad_samples_pre2_session_test_ridge.png`
- 运行 stdout：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/logs/evaluate_stage3_vehicle_baselines.stdout.log`
- 运行 stderr：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/logs/evaluate_stage3_vehicle_baselines.stderr.log`
- JSON 摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/logs/stage03_baseline_summary.json`

## 服务器日志

- 本阶段未使用服务器。

## 重要 Git commit

- `e9d302f Add raw rebuild stage 0 and 1 audit`
- `bae5618 Add stage 0 and 1 completion audit`
- `9bef223 Record completion audit commit`
- `114208d Add stage 2 samples and processed vehicle windows`
- `b61e427 Add stage 3 vehicle baseline evaluation`

## 适合用户/老师直接查看的材料

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage01_user_summary_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_user_summary_cn.md`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/processed_vehicle_windows_v0_2_cn.md`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_v0_2_cn.md`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_user_summary_cn.md`
6. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_baseline_summary_cn.md`
7. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_fixed_predictions_pre2_session_test.png`
8. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_bad_samples_pre2_session_test_ridge.png`
9. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/figures/stage02_anchor_overlay_example.png`
10. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/samples_master.csv`
11. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/tables/processed_vehicle_window_outputs.csv`
12. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/STAGE00_01_COMPLETION_AUDIT_CN.md`
