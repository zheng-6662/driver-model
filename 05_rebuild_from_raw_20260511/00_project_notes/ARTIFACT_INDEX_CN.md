# 阶段产物索引

## 最新新增：道路设定引导的车辆失稳事件判定 v0.1

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/road_guided_instability_v0_1_cn.md`
- 数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_road_guided_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_road_guided_instability_events_v0_1.py`
- 全量判定表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_instability_events_v0_1.csv`
- 自动采用表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_auto_accepted_events_v0_1.csv`
- 中间复核队列表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_review_queue_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_instability_summary_v0_1.csv`
- 道路模块交叉表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_module_summary_v0_1.csv`
- 人工抽查校准表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_manual_calibration_v0_1.csv`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/logs/road_guided_instability_run_summary_v0_1.json`
- 旧日志依据：`F:/data_set_process/data_process/04_project_logs/reports/trigger_response_lag_20260421/TASK_DEFINITION_AND_EVENT_LOGIC.md`
- 道路设计依据：`F:/data_set_process/data_process/01_datasets/多模态数据/被试数据集合/道路信息/full_centerline_layout.csv`
- 重要 Git commit：`ad981f6 Add road-guided instability event adjudication`
- 适合用户/老师直接查看：优先看中文说明和自动采用表；这版用于替代“全人工逐条标注”的第一轮车辆失稳事件筛选。

## 最新新增：全部原始车辆 CSV 失稳样本重筛 v0.1

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/all_raw_vehicle_instability_rescreen_v0_1_cn.md`
- 数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_all_raw_vehicle_instability_rescreen_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/rescreen_all_raw_vehicle_instability_v0_1.py`
- 全量候选表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_candidates_v0_1.csv`
- 高置信主清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_primary_high_confidence_v0_1.csv`
- 自动采用扩展清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_auto_accepted_v0_1.csv`
- 中间复核队列：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_review_queue_v0_1.csv`
- 低证据剔除表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_rejected_v0_1.csv`
- 文件读取状态：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_rescreen_file_status_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_summary_v0_1.csv`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/logs/all_raw_vehicle_instability_rescreen_run_summary_v0_1.json`
- 重要 Git commit：`12c30cf Rescreen all raw vehicle instability events`
- 适合用户/老师直接查看：优先看中文说明、数据版本卡和高置信主清单；该版本覆盖全部 91 个原始车辆 CSV。

更新时间：2026-05-12 14:03:26

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
- 人工事件标注审查包 v0.1 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_label_review_pack_v0_1_cn.md`
- 人工事件标注审查 HTML：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/review_index.html`
- 人工标注模板：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/manual_event_labels_template_v0_1.csv`
- 人工审查记录清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/session_review_manifest_v0_1.csv`
- 人工审查时间线图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/figures`
- 人工标注包脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_manual_event_label_review_pack.py`
- 人工标注包运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/logs`
- 键盘式人工标注播放器页面：`http://127.0.0.1:8766/`
- 键盘式人工标注播放器说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_keyboard_player_v0_1_cn.md`；当前页面已升级为候选段审查模式，初版整段播放器可从 `/legacy` 查看。
- 键盘式人工标注播放器脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/run_manual_event_keyboard_player.py`
- 键盘式人工标签输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv`
- 键盘式人工标注播放器日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/logs`
- Codex 自动事件审阅说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/codex_event_review_v0_1_cn.md`
- Codex 自动事件审阅脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_codex_event_review_v0_1.py`
- Codex 自动审阅总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_reviewed_event_labels_v0_1.csv`
- Codex 自动采用标签：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_auto_accepted_event_labels_v0_1.csv`
- Codex 需要复核队列：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_needs_human_review_v0_1.csv`
- Codex 自动审阅汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_event_review_summary_v0_1.csv`
- Codex 自动审阅图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/figures`
- Codex 自动审阅运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/logs`

## 阶段 2 修正：车辆失稳事件候选 v0.1

- 说明：`codex_event_review_v0_1` 的 404 个样本是弯道/道路曲率候选，已降级为道路上下文参考；当前主线改为车辆失稳候选。
- 车辆失稳中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/instability_event_review_v0_1_cn.md`
- 车辆失稳数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_v0_1_cn.md`
- 车辆失稳审阅脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_instability_event_review_v0_1.py`
- 全量车辆失稳候选表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`
- 自动采用车辆失稳候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_auto_accepted_events_v0_1.csv`
- 需要人工复核车辆失稳候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_needs_human_review_v0_1.csv`
- 车辆失稳审阅汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_event_review_summary_v0_1.csv`
- 车辆失稳概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/instability_event_score_overview_v0_1.png`
- 车辆失稳示例图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures`
- 车辆失稳运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/logs/build_instability_event_review_v0_1.json`
- 本地车辆失稳审查页面：`http://127.0.0.1:8766/`
- 车辆失稳人工标签输出：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_instability_event_labels_v0_1.csv`

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
- v0.3 诊断总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_diagnostics_v0_3_cn.md`
- v0.3 诊断脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_diagnostics_v0_3.py`
- v0.3 无被试 ID 模型指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_stronger_vehicle_metrics_v0_3.csv`
- v0.3 模型对照表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_vehicle_model_comparison_v0_3.csv`
- v0.3 坏样本诊断表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_bad_sample_diagnostics_v0_3.csv`
- v0.3 固定图样本诊断表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_fixed_plot_diagnostics_v0_3.csv`
- v0.3 错误桶：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_error_bucket_summary_pre2_session_v0_3.csv`
- v0.3 小样本过拟合测试：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/tables/stage03_small_overfit_report_v0_3.csv`
- v0.3 模型对照图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/figures/stage03_pre2_session_model_rmse_comparison_v0_3.png`
- v0.3 坏样本诊断图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/figures/stage03_pre2_session_bad_sample_diagnostic_v0_3.png`
- v0.3 运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/logs`
- v0.4 RBF KRR 候选模型卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_rbf_krr_candidate_model_card_v0_4_cn.md`
- v0.4 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_rbf_krr_model_card_v0_4.py`
- v0.4 候选指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/tables/stage03_rbf_krr_candidate_metrics_v0_4.csv`
- v0.4 分被试表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/tables/stage03_rbf_krr_per_subject_v0_4.csv`
- v0.4 分响应组表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/tables/stage03_rbf_krr_response_group_summary_v0_4.csv`
- v0.4 画图样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/tables/stage03_rbf_krr_plot_sample_set_v0_4.csv`
- v0.4 pre2 固定样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre2_session_v0_4.png`
- v0.4 pre2 坏样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre2_session_v0_4.png`
- v0.4 pre3 固定样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre3_session_v0_4.png`
- v0.4 pre3 坏样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre3_session_v0_4.png`
- v0.4 运行日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/logs`

## 服务器日志

- 本阶段未使用服务器。

## 重要 Git commit

- `e9d302f Add raw rebuild stage 0 and 1 audit`
- `bae5618 Add stage 0 and 1 completion audit`
- `9bef223 Record completion audit commit`
- `114208d Add stage 2 samples and processed vehicle windows`
- `b61e427 Add stage 3 vehicle baseline evaluation`
- `a2379c5 Record stage 3 artifact commit`
- `82d6a1a Add stage 3 no-subject vehicle diagnostics`
- `db1ff13 Record stage 3 no-subject diagnostics commit`
- `6c3c9f3 Add stage 3 RBF KRR model card`
- `9907aa5 Add manual event labeling review pack`
- `cf9d06b Add keyboard event labeling player`
- `5b8abbf Focus manual labeler on candidate event segments`
- `4019653 Clarify manual labeler event line legend`
- `2819f3f Add codex automatic event review`
- `d0dbf5d Rebuild event review around vehicle instability`

## 适合用户/老师直接查看的材料

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage01_user_summary_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_user_summary_cn.md`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/processed_vehicle_windows_v0_2_cn.md`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_v0_2_cn.md`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_user_summary_cn.md`
6. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_baseline_summary_cn.md`
7. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_diagnostics_v0_3_cn.md`
8. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_rbf_krr_candidate_model_card_v0_4_cn.md`
9. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre2_session_v0_4.png`
10. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre2_session_v0_4.png`
11. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre3_session_v0_4.png`
12. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre3_session_v0_4.png`
13. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/figures/stage03_pre2_session_model_rmse_comparison_v0_3.png`
14. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_diagnostics_v0_3/figures/stage03_pre2_session_bad_sample_diagnostic_v0_3.png`
15. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_fixed_predictions_pre2_session_test.png`
16. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_baselines_v0_2/figures/stage03_bad_samples_pre2_session_test_ridge.png`
17. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/figures/stage02_anchor_overlay_example.png`
18. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/samples_master.csv`
19. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/tables/processed_vehicle_window_outputs.csv`
20. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/00_project_notes/STAGE00_01_COMPLETION_AUDIT_CN.md`
21. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/review_index.html`
22. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_label_review_pack_v0_1_cn.md`
23. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/manual_event_labels_template_v0_1.csv`
24. `http://127.0.0.1:8766/`
25. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_event_keyboard_player_v0_1_cn.md`
26. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_event_labels_v0_1.csv`
27. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/codex_event_review_v0_1_cn.md`
28. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_auto_accepted_event_labels_v0_1.csv`
29. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_needs_human_review_v0_1.csv`

## 阶段 2 追加：道路事件位置与锚点重建审计 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_road_anchor_audit_user_summary_cn.md`
- 中文审计报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/road_event_anchor_audit_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_road_event_anchor_audit_v0_1.py`
- 道路模块位置表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/road_event_position_map_v0_1.csv`
- 每条记录道路映射摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/session_road_mapping_summary_v0_1.csv`
- 每条记录道路模块进入/离开时间：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/session_module_entry_exit_v0_1.csv`
- 旧锚点对齐表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/old_new_anchor_alignment_v0_1.csv`
- 道路引导候选对齐表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/road_guided_anchor_alignment_v0_1.csv`
- 审计汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/tables/road_event_anchor_audit_summary_v0_1.csv`
- 道路模块位置图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/figures/road_event_position_map_v0_1.png`
- 锚点审计概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/figures/road_anchor_audit_overview_v0_1.png`
- 代表样本面板目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/figures/representative_panels`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_event_anchor_audit_v0_1/logs/road_event_anchor_audit_run_summary_v0_1.json`

## 阶段 2/3 追加：旧代码测试全原始车辆失稳高置信样本 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_oldcode_instability_user_summary_cn.md`
- 旧车辆代码诊断报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_baseline_on_instability_v0_1_cn.md`
- 旧深度模型 smoke 报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_direct_smoke_on_instability_v0_1_cn.md`
- 旧代码兼容数据包说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/instability_oldcode_dataset_v0_1_cn.md`
- 窗口生成脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_instability_oldcode_windows_v0_1.py`
- 旧车辆基线诊断脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/evaluate_oldcode_vehicle_baselines_on_instability_v0_1.py`
- 处理后车辆窗口数组：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/arrays`
- 旧代码 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_session_level_split.csv`
- 旧代码可用性表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/instability_highconf_events_oldcode_eligibility_v0_1.csv`
- 窗口索引和 split 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables`
- 旧车辆基线指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_baseline_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_per_sample_metrics.csv`
- 各窗口/切分最佳测试行：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/tables/oldcode_instability_best_test_by_window_split.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/figures/oldcode_fixed_predictions_pre2_session_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_baselines_on_instability_v0_1/figures/oldcode_bad_samples_pre2_session_test_ridge.png`
- 旧 manifest loader smoke 记录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/logs/oldcode_manifest_loader_smoke_check.json`
- 旧深度模型 smoke run 目录：`F:/data_set_process/data_process/tmp/event_conditioned_runs/OLD_SMOKE_INSTABILITY_HIGHCONF_V0_1_20260512_165950`
