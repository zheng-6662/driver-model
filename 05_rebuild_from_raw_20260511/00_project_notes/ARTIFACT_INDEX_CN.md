## 2026-05-20 v0.5 服务器处理后样本重筛 + 被试划分旧流程车辆-only

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_server_aligned_subject_oldflow_fair09_user_summary_cn.md`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v05_server_aligned_subject_oldflow_fair09.py`
- v0.5 样本筛选表目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_server_aligned_v0_5\tables`
- 旧流程 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\stage03_v05_server_aligned_subject_oldflow_fair09\tables\oldflow_fair09_vehicle_only_server_aligned_v05_subject_split_manifest.csv`
- 运行记录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\server_aligned_v05_subject_oldflow_fair09_run_record.csv`
- 分被试样本级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\test_subject_sample_metrics_v0_5.csv`
- 分道路类型样本级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\test_road_type_sample_metrics_v0_5.csv`
- 分机制标签样本级指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_server_aligned_subject_oldflow_fair09\tables\test_mechanism_sample_metrics_v0_5.csv`
- 本地运行目录：`F:\data_set_process\data_process\tmp\event_conditioned_runs\V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144`
- 预测总览图：`F:\data_set_process\data_process\tmp\event_conditioned_runs\V05_SERVER_ALIGNED_SUBJECT_FAIR09_vehicle_only_seed2026_20260520_120144\prediction_figures\test\overview.png`
- 服务器训练日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\00_project_notes\server_logs\stage03_v05_server_aligned_subject_train_latest.log`
- 结果摘要：loader 保留 1376 个样本，train/val/test=953/260/163；车辆-only test RMSE=0.3386，primary=0.2184，tail=0.3105，selection=0.8206。
- Git 提交：`77f5809 stage03: run server-aligned subject fair09 vehicle baseline`

---

## 2026-05-19 v0.3 样本筛选策略 GPU 快速对比完成

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_screening_sweep_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu\tables\v03_screening_sweep_gpu_summary.csv`
- 排序表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu\tables\v03_screening_sweep_gpu_ranking.csv`
- 额外样本来源统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu\tables\v03_screening_sweep_gpu_extra_source_counts.csv`
- 服务器日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\00_project_notes\server_logs\stage03_v03_screening_sweep_gpu_20260519_211258.log`
- 代码入口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v03_screening_sweep_gpu.py`
- 当前结论：`s16_weakpost_lat` 排名第一，但需继续复核横向偏移坐标风险和 16 个新增样本。

---

## 2026-05-19 v0.3 样本筛选策略 GPU 快速对比

- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v03_screening_sweep_gpu.py`
- CPU 对照脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v03_screening_sweep.py`
- 服务器任务：screen `v03gpu`
- 服务器日志：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v03_screening_sweep_gpu_20260519_211258.log`
- 预计本地结果目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_screening_sweep_gpu`
- 预计用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_screening_sweep_gpu_user_summary_cn.md`
- 状态：运行中，完成后拉回结果并更新本索引。

---

## 2026-05-18 车辆响应锚点前方向盘动作重新筛选 v0.2

- 用户查看版：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_vehicle_response_presteer_rescreen_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\vehicle_response_presteer_rescreen_v0_2_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_candidates_v0_2.csv`
- P1 最干净核心样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\primary_roll_presteer_events_P1_v0_2.csv`
- P2 最干净次级样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\secondary_lateral_presteer_events_P2_v0_2.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_summary_v0_2.csv`
- 分场景表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_by_module_v0_2.csv`
- 时间差表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_latency_quantiles_v0_2.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_review_panel_index_v0_2.csv`

## 2026-05-18 车辆响应锚点前方向盘动作重新筛选 v0.2

- 用户查看版：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_vehicle_response_presteer_rescreen_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\vehicle_response_presteer_rescreen_v0_2_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_candidates_v0_2.csv`
- P1 核心样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\primary_roll_presteer_events_P1_v0_2.csv`
- P2 次级样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\secondary_lateral_presteer_events_P2_v0_2.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_summary_v0_2.csv`
- 分场景表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_by_module_v0_2.csv`
- 时间差表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_latency_quantiles_v0_2.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_review_panel_index_v0_2.csv`

## 2026-05-14 方向盘动作候选漏斗审计 v0.1

- 报告：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\funnel_audit_v0_1\steering_episode_funnel_audit_v0_1.md`
- 逐记录漏斗表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\funnel_audit_v0_1\steering_funnel_by_record_v0_1.csv`
- 候选明细表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\funnel_audit_v0_1\steering_funnel_candidates_v0_1.csv`
- 宽松候选池：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\funnel_audit_v0_1\loose_steering_candidates_v0_1.csv`
- 严格通过表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\funnel_audit_v0_1\strict_steering_episode_candidates_v0_1.csv`

## 2026-05-14 方向盘动作 episode 样本重建 v0.6

- 汇总报告：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episode_summary_v0_6.md`
- 主 episode 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episodes_all_v0_6.csv`
- P1 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\primary_positive_episodes_P1_v0_6.csv`
- P2 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\secondary_episodes_P2_v0_6.csv`
- C 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\context_control_C_v0_6.csv`
- N 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\trigger_no_effect_N_v0_6.csv`
- U 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\manual_review_U_v0_6.csv`
- X 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\excluded_X_v0_6.csv`
- 复核图目录：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figures`
- 复核图索引：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figure_index_v0_6.csv`
- 日志：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\build_event_episodes_v0_6.log`

## 2026-05-14 方向盘动作 episode 样本重建 v0.6

- 汇总报告：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episode_summary_v0_6.md`
- 主 episode 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episodes_all_v0_6.csv`
- P1 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\primary_positive_episodes_P1_v0_6.csv`
- P2 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\secondary_episodes_P2_v0_6.csv`
- C 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\context_control_C_v0_6.csv`
- N 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\trigger_no_effect_N_v0_6.csv`
- U 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\manual_review_U_v0_6.csv`
- X 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\excluded_X_v0_6.csv`
- 复核图目录：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figures`
- 复核图索引：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figure_index_v0_6.csv`
- 日志：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\build_event_episodes_v0_6.log`

## 2026-05-14 方向盘动作 episode 样本重建 v0.6

- 汇总报告：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episode_summary_v0_6.md`
- 主 episode 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\event_episodes_all_v0_6.csv`
- P1 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\primary_positive_episodes_P1_v0_6.csv`
- P2 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\secondary_episodes_P2_v0_6.csv`
- C 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\context_control_C_v0_6.csv`
- N 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\trigger_no_effect_N_v0_6.csv`
- U 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\manual_review_U_v0_6.csv`
- X 表：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\excluded_X_v0_6.csv`
- 复核图目录：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figures`
- 复核图索引：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\review_figure_index_v0_6.csv`
- 日志：`F:\data_set_process\data_process\outputs\event_episodes_v0_6\build_event_episodes_v0_6.log`

## 2026-05-14 方向盘到车辆动态时间差审计

- 用户查看版：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\steer_to_vehicle_dynamics_latency_v0_1_cn.md`
- 明细表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_summary_v0_1.csv`
- 分组表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_by_bucket_module_v0_1.csv`
- 分位数表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_quantiles_v0_1.csv`
- 代表性复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`
- 代表性复核图数量：24

图表：
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_histogram_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_bucket_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_module_box_v0_1.png`

## 2026-05-14 方向盘到车辆动态时间差审计

- 用户查看版：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\steer_to_vehicle_dynamics_latency_v0_1_cn.md`
- 明细表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_summary_v0_1.csv`
- 分组表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_by_bucket_module_v0_1.csv`
- 分位数表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_quantiles_v0_1.csv`
- 代表性复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`
- 代表性复核图数量：24

图表：
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_histogram_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_bucket_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_module_box_v0_1.png`

## 2026-05-14 方向盘到车辆动态时间差审计

- 用户查看版：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_steer_to_vehicle_dynamics_latency_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\steer_to_vehicle_dynamics_latency_v0_1_cn.md`
- 明细表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_summary_v0_1.csv`
- 分组表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_by_bucket_module_v0_1.csv`
- 分位数表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_quantiles_v0_1.csv`
- 代表性复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`
- 代表性复核图数量：24

图表：
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_histogram_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_bucket_v0_1.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_by_module_box_v0_1.png`

# 阶段产物索引

## 最新更新：2026-05-13 08:09

## Stage 7j session 多折稳定性验证 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07j_session_cv_stability_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07j_session_cv_stability_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07j_session_cv_stability_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/logs/stage07j_session_cv_stability_summary.json`
- session CV split 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_session_cv_split_table.csv`
- 候选指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_candidate_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_candidate_per_sample_metrics.csv`
- 候选分数表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_candidate_score_table.csv`
- policy fold 指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_policy_fold_metrics.csv`
- policy 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_policy_aggregate.csv`
- 原始 val gate 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_original_val_gate_selection_table.csv`
- feature audit：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_feature_audit.csv`
- allowed features：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_allowed_features.csv`
- fold RBF fit info：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_fold_rbf_fit_info.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_gate_table.csv`
- policy fold delta 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/figures/stage07j_policy_fold_deltas.png`
- 选中模型计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/figures/stage07j_selected_model_counts.png`
- val/test delta 散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/figures/stage07j_candidate_val_test_delta_scatter.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器，未读取服务器指令与密码文件。
- 重要 Git commit：`11296297 Add stage7j session cv stability audit`。
- 适合用户/老师直接查看：优先看用户查看版总结、policy 汇总、gate 表和 fold delta 图。

## Stage 7i 稳定性校准候选选择 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07i_stability_calibrated_selection_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07i_stability_calibrated_selection_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07i_stability_calibrated_selection_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/logs/stage07i_stability_calibrated_selection_summary.json`
- 候选分数表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_candidate_score_table.csv`
- policy split 指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_policy_split_metrics.csv`
- policy test 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_policy_test_summary.csv`
- 逐样本收益表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_selected_policy_gain_samples.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_gate_table.csv`
- policy 对照图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/figures/stage07i_policy_summary.png`
- 稳定性分数组成图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/figures/stage07i_stability_score_components.png`
- 逐样本收益分布图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/figures/stage07i_selected_gain_distribution.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器，未读取服务器指令与密码文件。
- 重要 Git commit：`d294a520 Add stage7i stability calibrated selection`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、policy test 汇总和 policy 对照图。

## 最新新增：Stage 7b 非 oracle top-K selector 轻量实验 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07b_non_oracle_topk_selector_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07b_non_oracle_topk_selector_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07b_non_oracle_topk_selector_v0_1.py`
- feature audit：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_feature_audit.csv`
- allowed features：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_allowed_features.csv`
- model info：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_model_info.csv`
- selector decisions：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_selector_decisions.csv`
- all policy metrics：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_all_policy_metrics.csv`
- selected policy metrics：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_selected_policy_metrics.csv`
- decision diagnostics：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_decision_diagnostics.csv`
- coverage-risk：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_coverage_risk.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_gate_table.csv`
- RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/figures/stage07b_selector_test_rmse.png`
- choice accuracy 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/figures/stage07b_selector_choice_accuracy.png`
- coverage-risk 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/figures/stage07b_coverage_risk.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/logs/stage07b_selector_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`d431cd11 Add stage7b non-oracle topk selector`。
- 适合用户/老师直接查看：用户查看版总结、gate 表、selected policy metrics、feature audit。

## 最新新增：Stage 7a 非 oracle 多候选选择协议 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07a_non_oracle_selection_protocol_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07a_non_oracle_selection_protocol_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07a_non_oracle_selection_protocol_v0_1.py`
- 候选池 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_candidate_pool_manifest.csv`
- 特征守卫表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_feature_guard_table.csv`
- 选择流程表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_selection_protocol.csv`
- 评价计划表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_evaluation_plan.csv`
- 固定图协议：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_fixed_plot_protocol.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_gate_table.csv`
- 候选池 RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/figures/stage07a_candidate_pool_rmse.png`
- gate 状态图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/figures/stage07a_protocol_gate_status.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/logs/stage07a_protocol_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`dfacb38d Add stage7a non-oracle selection protocol`。
- 适合用户/老师直接查看：用户查看版总结、特征守卫表、选择流程表、gate 表。

## 最新新增：Stage 6e 多候选 oracle gap 复核 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06e_multicandidate_oracle_gap_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06e_multicandidate_oracle_gap_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/scripts/stage06e_multicandidate_oracle_gap_v0_1.py`
- source manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_source_manifest.csv`
- 候选可用性表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_model_availability.csv`
- 全指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_all_metrics.csv`
- oracle gap 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_gap_table.csv`
- oracle winner 明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_winner_detail.csv`
- oracle winner 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_winner_summary.csv`
- oracle gain 样本明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_gain_sample_detail.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/tables/multicandidate_oracle_gap_gate_table.csv`
- RMSE gap 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/figures/multicandidate_oracle_gap_rmse.png`
- winner count 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/figures/multicandidate_oracle_winner_counts.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06e_multicandidate_oracle_gap_v0_1/logs/multicandidate_oracle_gap_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`cb4d8eec Add stage6e multicandidate oracle gap audit`。
- 适合用户/老师直接查看：用户查看版总结、oracle gap 表、winner 汇总、RMSE gap 图。

## 最新新增：Stage 6d RBF/KNN reliability gate v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06d_reliability_gate_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06d_reliability_gate_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/scripts/stage06d_reliability_gate_v0_1.py`
- 全阈值指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_all_threshold_metrics.csv`
- 选中 policy 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_selected_policies.csv`
- policy 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_policy_metrics.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_gate_table.csv`
- best confusion 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_best_confusion.csv`
- RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/figures/reliability_gate_test_rmse.png`
- 物理指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/figures/reliability_gate_physical_metrics.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/logs/reliability_gate_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`4264db88 Add stage6d reliability gate`。
- 适合用户/老师直接查看：用户查看版总结、gate 表、policy 指标表、物理指标图。

## 最新新增：Stage 6c selector feature revision v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06c_selector_feature_revision_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06c_selector_feature_revision_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/scripts/stage06c_selector_feature_revision_v0_1.py`
- 特征协议表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_feature_manifest.csv`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_metrics.csv`
- 阈值扫描表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_threshold_sweep.csv`
- 候选明细表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_candidate_details.csv`
- 最佳 selector 明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_best_detail.csv`
- 最佳 selector 混淆表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_best_confusion.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/tables/selector_revision_gate_table.csv`
- RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/figures/selector_revision_test_rmse.png`
- 物理指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/figures/selector_revision_physical_metrics.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06c_selector_feature_revision_v0_1/logs/selector_feature_revision_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`eae76c2f Add stage6c selector feature revision`。
- 适合用户/老师直接查看：用户查看版总结、gate 表、指标表、物理指标图。

## 最新新增：Stage 6b RBF/keypoint 选择器错误复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06b_keypoint_selector_error_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06b_keypoint_selector_error_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/scripts/stage06b_keypoint_selector_error_review_v0_1.py`
- 选择器样本明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_sample_detail.csv`
- 混淆表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_confusion_table.csv`
- 分组摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_group_summary.csv`
- top regret 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_top_regret_samples.csv`
- 漏选 keypoint 收益样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_missed_keypoint_gain_samples.csv`
- 错选 keypoint 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_false_keypoint_samples.csv`
- 下一步动作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/tables/keypoint_selector_next_actions.csv`
- 混淆矩阵图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_confusion_matrix.png`
- top regret 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_top_regret_samples.png`
- probability vs gain 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_probability_vs_gain.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/logs/keypoint_selector_error_review_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`753525fd Add stage6b keypoint selector error review`。
- 适合用户/老师直接查看：用户查看版总结、混淆表、top regret 样本、probability vs gain 图。

## 最新新增：阶段 6 车辆-only 结构化路线审计 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06_vehicle_only_structured_route_audit_user_summary_cn.md`
- 阶段通用用户总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage06_vehicle_only_structured_route_audit_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/scripts/stage06_vehicle_only_structured_route_audit_v0_1.py`
- 车辆-only候选 scorecard：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_candidate_scorecard.csv`
- 相对 RBF delta 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_metric_delta_vs_rbf.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_route_gate_table.csv`
- 下一步动作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/tables/vehicle_structured_next_actions.csv`
- RMSE 汇总图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/figures/vehicle_structured_route_rmse_summary.png`
- 相对 RBF delta 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/figures/vehicle_structured_route_delta_vs_rbf.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/logs/vehicle_structured_route_audit_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`b4d7ac20 Add stage6 vehicle structured route audit`。
- 适合用户/老师直接查看：用户查看版总结、gate 表、scorecard、相对 RBF delta 图。

## 最新新增：阶段 4 连续风格协议与候选特征 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_user_summary_cn.md`
- 同内容阶段 4 协议总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_continuous_style_protocol_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_continuous_style_protocol_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_continuous_style_protocol_v0_1.py`
- 候选风格 long 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_feature_candidate_long.csv`
- 候选风格 wide 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_feature_candidate_wide.csv`
- train-only 标准化后 wide 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_feature_candidate_wide_trainz_session_split.csv`
- train-only 标准化参数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_train_only_scaler_session_split.csv`
- 来源协议表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_source_protocol_table.csv`
- 泄漏边界表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_leakage_guard_table.csv`
- 置乱对照计划：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_permutation_plan.csv`
- split 可行性表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_split_feasibility.csv`
- 被试/道路耦合审计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_subject_road_coupling_audit.csv`
- 阶段 4 gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_protocol_gate_table.csv`
- 风格窗口可用性图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/figures/style_feature_availability_by_window.png`
- split-道路分布图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/figures/style_split_road_distribution_heatmap.png`
- 被试-道路耦合热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/figures/style_subject_road_coupling_heatmap.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/logs/run_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`012f4803 Add continuous style protocol audit`。
- 适合用户/老师直接查看：用户查看版总结、阶段 4 gate 表、泄漏边界表、风格窗口可用性图、被试-道路耦合热图。

## 最新新增：阶段 3 RBF 主参照冻结审计 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_reference_freeze_audit_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1.py`
- RBF 指标画像：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_metric_profile.csv`
- RBF 失败画像：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_failure_profile.csv`
- RBF top bad 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_top_bad_samples.csv`
- 冻结 gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_freeze_gate_table.csv`
- 稳健性快照：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_robustness_snapshot.csv`
- 失败画像图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/figures/rbf_reference_failure_profile.png`
- 关键指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/figures/rbf_reference_key_metrics.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/logs/rbf_reference_freeze_audit_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`112824f7 Add rbf reference freeze audit`。
- 适合用户/老师直接查看：用户查看版总结、冻结 gate 表、失败画像图、关键指标图、RBF top bad 样本。

## 最新新增：阶段 3 车辆-only 主参照决策表 v0.2

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_only_decision_table_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_only_decision_table_v0_2_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_vehicle_only_decision_table_v0_2.py`
- 候选决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_candidate_decision_table_v0_2.csv`
- gate 状态表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_stage3_gate_status_v0_2.csv`
- 角色汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_decision_role_summary_v0_2.csv`
- 阶段 3 指标库存：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_stage3_metric_inventory_v0_2.csv`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/figures/vehicle_only_decision_key_metrics_test.png`
- RMSE vs 错侧图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/figures/vehicle_only_decision_rmse_vs_wrong_side_test.png`
- 角色计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/figures/vehicle_only_decision_role_counts.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/logs/vehicle_only_decision_table_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`e04bdb2f Add vehicle-only decision table`。
- 适合用户/老师直接查看：用户查看版总结、gate 状态表、候选决策表、指标概览图、RMSE vs 错侧图。

## 最新新增：阶段 3 top-K 可靠性选择/回退 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_reliability_selector_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_reliability_selector_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_topk_reliability_selector_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_per_sample_metrics.csv`
- 选择器特征表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_feature_table.csv`
- 决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_decisions.csv`
- validation 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_validation_selection.csv`
- 阈值扫描表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_threshold_sweep.csv`
- 分被试汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_subject_summary.csv`
- 分道路模块汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/tables/topk_reliability_selector_road_module_summary.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_bad_samples_test.png`
- oracle 增益样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_oracle_gain_samples_test.png`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_metric_summary_test.png`
- 决策计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_decision_counts_test.png`
- fallback 风险散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/figures/topk_reliability_selector_fallback_scatter_test.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_reliability_selector_v0_1/logs/topk_reliability_selector_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`fbb8d94d Add topk reliability selector`。
- 适合用户/老师直接查看：用户查看版总结、指标概览图、固定预测图、坏样本图、决策计数图、fallback 风险散点图。

## 最新新增：阶段 3 top-K top1/bestK 差距复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_gap_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_gap_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_topk_gap_review_v0_1.py`
- 样本详情：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_sample_detail.csv`
- 总体摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_overall_summary.csv`
- train 阈值：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_thresholds.csv`
- 相关性表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_correlations.csv`
- 分桶汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_bucket_summary.csv`
- top gap 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_top_gap_samples.csv`
- top1 比 RBF 更差样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/tables/topk_gap_review_top1_worse_than_rbf_samples.csv`
- top gap 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_top_samples.png`
- 风险散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_risk_scatter.png`
- 分支混淆图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_branch_confusion.png`
- 物理错误图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/figures/topk_gap_error_flags.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_gap_review_v0_1/logs/topk_gap_review_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`1ace03f2 Add topk gap review`。
- 适合用户/老师直接查看：用户查看版总结、总体摘要、top gap 样本、top gap 图、风险散点图、分支混淆图。

## 最新新增：阶段 3 top-K 车辆-only Transformer v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_vehicle_transformer_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_topk_vehicle_transformer_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_topk_vehicle_transformer_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_per_sample_metrics.csv`
- 分支诊断：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_branch_diagnostics.csv`
- 可靠性分箱：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_reliability_bins.csv`
- 训练历史：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_training_history.csv`
- 与参照对照表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_comparison_with_references.csv`
- 模型信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/tables/topk_vehicle_transformer_model_info.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_bad_samples_test.png`
- top1/bestK 差距图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_top1_bestk_gap_samples_test.png`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_metric_summary_test.png`
- 可靠性散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/figures/topk_reliability_scatter_test.png`
- checkpoint：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_topk_vehicle_transformer_top1_no_subject_best.pt`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/logs/topk_vehicle_transformer_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`03165475 Add topk vehicle transformer`。
- 适合用户/老师直接查看：用户查看版总结、指标表、固定预测图、top1/bestK 差距图、可靠性散点图。

## 最新新增：阶段 3 RBF/keypoint 多候选车辆-only 复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1.py`
- 统一指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/multihypothesis_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/multihypothesis_per_sample_metrics.csv`
- 选择摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/choice_summary.csv`
- 选择详情：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/choice_detail.csv`
- test 误选样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/test_misselected_samples.csv`
- test oracle 增益样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/tables/test_oracle_gap_samples.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_fixed_predictions_test.png`
- selector 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_selector_bad_samples_test.png`
- oracle 增益样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/multihypothesis_oracle_gap_samples_test.png`
- 选择混淆图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/selector_choice_confusion_test.png`
- oracle 增益柱图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/figures/oracle_gap_top_samples.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_multihypothesis_review_v0_1/logs/multihypothesis_review_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`01033e3e Add rbf keypoint multihypothesis review`。
- 适合用户/老师直接查看：用户查看版总结、统一指标表、选择摘要、固定预测图、oracle 增益样本图。

## 最新新增：阶段 3 RBF vs keypoint train/val 选择器 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_selector_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_rbf_keypoint_selector_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_rbf_keypoint_selector_v0_1.py`
- selector 训练表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_training_table.csv`
- selector 决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_decisions.csv`
- selector 统一指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_metrics.csv`
- 阈值扫描表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_threshold_sweep.csv`
- 选择后逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_selected_per_sample_metrics.csv`
- 数值特征表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_numeric_features.csv`
- 类别特征表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_categorical_features.csv`
- test 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_test_metrics.png`
- 阈值扫描图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_threshold_sweep.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/logs/rbf_keypoint_selector_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`7e3d53f6 Add rbf keypoint selector`。
- 适合用户/老师直接查看：用户查看版总结、selector 统一指标表、selector 决策表、test 指标图、阈值扫描图。

## 最新新增：阶段 3 keypoint+residual vs RBF 坏样本差异复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1.py`
- 样本级差异表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_sample_delta.csv`
- 错误变化计数表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_change_counts.csv`
- 总体摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_overall_summary.csv`
- 分被试摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_subject_summary.csv`
- Top 改善样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_top_improved.csv`
- Top 退化样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_top_degraded.csv`
- RMSE 差异图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/figures/keypoint_vs_rbf_rmse_delta_top_samples.png`
- 错误变化计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/figures/keypoint_vs_rbf_error_change_counts.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/logs/keypoint_vs_rbf_bad_sample_review_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、样本级差异表、RMSE 差异图、错误变化计数图。

## 最新新增：阶段 3 B 轨道车辆-only 关键点 + 残差 Transformer v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_per_sample_metrics.csv`
- 关键点误差表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_keypoint_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_model_info.csv`
- 训练历史：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_training_history.csv`
- val 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_val_selected_models.csv`
- B 轨道固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/B_response3s_strict_core_keypoint_residual_bad_samples_test.png`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/keypoint_residual_vehicle_transformer_metric_summary_test.png`
- checkpoint：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_keypoint_residual_vehicle_transformer_no_subject_best.pt`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/logs/keypoint_residual_vehicle_transformer_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、指标表、关键点误差表、B 轨道固定预测图、B 轨道坏样本图、指标概览图。

## 最新新增：阶段 3 B 轨道车辆-only 响应分解/结构化 Transformer v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_structured_vehicle_transformer_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_structured_vehicle_transformer_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_structured_vehicle_transformer_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_per_sample_metrics.csv`
- 辅助响应标签准确率表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_aux_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_model_info.csv`
- 训练历史：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_training_history.csv`
- val 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_val_selected_models.csv`
- B 轨道固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/B_response3s_strict_core_structured_bad_samples_test.png`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/structured_vehicle_transformer_metric_summary_test.png`
- checkpoint：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_structured_vehicle_transformer_aux_no_subject_best.pt`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/logs/structured_vehicle_transformer_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、指标表、B 轨道固定预测图、B 轨道坏样本图、指标概览图。

## 最新新增：阶段 3 干净响应任务车辆-only Transformer v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_vehicle_transformer_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_per_sample_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_model_info.csv`
- 训练历史：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_training_history.csv`
- val 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_val_selected_models.csv`
- 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/clean_task_vehicle_transformer_metric_summary_test.png`
- B 轨道固定图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/B_response3s_strict_core_transformer_bad_samples_test.png`
- checkpoint 目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/checkpoints`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/logs/clean_task_vehicle_transformer_summary.json`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、B 轨道固定图、B 轨道坏样本图、指标概览图。

## 最新新增：阶段 3 车辆-only 响应分解标签 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_response_decomposition_labels_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_response_decomposition_labels_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_response_decomposition_labels_v0_1.py`
- 样本级响应分解标签：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_sample_labels.csv`
- train-only 阈值表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_train_thresholds.csv`
- 轨道汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_track_summary.csv`
- split 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_split_summary.csv`
- 响应形态汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_morphology_summary.csv`
- 响应族汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_response_family_summary.csv`
- 道路模块汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_road_module_summary.csv`
- 被试汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/tables/response_decomposition_subject_summary.csv`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/figures/response_decomposition_morphology_counts.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/figures/response_decomposition_peak_time_amp_scatter.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/figures/b_track_mean_gt_trajectories_by_morphology.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_response_decomposition_labels_v0_1/logs/response_decomposition_labels_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、轨道汇总、响应形态计数图、B 轨道分形态平均轨迹图。

## 最新新增：阶段 3 响应任务定义决策 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_response_task_decision_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_response_task_decision_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_vehicle_instability_response_task_decision_v0_1.py`
- 事件级任务决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/event_response_task_decision_table.csv`
- 样本级任务 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`
- 任务类别计数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_decision_counts.csv`
- 任务轨道计数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_track_counts.csv`
- split 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_split_summary.csv`
- subject 汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/response_task_subject_summary.csv`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/figures/response_task_decision_counts.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/figures/response_task_sample_roles_by_window.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/logs/response_task_decision_summary.json`

## 最新新增：阶段 3 标签窗口覆盖审计 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_label_window_coverage_audit_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_label_window_coverage_audit_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_label_window_coverage_audit_v0_1.py`
- 样本级窗口指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_sample_metrics.csv`
- 事件级窗口策略表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_event_policy_table.csv`
- Top 坏事件叠加表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_bad_event_overlay.csv`
- 窗口级统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_window_summary.csv`
- 推荐策略计数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_policy_counts.csv`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_policy_counts.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_peak_tail_scatter_pre3.png`
- 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_coverage_rates_by_window.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/logs/label_window_coverage_audit_summary.json`

## 最新新增：阶段 3 复发坏样本失败来源归因 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_bad_event_failure_attribution_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_bad_event_failure_attribution_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_bad_event_failure_attribution_v0_1.py`
- 归因明细表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_failure_attribution_table.csv`
- 归因旗标统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_failure_flag_counts.csv`
- 主归因统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_primary_attribution_counts.csv`
- 归因旗标热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/figures/bad_event_failure_attribution_flags.png`
- 主归因计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/figures/bad_event_primary_attribution_counts.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/logs/bad_event_failure_attribution_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、归因明细表、归因旗标热图、单事件曲线总览拼图。

## 最新新增：阶段 3 复发坏样本详细曲线复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_bad_event_curve_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_bad_event_curve_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_bad_event_curve_review_v0_1.py`
- 总览拼图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/bad_event_curve_contact_sheet.png`
- 总览拼图 PDF：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/bad_event_curve_contact_sheet.pdf`
- 单事件曲线目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event`
- 图索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_figure_index.csv`
- 模型逐事件误差表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_model_error_table.csv`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/logs/bad_event_curve_review_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、总览拼图、单事件曲线目录、模型逐事件误差表。

## 最新新增：阶段 3 稳健性坏样本复盘 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_robustness_bad_sample_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_robustness_bad_sample_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_robustness_bad_sample_review_v0_1.py`
- 复发坏样本总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_bad_event_recurrence.csv`
- 代表坏样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_representative_bad_events.csv`
- 带错误标记的逐样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_test_per_sample_with_error_flags.csv`
- 物理错误汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_error_flag_summary_by_config_model.csv`
- 分被试坏样本汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_subject_bad_summary.csv`
- 坏样本矩阵表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_bad_event_matrix.csv`
- 复发事件图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_recurrent_bad_events.png`
- 物理错误热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_error_flag_heatmap.png`
- 分被试坏样本率图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_subject_bad_rate.png`
- 坏样本矩阵图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_bad_event_matrix.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/logs/robustness_bad_sample_review_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、代表坏样本表、复发坏样本图、物理错误热图、坏样本矩阵图。

## 最新新增：阶段 3 强车辆基线稳健性验证 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_robustness_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_robustness_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_strong_vehicle_robustness_v0_1.py`
- 决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_decision_table.csv`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_per_sample_metrics.csv`
- 模型信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_model_info.csv`
- RMSE 热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_rmse_heatmap.png`
- 大幅响应召回热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_large_recall_heatmap.png`
- 反向修正匹配热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_reversal_heatmap.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/logs/strong_vehicle_robustness_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、决策表、RMSE 热图、大幅响应召回热图、反向修正热图。

## 最新新增：阶段 3 车辆-only 统一对照 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_unified_vehicle_comparison_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_unified_vehicle_comparison_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_unified_vehicle_comparison_v0_1.py`
- test 指标总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_comparison_metrics_test.csv`
- all-split 指标总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_comparison_metrics_all_splits.csv`
- 相对 formal ridge 差异表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_comparison_delta_vs_formal_test.csv`
- 候选决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_candidate_decision_table.csv`
- 指标排名表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_metric_rankings_test.csv`
- 坏样本重合表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/tables/unified_vehicle_top_bad_overlap.csv`
- 关键指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/figures/unified_vehicle_key_metrics_test.png`
- 物理错误热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/figures/unified_vehicle_physical_failure_heatmap_test.png`
- RMSE/错侧权衡图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/figures/unified_vehicle_rmse_vs_wrong_side_test.png`
- 坏样本重合图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/figures/unified_vehicle_top_bad_overlap.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_unified_vehicle_comparison_v0_1/logs/unified_vehicle_comparison_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、test 指标总表、候选决策表、关键指标图、物理错误热图。

## 最新新增：阶段 3 车辆-only Transformer 时序基线 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_transformer_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_transformer_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_vehicle_transformer_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_per_sample_metrics.csv`
- 训练历史：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_training_history.csv`
- 模型信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_model_info.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/figures/vehicle_transformer_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/figures/vehicle_transformer_bad_samples_test.png`
- checkpoint：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/checkpoints/vehicle_transformer_context_no_subject_best.pt`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/logs/vehicle_transformer_summary.json`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、固定预测图、坏样本图、指标表。

## 最新新增：阶段 3 强车辆-only 时序/结构化基线 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_baselines_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_strong_vehicle_baselines_v0_1_cn.md`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/strong_vehicle_baseline_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/strong_vehicle_baseline_per_sample_metrics.csv`
- 模型信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/strong_vehicle_model_info.csv`
- val 选择模型错误分型：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/selected_model_error_flag_summary.csv`
- 与 formal ridge 逐样本差异：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/tables/selected_vs_formal_per_sample_delta.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_bad_samples_test.png`
- test 指标柱状图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_model_metric_bars_test.png`
- 与 formal ridge 的 RMSE 差异图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_selected_vs_formal_rmse_delta.png`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_strong_vehicle_baselines_v0_1.py`
- 服务器日志：无，本轮未使用服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、固定预测图、坏样本图、指标表。

## 最新新增：阶段 3 车辆基线坏样本物理错误分型 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_error_analysis_user_summary_cn.md`
- 正式中文报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_error_analysis_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_error_analysis_v0_1.py`
- 逐样本错误标签表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/per_sample_error_taxonomy.csv`
- 错误标签汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/error_flags_summary.csv`
- 分被试汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/subject_summary.csv`
- 分响应类型汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/morphology_summary.csv`
- 与旧 deep 对照：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/old_comparison_summary.csv`
- 错误标签柱状图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/formal_error_flag_counts.png`
- 与旧 deep RMSE 散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/formal_vs_old_deep_rmse_scatter.png`
- top bad 错误矩阵：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/top_bad_sample_error_matrix.png`
- 分被试错误热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/subject_error_rate_heatmap.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/logs/stage03_error_analysis_summary.json`
- 关键结果：反向修正计数不匹配 126/139，尾段漂移 87/139，严重幅值不足 81/139，多段修正结构不匹配 46/139，错侧 32/139，大幅响应漏召回 23/139；旧 deep 与 formal ridge top20%坏样本重叠 21/28。
- 重要边界：错误标签只用于解释 test 集失败类型，不参与训练、split、标准化或任何生理/风格有效性结论。
- 适合用户/老师直接查看：优先看用户查看版总结、错误标签柱状图、top bad 错误矩阵和逐样本错误标签表。

## 最新新增：阶段 3 正式车辆失稳样本车辆-only 基线 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_formal_baselines_user_summary_cn.md`
- 正式中文报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_formal_baselines_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_formal_baselines_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_per_sample_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_model_info.csv`
- 固定预测图样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_fixed_plot_samples.csv`
- 坏样本图样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_bad_plot_samples.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/figures/formal_baseline_bad_samples_test.png`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/logs/formal_baseline_summary.json`
- 关键结果：主窗口 `pre2_label2_old_main` + session-level test 中，`ridge_vehicle_context_no_subject` RMSE=0.649341、错侧率=0.230216、大幅响应召回=0.080000、严重幅值不足率=0.582734、反向修正计数完全匹配率=0.093525。
- 重要边界：该结果是车辆-only 浅层基线，不使用生理、脑电、连续风格或驾驶员 ID；不能支持风格/生理有效性结论。
- 适合用户/老师直接查看：优先看用户查看版总结、固定预测图、坏样本图和指标表。

## 最新新增：车辆失稳高置信正式样本清单 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_vehicle_instability_highconf_user_summary_cn.md`
- 数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_highconf_v0_1_cn.md`
- 正式中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/vehicle_instability_highconf_samples_v0_1_cn.md`
- 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_vehicle_instability_highconf_samples_v0_1.py`
- 样本主表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 样本 JSONL：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.jsonl`
- 事件锚点表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/event_anchor_table.csv`
- split 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/split_table.csv`
- split 可行性报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/split_feasibility_report.csv`
- 排除原因表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/sample_exclusion_reasons.csv`
- eval-only 响应类型统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/label_eval_only_response_summary.csv`
- 窗口配置表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/window_config_table.csv`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/logs/vehicle_instability_highconf_samples_summary_v0_1.json`
- 关键数量：908 个高置信事件，906 个进入正式 v0.1，2 个因 3 秒历史窗口不足排除；3 个窗口共 2718 行；主窗口 session-level split 为 train 611、val 156、test 139。
- 重要边界：`eval_label_*` 字段只允许用于评估分层、固定图和困难样本分析，不允许作为模型输入、split 决策或标准化依据。
- 适合用户/老师直接查看：优先看用户查看版总结、数据版本卡、split 可行性报告和排除原因表。

## 最新新增：旧 `vehicle_direct` 全量车辆-only clean 对照 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_oldcode_vehicle_direct_full_clean_user_summary_cn.md`
- 正式中文报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_direct_full_clean_on_instability_v0_1_cn.md`
- clean manifest 说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_deep_clean_vehicle_manifest_v0_1_cn.md`
- clean manifest 脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_oldcode_deep_clean_vehicle_manifest_v0_1.py`
- 评估和画图脚本入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/evaluate_oldcode_vehicle_direct_full_instability_v0_1.py`
- clean session-level manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_session_level_split_clean_vehicle_v0_1.csv`
- clean random split manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_random_event_split_clean_vehicle_v0_1.csv`
- clean subject-level manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_subject_level_split_clean_vehicle_v0_1.csv`
- clean 文件状态表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_deep_clean_vehicle_status_v0_1.csv`
- clean manifest 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/logs/oldcode_deep_clean_vehicle_manifest_summary_v0_1.json`
- 训练日志：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/logs/oldcode_vehicle_direct_full_clean_train_stdout.log`
- 评估摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/logs/oldcode_vehicle_direct_full_eval_summary.json`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_per_sample_metrics.csv`
- 分被试结果：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_by_subject_test.csv`
- 分响应类型结果：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_by_response_type_test.csv`
- 固定预测图样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_fixed_plot_samples.csv`
- 坏样本图样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_bad_plot_samples.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_bad_samples_test.png`
- 本地旧训练 run：`F:/data_set_process/data_process/tmp/event_conditioned_runs/OLD_FULL_INSTABILITY_HIGHCONF_VEHICLE_DIRECT_CLEAN_V0_1_20260512_181413`
- 重要结论：旧 `vehicle_direct` active checkpoint 在 session-level test 上 RMSE=0.637366，但严重幅值不足率=0.683453、大幅响应召回=0.142857、反向修正计数完全匹配率=0.086331；只能作为旧流程历史对照和坏样本来源，不能替代新流程强车辆基线。
- 重要风险记录：raw manifest 直读原始 CSV 的旧 deep run 已判定无效，因为旧 loader 会把原始交替缺失点填 0；正式结果只采用 clean manifest。
- 适合用户/老师直接查看：优先看用户查看版总结、固定预测图、坏样本图和正式中文报告。

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


## 场景触发点审计 v0.2（2026-05-12）

| 产物 | 路径 | 说明 |
|---|---|---|
| 审计脚本 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_scene_trigger_audit_v0_2.py` | 解析 `.aed` 交通对象、触发点并和旧锚点对齐 |
| 用户版说明 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_scene_trigger_user_summary_cn.md` | 面向用户/老师的白话说明 |
| 完整报告 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/scene_trigger_audit_v0_2_cn.md` | 包含 longstraight 交通对象、触发点和旧锚点对齐结论 |
| 交通对象表 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/aed_traffic_objects_v0_2.csv` | `.aed` 中交通车、车流源等对象 |
| 场景触发点表 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/aed_traffic_triggers_v0_2.csv` | Activate、Stop、ChangeLane 触发点 |
| 触发点时间映射 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/scene_trigger_session_times_v0_2.csv` | 触发点换算到每条被试记录相对时间轴 |
| 旧锚点对齐表 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/old_anchor_vs_scene_trigger_v0_2.csv` | 旧 v400 锚点与最近场景触发点时间差 |
| longstraight 图 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/figures/longstraight_scene_trigger_map_v0_2.png` | 25/26 车道交通对象与触发点图 |
| 时间差图 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/figures/old_anchor_scene_trigger_delta_hist_v0_2.png` | 旧锚点相对最近场景触发点的时间差分布 |


| longstraight 被试车道投影图 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/figures/longstraight_ego_lane_projection_v0_2.png` | 把被试车横向位置与交通触发车道放在同一横向坐标上 |
| longstraight 被试车道表 | `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/longstraight_ego_lane_at_scene_triggers_v0_2.csv` | 每个 longstraight 场景触发点处的被试车道估计 |
## 阶段 2 追加：场景设计与被试方向锚点工作图 v0.3

- 用户查看版报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/scene_design_working_map_v0_3_cn.md`
- 场景事件来源工作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/ego_direction_scene_event_source_map_v0_3.csv`
- 更新后的用户总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_scene_trigger_user_summary_cn.md`
- 依赖的完整触发审计报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/scene_trigger_audit_v0_2_cn.md`
- 依赖的触发审计表目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables`
- 说明：根据用户最新补充，`longstraight` 25/26 普通连续车流按背景处理，但 MAN TGL 25->26 变道和 Chrysler300 Stop 要进入候选锚点审查；`fix_road` 也已确认存在显式变道触发。
## 阶段 2 追加：被试方向设计点与候选锚点重建 v0.4

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_ego_direction_design_anchor_user_summary_cn.md`
- 完整中文报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/ego_direction_design_anchor_rebuild_v0_4_cn.md`
- 小论文场景依据摘录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/small_paper_scene_design_extract_v0_4.md`
- 审计脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_ego_direction_design_anchors_v0_4.py`
- 小论文场景表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/small_paper_scene_design_tables_v0_4.csv`
- 配置车道/附着表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/cfg_lane_mu_geometry_v0_4.csv`
- 被试方向低附着段表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/cfg_ego_direction_mu_segments_v0_4.csv`
- 候选锚点清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`
- 场景模块汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_module_summary_v0_4.csv`
- 说明：本轮确认小论文中弯道、低附着、急停、施工/维修、汇入等场景设计可作为锚点重建依据；根据后续用户补充，`middle_section`、`longstraight` 和 `fix_road` 也已纳入高优先级候选锚点审查。

### middle_section 连续超车修正

- 修正说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/middle_section_continuous_overtaking_correction_20260512_cn.md`
- 更新后的候选锚点清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`
- 更新后的模块汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_module_summary_v0_4.csv`
- 说明：用户确认道路连接段存在连续超车事件，因此 `middle_section` 已从“背景/过渡段”修正为“连续超车负荷事件段”。当前新增连续超车段入口、中点、横向偏移变化峰值、横向加速度峰值、横摆角速度峰值五类候选。

### longstraight 与维修路段变道触发修正

- 修正说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/longstraight_fixroad_lanechange_trigger_correction_20260512_cn.md`
- 更新后的候选锚点清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_candidates_v0_4.csv`
- 更新后的模块汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/ego_direction_design_anchor_v0_4/tables/ego_direction_design_anchor_module_summary_v0_4.csv`
- 更新后的场景事件来源工作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scene_trigger_audit_v0_2/tables/ego_direction_scene_event_source_map_v0_3.csv`
- 说明：用户确认 `longstraight` 和维修路段都涉及变道触发点。当前 `longstraight` 已新增显式变道/停车候选，`fix_road` 已新增两类显式变道候选。候选总数更新为 4519 行。

## 阶段 2 追加：事件候选筛选 v0.5

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_event_filter_user_summary_cn.md`
- 完整中文报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/event_candidate_filter_v0_5_cn.md`
- 筛选脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/filter_event_anchor_candidates_v0_5.py`
- 全部候选评分表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidate_scores_v0_5.csv`
- 去重后复核清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidates_for_review_v0_5.csv`
- 高置信复核清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidates_high_confidence_v0_5.csv`
- 分场景汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidate_module_summary_v0_5.csv`
- 分类型汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidate_decision_summary_v0_5.csv`
- 复核图索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/tables/event_candidate_review_panel_index_v0_5.csv`
- 概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/figures/event_candidate_filter_overview_v0_5.png`
- 代表性复核图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/event_candidate_filter_v0_5/figures/review_panels`
- 说明：本轮没有训练模型，只是把 4519 个候选锚点按设计证据、车身响应、窗口可用性和旧锚点接近程度进行初筛。去重后建议复核 534 个，高置信复核 314 个。

## GPTPro 事件锚点审查证据包（2026-05-12）

- 证据包目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512`
- 压缩包：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512.zip`
- README：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512/00_README_CN.md`
- GPTPro 提问词：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_review_pack_20260512/01_GPTPRO_PROMPT_CN.md`
- 说明：该包包含事件筛选中文报告、核心表格、概览图和按场景精选的 19 张复核图，不包含原始数据、模型 checkpoint、服务器密码或连接凭据。

## GPTPro 事件锚点审查回复与 v0.6 规则（2026-05-12）

- GPTPro 回复归档：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_response_manualpaste.md`
- 已填充决策记录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_decision_filled.md`
- 已填充行动项：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/gptpro_event_anchor_reply_20260512/20260512_event_anchor_v05_action_items_filled.md`
- v0.6 筛选规则草案：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/event_v0_6_screening_rule_from_gptpro_20260512_cn.md`
- 说明：GPTPro 支持先重审事件锚点，建议 v0.6 输出四类事件表，并先用小而干净的核心样本训练车辆/道路-only 基线。
## 阶段 3 追加：干净响应任务车辆-only 基线 v0.1（2026-05-13）

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_vehicle_baselines_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1.py`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/logs/clean_task_vehicle_baselines_summary.json`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_per_sample_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_model_info.csv`
- 任务轨道汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_track_summary.csv`
- val 选择模型表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_val_selected_models.csv`
- test 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/clean_task_vehicle_metric_summary_test.png`
- A 轨道固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/A_instant2s_core_fixed_predictions_test.png`
- A 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/A_instant2s_core_bad_samples_test.png`
- B 轨道固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/B_response3s_strict_core_bad_samples_test.png`
- 重要 Git commit：待提交。
- 适合用户/老师直接查看的材料：优先看用户查看版总结、test 指标概览图和 B 轨道坏样本图。
## 阶段 3 追加：B 轨道 RBF KRR 坏样本物理复查 v0.1（2026-05-13）

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_bad_sample_review_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1.py`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/logs/clean_task_bad_sample_review_summary.json`
- B 轨道坏样本总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_bad_sample_table.csv`
- 失败标记汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_failure_summary.csv`
- top bad 样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_top_bad_samples.csv`
- 分响应形态汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_failure_by_morphology.csv`
- 分被试汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_failure_by_subject.csv`
- 分道路模块汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/tables/b_track_rbf_failure_by_road_module.csv`
- 失败标记率图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/figures/b_track_rbf_failure_flag_rates.png`
- top bad RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/figures/b_track_rbf_top_bad_rmse.png`
- 主峰幅值散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_bad_sample_review_v0_1/figures/b_track_rbf_peak_amp_scatter.png`
- 重要 Git commit：待提交。
- 适合用户/老师直接查看的材料：优先看用户查看版总结、失败标记率图、top bad 样本表和主峰幅值散点图。

## 阶段 2 回补：episode-first 事件样本 v0.6（2026-05-13）

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage02_episode_first_v0_6_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/episode_first_event_v0_6_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/scripts/build_episode_first_events_v0_6.py`
- episode 总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_candidates_v0_6.csv`
- 第一版最干净核心候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/primary_training_events_v0_6.csv`
- 坐标需复核扩展候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/coordinate_flagged_expansion_events_v0_6.csv`
- 弱响应/负样本候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/trigger_no_effect_or_weak_response_v0_6.csv`
- 分桶汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_decision_summary_v0_6.csv`
- 类型汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_label_summary_v0_6.csv`
- 场景汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_module_summary_v0_6.csv`
- 代表图索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/tables/episode_review_panel_index_v0_6.csv`
- 概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/figures/episode_first_v0_6_summary.png`
- 分组代表图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/episode_first_event_v0_6/figures/episode_review_panels`
- 重要 Git commit：待提交。
- 适合用户/老师直接查看的材料：优先看用户查看版总结、概览图、严格核心表、坐标需复核扩展表和分组代表图目录。

## 阶段 3 追加：episode-first v0.6 纯车辆/道路预测对照 v0.1（2026-05-13）

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_episode_first_vehicle_baselines_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_episode_first_vehicle_baselines_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_episode_first_vehicle_baselines_v0_1.py`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/logs/episode_first_vehicle_baselines_summary.json`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_vehicle_metrics.csv`
- 逐样本指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_vehicle_per_sample_metrics.csv`
- 模型信息表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_vehicle_model_info.csv`
- 轨道汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_track_summary.csv`
- val 选择表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/tables/episode_first_vehicle_val_selected_models.csv`
- test 指标概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/figures/episode_first_vehicle_metric_summary_test.png`
- 3 秒不使用横向偏移坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_episode_first_vehicle_baselines_v0_1/figures/EP3_expanded_no_lateral_3s_bad_samples_test.png`
- 重要 Git commit：待提交。
- 适合用户/老师直接查看的材料：优先看用户查看版总结、val 选择表、指标概览图和 3 秒不使用横向偏移坏样本图。

## 目标完成审计：事件锚点筛选与样本重建（2026-05-13）

- 完成审计报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/goal_completion_audit_event_v0_6_20260513_cn.md`
- 审计结论：v0.6 样本清单、四类/多类分桶、复核图、分层统计、物理指标和纯车辆/道路预测对照均已完成；车辆-only 指标未优于旧 B 轨道，但证明 v0.6 更集中在复杂真实 episode，下一阶段应进入车辆-only 响应分解模型。

## 最新新增：阶段 4 连续驾驶风格探索性增量对照 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_increment_exploratory_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_increment_exploratory_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_style_increment_exploratory_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_metrics.csv`
- 逐样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_per_sample_metrics.csv`
- 置乱汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_permutation_summary.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_gate_table.csv`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_bad_samples_test.png`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、gate 表、指标概览图、固定预测图、坏样本图。

## 最新新增：阶段 4 连续风格跨 split 复核 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_cross_split_validation_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_cross_split_validation_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_style_cross_split_validation_v0_1.py`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_metrics.csv`
- 逐样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_per_sample_metrics.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_gate_table.csv`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/figures/style_cross_split_metric_summary_test.png`
- subject-level 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/figures/style_cross_split_subject_bad_samples_test.png`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：待提交。
- 适合用户/老师直接查看：用户查看版总结、gate 表、指标图、subject-level 坏样本图。

## 最新新增：阶段 4 连续风格路线收口决策 v0.1

- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_route_decision_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage04_style_route_decision_v0_1_cn.md`
- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/scripts/stage04_style_route_decision_v0_1.py`
- 证据摘要表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_evidence_summary.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_decision_gate_table.csv`
- 下一步动作表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/tables/style_route_next_actions.csv`
- RMSE delta 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_route_decision_v0_1/figures/style_route_rmse_delta_summary.png`
- 服务器日志：无，本轮未使用远程服务器。
- GPTPro 提问和回复：无。
- 重要 Git commit：`4064bf64 Add style route decision`。
- 适合用户/老师直接查看：用户查看版总结、gate 表、RMSE delta 图、下一步动作表。
# R2E-Steering 阶段产物索引
## 最新更新：2026-05-13 07:42

## Stage 7h val/test 选择不稳定诊断 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07h_val_test_selection_diagnostics_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07h_val_test_selection_diagnostics_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07h_val_test_selection_diagnostics_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/logs/stage07h_val_test_selection_diagnostics_summary.json`
- 候选 split 稳定性：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_candidate_split_stability.csv`
- 类别分布长表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_categorical_shift_long.csv`
- 类别分布摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_categorical_shift_summary.csv`
- 数值分布摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_numeric_shift_summary.csv`
- 逐样本候选收益：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_candidate_gain_samples.csv`
- 分 bucket 候选收益：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_candidate_gain_by_bucket.csv`
- keypoint target 指标副本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_keypoint_target_metrics_copy.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_gate_table.csv`
- 候选 val/test 稳定性图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_candidate_val_test_stability.png`
- val/test 类别偏移图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_val_test_categorical_shift.png`
- 候选逐样本收益图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_candidate_gain_by_split.png`
- keypoint target RMSE 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_keypoint_target_rmse_by_split.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`d990f8e3 Add stage7h selection diagnostics`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、候选稳定性表、类别/数值偏移表和候选稳定性图。

## 最新更新：2026-05-13 07:33

## Stage 7g keypoint/segment 车辆-only 候选 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07g_keypoint_segment_candidates_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07g_keypoint_segment_candidates_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07g_keypoint_segment_candidates_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/logs/stage07g_keypoint_segment_candidates_summary.json`
- feature audit：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_feature_audit.csv`
- allowed features：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_allowed_features.csv`
- 关键点预测表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_keypoint_prediction_table.csv`
- 关键点 target 指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_keypoint_target_metrics.csv`
- 候选指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_candidate_metrics.csv`
- 候选逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_candidate_per_sample_metrics.csv`
- validation selection：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_validation_selection_table.csv`
- oracle 诊断：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_oracle_diag.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_gate_table.csv`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_metric_summary_test.png`
- 关键点散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_keypoint_target_scatter.png`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_fixed_predictions_test.png`
- oracle gain 预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_oracle_gain_predictions_test.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`52de7176 Add stage7g keypoint segment candidates`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、候选指标、关键点散点图和 oracle gain 预测图。

## 最新更新：2026-05-13 07:19

## Stage 7f response-factorized vehicle-only candidate v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07f_response_factorized_candidates_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07f_response_factorized_candidates_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07f_response_factorized_candidates_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/logs/stage07f_response_factorized_candidates_summary.json`
- feature audit：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_feature_audit.csv`
- allowed features：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_allowed_features.csv`
- factor 预测指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_factor_prediction_metrics.csv`
- factor 预测明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_factor_predictions_long.csv`
- 候选逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_candidate_per_sample_metrics.csv`
- policy 指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_policy_metrics.csv`
- policy 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_policy_per_sample_metrics.csv`
- policy 和候选总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_policy_and_candidate_metrics.csv`
- validation selection：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_validation_selection_table.csv`
- response-factorized oracle 诊断：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_response_factorized_oracle_diag.csv`
- combo oracle 诊断：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_combo_oracle_diag.csv`
- prototype trace：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_prototype_trace.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_gate_table.csv`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/figures/stage07f_metric_summary_test.png`
- 固定预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/figures/stage07f_fixed_predictions_test.png`
- oracle gain 预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/figures/stage07f_oracle_gain_predictions_test.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`12cef06b Add stage7f response factorized candidates`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、factor 预测指标、固定预测图和 oracle gain 预测图。

## 最新更新：2026-05-13 06:50

## Stage 7c 候选轨迹导出与差异审计 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07c_candidate_trajectory_export_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07c_candidate_trajectory_export_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07c_candidate_trajectory_export_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/logs/stage07c_candidate_trajectory_export_summary.json`
- 轨迹数组：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/arrays/stage07c_candidate_trajectories.npz`
- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_export_metrics.csv`
- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_export_per_sample_metrics.csv`
- 候选两两差异明细：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_pairwise_disagreement_long.csv`
- 候选两两差异摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_pairwise_disagreement_summary.csv`
- 候选特征与标签诊断：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_feature_and_label_diagnosis.csv`
- oracle 摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_oracle_summary.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_export_gate_table.csv`
- 固定样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_fixed_predictions_test.png`
- 高候选差异图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_high_disagreement_predictions_test.png`
- oracle gain 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_oracle_gain_predictions_test.png`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_metric_summary_test.png`
- 差异-上限散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_disagreement_vs_oracle_gain_test.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`48b8c438 Add stage7c candidate trajectory export`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、指标图、oracle gain 图和轨迹数组说明。
# R2E-Steering 阶段产物索引
## 最新更新：2026-05-13 06:58

## Stage 7d 非 oracle selector v0.2

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07d_non_oracle_selector_v0_2.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07d_non_oracle_selector_v0_2_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07d_non_oracle_selector_v0_2_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/logs/stage07d_non_oracle_selector_summary.json`
- feature audit：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_feature_audit.csv`
- allowed features：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_allowed_features.csv`
- policy metrics：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_policy_metrics.csv`
- decision diagnostics：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_decision_diagnostics.csv`
- selected decisions：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_selected_policy_decisions.csv`
- validation selection：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_validation_selection_table.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_gate_table.csv`
- 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_policy_metrics_test.png`
- val delta 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_validation_rmse_delta.png`
- 选择计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_selected_choice_counts.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`eb785f4a Add stage7d non-oracle selector`。
- 适合用户/老师直接查看：优先看用户查看版总结、gate 表、val delta 图和 policy metrics 图。
# R2E-Steering 阶段产物索引
## 最新更新：2026-05-13 07:05

## Stage 7e 候选生成重设计审计 v0.1

- 代码入口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/scripts/stage07e_candidate_generation_redesign_v0_1.py`
- 用户查看版总结：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07e_candidate_generation_redesign_user_summary_cn.md`
- 技术报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage07e_candidate_generation_redesign_v0_1_cn.md`
- 运行摘要：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/logs/stage07e_candidate_generation_redesign_summary.json`
- 响应类型表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_response_label_table.csv`
- 样本候选缺口表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_sample_candidate_gap_table.csv`
- bucket 覆盖表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_existing_candidate_coverage_by_bucket.csv`
- oracle winner 分布：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_oracle_winner_distribution.csv`
- 候选生成蓝图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_candidate_generation_blueprint.csv`
- 下一实验计划：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_next_experiment_plan.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_gate_table.csv`
- oracle gain by family 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_oracle_gain_by_response_family_test.png`
- winner 分布图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_oracle_winner_distribution_test.png`
- 样本缺口散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_candidate_gap_scatter_test.png`
- 候选生成蓝图图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_candidate_generation_blueprint.png`
- GPTPro 提问和回复：本阶段未调用。
- 服务器日志：本阶段未使用服务器。
- 重要 Git commit：`98552bf3 Add stage7e candidate generation redesign`。
- 适合用户/老师直接查看：优先看用户查看版总结、候选生成蓝图、gate 表、oracle gain 图和 winner 分布图。


## v0.3 全量原始数据极限工况 episode 重筛

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_episode_v0_3_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\extreme_condition_episode_v0_3_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_episodes_all_v0_3.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`
- 分类统计图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\extreme_condition_category_counts_v0_3.png`


## v0.3 全量原始数据极限工况 episode 重筛

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_episode_v0_3_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\extreme_condition_episode_v0_3_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_episodes_all_v0_3.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`
- 分类统计图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\extreme_condition_category_counts_v0_3.png`


## v0.3 全量原始数据极限工况 episode 重筛

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_episode_v0_3_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\extreme_condition_episode_v0_3_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_episodes_all_v0_3.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`
- 分类统计图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\extreme_condition_category_counts_v0_3.png`


## v0.3 全量原始数据极限工况 episode 重筛

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_episode_v0_3_user_summary_cn.md`
- 技术报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\extreme_condition_episode_v0_3_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_episodes_all_v0_3.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\tables\extreme_condition_review_panel_index_v0_3.csv`
- 分类统计图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_3\figures\extreme_condition_category_counts_v0_3.png`


## v0.3 车辆-only 数据集与基线

- 数据集数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\arrays\v03_vehicle_only_pre2_label5_20hz.npz`
- 数据集 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\tables\v03_vehicle_only_manifest.csv`
- 指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_baseline_metrics.csv`
- 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_per_sample_metrics.csv`
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_vehicle_only_baselines_user_summary_cn.md`
- 固定预测图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_fixed_predictions_test.png`
- 坏样本图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_bad_samples_test.png`


## v0.3 车辆-only 数据集与基线（中文修正版）

- 数据集数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\arrays\v03_vehicle_only_pre2_label5_20hz.npz`
- 数据集 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\tables\v03_vehicle_only_manifest.csv`
- 总指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_baseline_metrics.csv`
- 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_per_sample_metrics.csv`
- 分样本类型表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_category_test.csv`
- 分被试表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_subject_test.csv`
- 分工况上下文表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_context_test.csv`
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_vehicle_only_baselines_user_summary_cn.md`
- 固定预测图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_fixed_predictions_test.png`
- 坏样本图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_bad_samples_test.png`


## v0.3 车辆-only 数据集与基线（中文修正版）

- 数据集数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\arrays\v03_vehicle_only_pre2_label5_20hz.npz`
- 数据集 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\extreme_condition_v0_3_vehicle_only\tables\v03_vehicle_only_manifest.csv`
- 总指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_baseline_metrics.csv`
- 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_per_sample_metrics.csv`
- 分样本类型表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_category_test.csv`
- 分被试表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_subject_test.csv`
- 分工况上下文表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_context_test.csv`
- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_vehicle_only_baselines_user_summary_cn.md`
- 固定预测图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_fixed_predictions_test.png`
- 坏样本图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_bad_samples_test.png`

## v0.3 样本纳入范围消融

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_vehicle_only_inclusion_ablation_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_inclusion_ablation\tables\v03_vehicle_only_inclusion_ablation_summary.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_inclusion_ablation`

## v0.3 excluded 分层加入实验

- 用户查看版报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_v03_excluded_stratified_inclusion_user_summary_cn.md`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion/tables/v03_excluded_stratified_inclusion_summary.csv`
- 输出目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_excluded_stratified_inclusion`
- 服务器日志本地副本：`F:/data_set_process/data_process/04_project_logs/reports/server_logs/v03_excluded_stratified_20260519/run.log`

## 横滚/姿态 excluded paired 诊断

- 用户查看版报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_v03_roll_excluded_pair_diagnosis_user_summary_cn.md`
- paired 明细表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis/tables/roll_vs_ref_common_test_paired_metrics.csv`
- 输出目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_roll_excluded_pair_diagnosis`

## v0.3 极限工况样本人工复核清单

- 复核指南：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/manual_review_guide_extreme_condition_v0_3_cn.md`
- 优先复核清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/manual_review_priority_list_v0_3.csv`
- 复核图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/review_panels`

## 新人工规则下的 v0.3 自动候选分组

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/new_rule_auto_candidate_groups_v0_3_cn.md`
- 全量分组表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/new_rule_auto_candidate_groups_v0_3.csv`
- 分组数量表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/new_rule_auto_candidate_group_summary_v0_3.csv`
- 每组代表样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/new_rule_auto_candidate_representatives_v0_3.csv`
- 分组图片目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/new_rule_review_groups_v0_3`
- 分组图片说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/new_rule_review_groups_v0_3/00_先看这里_图片说明.md`
- 分组图片索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/new_rule_review_groups_v0_3/new_rule_review_image_index_v0_3.csv`

## v0.3 方向盘角速度候选复核

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/new_rule_fast_steer_candidates_v0_3_cn.md`
- 候选表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/new_rule_fast_steer_candidates_v0_3.csv`
- 数量表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/new_rule_fast_steer_candidate_summary_v0_3.csv`
- 图片目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_review_v0_3`
- 图片说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_review_v0_3/00_先看这里_方向盘角速度候选说明.md`
- 图片索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_review_v0_3/fast_steer_review_image_index_v0_3.csv`

## v0.3 快速转向候选按车辆响应强弱拆分

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/fast_steer_vehicle_response_split_v0_3_cn.md`
- 拆分表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/fast_steer_vehicle_response_split_v0_3.csv`
- 数量表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/fast_steer_vehicle_response_split_summary_v0_3.csv`
- 图片目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_vehicle_response_split_v0_3`
- 图片说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_vehicle_response_split_v0_3/00_先看这里_快速转向按车辆响应拆分说明.md`
- 图片索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_vehicle_response_split_v0_3/fast_steer_vehicle_response_split_image_index_v0_3.csv`

## v0.3 快速转向候选锚点时序审计

- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/fast_steer_anchor_timing_audit_v0_3_cn.md`
- 审计表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/fast_steer_anchor_timing_audit_v0_3.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/tables/fast_steer_anchor_timing_audit_summary_v0_3.csv`
- 图片目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_anchor_timing_split_v0_3`
- 图片说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_anchor_timing_split_v0_3/00_先看这里_锚点时序复核说明.md`
- 图片索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/extreme_condition_episodes_v0_3/figures/fast_steer_anchor_timing_split_v0_3/fast_steer_anchor_timing_image_index_v0_3.csv`

## v0.3 临时加入锚点后响应弱样本训练

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v03_fast_weakpost_temp_train_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train\tables\v03_fast_weakpost_temp_train_summary.csv`
- 临时加入 episode 清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train\tables\v03_fast_weakpost_extra_episode_uids.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_fast_weakpost_temp_train`
## v0.3 样本筛选策略连续对比

- 脚本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/scripts/stage03_v03_screening_sweep.py`
- 服务器日志：`/root/autodl-tmp/data_process/05_rebuild_from_raw_20260511/00_project_notes/server_logs/stage03_v03_screening_sweep_20260519_203455.log`
- 预期输出目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v03_screening_sweep`
- 当前状态：服务器运行中，结果待拉回。

## 2026-05-19 v0.4 极限工况样本重新筛选

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_extreme_condition_refilter_v0_4_user_summary_cn.md`
- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\extreme_condition_episodes_refiltered_v0_4.csv`
- 主+次级训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\tables\train_candidate_episodes_v0_4.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\extreme_condition_episodes_v0_4\figures\review_panels`

## 2026-05-20 v0.4 重筛样本车辆-only GPU 基线

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v04_vehicle_only_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_vehicle_only_gpu_baseline\tables\v04_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_vehicle_only_gpu_baseline\tables\v04_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_vehicle_only_gpu_baseline`
## 2026-05-20 v0.4 主训练+次级+待复核 GPU 基线

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v04_review_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_review_gpu_baseline\tables\v04_review_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_review_gpu_baseline\tables\v04_review_gpu_ranking.csv`
- 逐模型指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_review_gpu_baseline\v04_primary_secondary_review_nolat\tables\v04_primary_secondary_review_nolat_gpu_metrics.csv`
- 逐样本指标：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v04_review_gpu_baseline\v04_primary_secondary_review_nolat\tables\v04_primary_secondary_review_nolat_gpu_per_sample_metrics.csv`
- 服务器日志：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\00_project_notes\server_logs\stage03_v04_review_gpu_20260520_102550.log`
- 脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v04_review_gpu_baseline.py`


## 2026-05-20 13:48:32 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`


## 2026-05-20 13:58:46 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`


## 2026-05-20 15:10:44 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`


## 2026-05-20 15:14:24 v0.5 生理机制验证

已更新 v0.5 生理机制验证材料。

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 机制表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`

## 2026-05-20 v0.5 连续风格与生理机制验证

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_mechanism_comparison_user_summary_cn.md`
- 实验脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v05_physio_mechanism_comparison.py`
- 旧流程生理对齐修正：`F:\data_set_process\data_process\02_code\final_code\model\training\run_event_conditioned_trajectory_baseline.py`
- 实验注册表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_experiment_registry.csv`
- 生理可用性表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_availability_check.csv`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 总指标对比表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 分被试表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_subject_metrics.csv`
- 机制判断表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_mechanism_table.csv`
- 服务器日志目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\logs`
- 启动命令模板：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\launch_commands_server_no_password.sh`

## 2026-05-20 v0.5 脑电原始数据审计与锚点前特征提取

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_eeg_feature_extraction_user_summary_cn.md`
- 提取脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v05_extract_eeg_features.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features`
- 记录级脑电清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_recording_inventory.csv`
- v0.5 锚点前 2 秒脑电特征表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_features_pre_anchor_hist2s.csv`
- 脑电可用性汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_feature_availability_summary.csv`
- 技术说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\stage03_v05_eeg_feature_extraction_report_cn.md`
## 2026-05-20 v0.5 生理/脑电补齐实验

- 用户查看版总结：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_physio_eeg_completion_user_summary_cn.md`
- 脑电特征提取说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v05_eeg_feature_extraction_user_summary_cn.md`
- 脑电锚点前特征表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_features_pre_anchor_hist2s.csv`
- 多版本结果白底表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_physio_eeg_result_table_white.png`
- 多版本指标柱状图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_physio_eeg_metric_overview.png`
- 脑电直接输入/全生理融合曲线图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_multiversion_overlay_eeg_direct.png`
- 教师蒸馏曲线图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\figures\v05_multiversion_overlay_teacher.png`
- 完整指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_comparison_table.csv`
- 运行状态表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_physio_mechanism_comparison\tables\v05_physio_run_status.csv`
- 汇图脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v05_build_multiversion_result_plots.py`
## 2026-05-20 完整记录级 episode 重建 v1.0

这一版不再把一条实验记录固定当成一个事件，也不继续以旧锚点或 `.aed` 触发点作为主入口。它从完整原始车辆 CSV 中重建车辆状态时间线，并允许一条完整实验记录自动切出多个 episode。

- 用户查看版说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_0_user_summary_cn.md`
- 构建脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scripts\build_record_level_episode_reconstruction_v1_0.py`
- 配置文件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\configs\record_episode_reconstruction_v1_0.json`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0`
- 全量 episode 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_level_episodes_all_v1_0.csv`
- 文件读取清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_level_file_inventory_v1_0.csv`
- 分组统计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_episode_group_summary_v1_0.csv`
- 上下文统计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_episode_context_summary_v1_0.csv`
- 分被试统计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\tables\record_episode_by_subject_v1_0.csv`
- 多信号复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\figures\review_panels`
- 静态 3D 轨迹图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_0\figures\trajectory_3d_static`

当前结果摘要：91 条原始车辆记录全部成功读取，自动检测到 1766 个 episode。其中核心极限样本 973 个，保守/弱操作极限样本 406 个，需要复核 335 个，边界复核 45 个。道路/场景字段只作为解释上下文，不作为最终事件真值。

## 2026-05-20 完整记录级 episode 人工复核整理 v1.1

用户查看 v1.0 复核图后判断：大部分样本可以保留，“需要复核”和“边界复核”类基本可以舍去。因此 v1.1 不重新检测 episode，只把 v1.0 结果整理成主训练候选、对照样本、舍弃/暂缓三类。

- 用户查看版说明：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_1_user_summary_cn.md`
- 构建脚本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scripts\build_record_episode_reviewed_v1_1.py`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed`
- 全量带复核决策表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\record_level_episodes_all_reviewed_v1_1.csv`
- 主训练候选表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\train_candidate_extreme_episodes_v1_1.csv`
- 舍弃/暂缓表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\discarded_review_episodes_v1_1.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\record_episode_figure_index_v1_1.csv`

当前结果摘要：v1.1 总 episode 仍为 1766 个；主训练候选 1383 个，对照样本 3 个，舍弃/暂缓 380 个。主训练候选由核心极限样本、保守/弱操作极限样本、次级训练样本组成；正常弯道或普通操控只作为对照，不进入主训练。

## 2026-05-20 v1.1 完整记录级样本车辆-only GPU 基线

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v11_vehicle_only_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v11_vehicle_only_gpu_baseline\tables\v11_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v11_vehicle_only_gpu_baseline\tables\v11_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v11_vehicle_only_gpu_baseline`

## 2026-05-21 完整记录级 episode 样本集 v1.2

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_2_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\record_level_episodes_all_v1_2.csv`
- 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\train_candidate_target_episodes_v1_2.csv`
- 疑似上下马路/路外恢复：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\suspected_offroad_or_road_recovery_episodes_v1_2.csv`
- 超长误合并：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\long_merged_episodes_v1_2.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\figures\review_panels_v1_2`

## 2026-05-21 完整记录级 episode 样本集 v1.3

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_3_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\record_level_episodes_all_v1_3.csv`
- 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\train_candidate_target_episodes_v1_3.csv`
- 疑似路边恢复或上下马路：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\suspected_roadedge_or_offroad_episodes_v1_3.csv`
- 长弯道/平滑坡度/弯道高动态复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\review_curve_or_grade_episodes_v1_3.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\figures\review_panels_v1_3`

## 2026-05-21 完整记录级 episode 样本集 v1.4

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_4_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\record_level_episodes_all_v1_4.csv`
- 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\train_candidate_target_episodes_v1_4.csv`
- 高度大幅下降保留样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\train_z_drop_extreme_keep_episodes_v1_4.csv`
- 上下马路但无明显大幅下降抛弃样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\tables\discard_roadedge_without_large_zdrop_episodes_v1_4.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_4_zdrop_reviewed\figures\review_panels_v1_4`

## 2026-05-21 完整记录级 episode 样本集 v1.5

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_5_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\record_level_episodes_all_v1_5.csv`
- 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\train_candidate_target_episodes_v1_5.csv`
- 弯道高度下降单独复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\review_curve_z_drop_separate_episodes_v1_5.csv`
- 全部弯道上下文表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\tables\all_curve_context_episodes_v1_5.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_5_curve_separated\figures\review_panels_v1_5`

## 2026-05-21 完整记录级 episode 样本集 v1.6

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_6_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\record_level_episodes_all_v1_6.csv`
- 非弯道主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\train_candidate_noncurve_episodes_v1_6.csv`
- 弯道侧倾候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\curve_roll_candidate_clean_episodes_v1_6.csv`
- 弯道高度异常排除：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\tables\discard_curve_slope_or_z_abnormal_episodes_v1_6.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_6_curve_roll_split\figures\review_panels_v1_6`

## 2026-05-22 完整记录级 episode 样本集 v1.7

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_7_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\record_level_episodes_all_v1_7.csv`
- 非弯道主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\train_candidate_noncurve_episodes_v1_7.csv`
- 平滑下坡弯道侧倾候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\curve_smooth_downhill_roll_candidate_episodes_v1_7.csv`
- 弯道高度轨迹异常排除：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\tables\discard_curve_z_profile_abnormal_episodes_v1_7.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_7_curve_zprofile_revised\figures\review_panels_v1_7`

## 2026-05-22 完整记录级 episode 样本集 v1.8

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_8_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\record_level_episodes_all_v1_8.csv`
- 全部训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\train_candidate_all_episodes_v1_8.csv`
- 非弯道主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\train_candidate_noncurve_episodes_v1_8.csv`
- 弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\train_candidate_curve_episodes_v1_8.csv`
- 平滑下坡弯道侧倾候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\curve_smooth_downhill_roll_candidate_episodes_v1_8.csv`
- 弯道高度异常排除：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\tables\discard_curve_height_or_z_abnormal_episodes_v1_8.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_8_anchor_curve_revised\figures\review_panels_v1_8`

## 2026-05-22 完整记录级 episode 样本集 v1.9 道路坐标判弯道

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v1_9_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\record_level_episodes_all_v1_9.csv`
- 全部训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\train_candidate_all_episodes_v1_9.csv`
- 道路坐标弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\train_candidate_curve_coord_episodes_v1_9.csv`
- 非弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\train_candidate_noncurve_episodes_v1_9.csv`
- 冲突审计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\metadata_vs_coord_curve_audit_v1_9.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\figures\review_panels_v1_9`

## 2026-05-22 v1.9 非弯道高度微小变化审计

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_v1_9_noncurve_height_micro_motion_audit_cn.md`
- 分组统计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\noncurve_height_micro_motion_by_module_v1_9.csv`
- 统计图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\figures\v1_9_noncurve_height_micro_motion_audit.png`


## 2026-05-22 完整记录级 episode 样本集 v2.0 全量无历史继承重审

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage02_record_episode_reconstruction_v2_0_user_summary_cn.md`
- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\record_level_episodes_all_v2_0.csv`
- 全部训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\train_candidate_all_episodes_v2_0.csv`
- 非弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\train_candidate_noncurve_episodes_v2_0.csv`
- 弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\train_candidate_curve_coord_episodes_v2_0.csv`
- 重新纳入训练样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\recovered_from_v1_9_nontrain_episodes_v2_0.csv`
- 待复核样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\manual_review_episodes_v2_0.csv`
- 对照样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\control_or_weak_episodes_v2_0.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\figures\review_panels_v2_0`

## 2026-05-22 v2.0 全量无历史继承重审样本车辆-only GPU 基线

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v20_no_history_vehicle_only_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_no_history_vehicle_only_gpu_baseline\tables\v20_no_history_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_no_history_vehicle_only_gpu_baseline\tables\v20_no_history_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_no_history_vehicle_only_gpu_baseline`

## 2026-05-22 v2.0 待复核样本纳入训练车辆-only GPU 基线

- 用户查看版报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_v20_review_inclusion_vehicle_only_gpu_user_summary_cn.md`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_review_inclusion_vehicle_only_gpu\tables\v20_review_inclusion_vehicle_only_gpu_summary.csv`
- 排名表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_review_inclusion_vehicle_only_gpu\tables\v20_review_inclusion_vehicle_only_gpu_ranking.csv`
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v20_review_inclusion_vehicle_only_gpu`

## 2026-05-25 goal1 v2.0 训练任务重定义执行

- 报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\stage03_goal1_v2_task_redesign_user_summary_cn.md`
- 最终报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\outputs\final_task_redesign_report.md`
- manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\manifests`
- E0-E5 输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\outputs`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\outputs\goal1_experiment_summary.csv`
