# 阶段 3 强车辆基线稳健性验证 v0.1

生成时间：2026-05-12

## 目的

上一轮统一对照显示 KNN/RBF/template 的 RMSE 很低，但可能存在模板记忆或分布依赖风险。本轮不进入风格/生理，只检查强车辆-only 候选在 subject-level split 和不同输入/标签窗口下是否稳定。

## 检查配置

- `random_main`：主 2 秒窗口 + random-event split。
- `subject_main`：主 2 秒窗口 + subject-level split。
- `session_pre1`：事件前 1 秒预测后 2 秒 + session-level split。
- `session_pre3`：事件前 3 秒预测后 3 秒 + session-level split。

## test 决策表

| robustness_config_id | val_selected_model | val_selected_test_rmse | best_test_model | best_test_rmse | formal_rmse | selected_rmse_improvement_pct_vs_formal | knn_train_rmse | knn_memory_risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_main | rbf_kernel_ridge_context_no_subject | 0.613049 | rbf_kernel_ridge_context_no_subject | 0.613049 | 0.699702 | 12.384355 | 0.000001 | True |
| session_pre1 | peak_scaled_template_context_no_subject | 0.539372 | rbf_kernel_ridge_context_no_subject | 0.520104 | 0.643752 | 16.214284 | 0.000003 | True |
| session_pre3 | direction_gated_knn_template_no_subject | 0.658432 | knn_template_context_no_subject | 0.590207 | 0.728463 | 9.613623 | 0.000002 | True |
| subject_main | rbf_kernel_ridge_context_no_subject | 0.609792 | knn_template_context_no_subject | 0.597936 | 0.672788 | 9.363457 | 0.000001 | True |

## 初步判断

- subject-level 主窗口中，val 选择模型为 `rbf_kernel_ridge_context_no_subject`，test RMSE=0.609792，formal RMSE=0.672788。
- KNN 在各配置中的 train RMSE 仍接近 0 时，继续标记为模板记忆风险，不能直接升级为主线。
- 本轮仍不支持任何生理、脑电或连续风格有效性结论。

## 产物

- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_metrics.csv`
- 决策表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_decision_table.csv`
- 模型信息：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_model_info.csv`
- RMSE 热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_rmse_heatmap.png`
- 大幅响应召回热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_large_recall_heatmap.png`
- 反向修正匹配热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_reversal_heatmap.png`
