# 阶段 3 用户查看版：干净响应任务车辆-only 基线 v0.1

## 为什么做

前一步已经把失稳样本拆成 2 秒即时响应和 3 秒响应覆盖两个相对干净的任务轨道。这里先只在这两个轨道上重跑车辆-only 对照，避免把长事件/持续控制样本混入核心训练后误判模型能力。

## 检查了什么

- A 轨道：2 秒即时响应核心候选，84 个事件，session-level test 12 个。
- B 轨道：3 秒响应覆盖严格核心候选，270 个事件，session-level test 40 个。
- 模型仍然只用车辆历史和事件/道路上下文，不使用生理、脑电、连续风格或驾驶员 ID。

## 当前结果

按验证集选择模型：

- A_instant2s_core: val 选择 `knn_template_context_no_subject`；test RMSE=0.428130，错侧率=0.333333，大幅响应召回=0.600000。
- B_response3s_strict_core: val 选择 `rbf_kernel_ridge_context_no_subject`；test RMSE=0.533667，错侧率=0.225000，大幅响应召回=0.750000。

按 test 事后排序的诊断结果：

- A_instant2s_core: 按 test RMSE 事后最小为 `zero_response_hold_current`，RMSE=0.336123，只用于诊断，不能替代 val 选择。
- B_response3s_strict_core: 按 test RMSE 事后最小为 `rbf_kernel_ridge_context_no_subject`，RMSE=0.533667，只用于诊断，不能替代 val 选择。

## 目前能说明什么

这个结果更适合作为后续车辆-only 主参照的候选，因为它不再把大量长事件和标签窗口未稳定样本混在一起。但 A 轨道 test 只有 12 个事件，且 KNN 在 train 上接近记忆，不能按 A 轨道单次 test 排名下强结论。是否进入风格/生理阶段，还要看这两个轨道上的固定图、坏样本图和物理指标是否足够稳定。

## 不能下的结论

这一步仍不能说明连续风格、生理或 EEG 有效，也不能说明长事件已经解决。D 轨道长事件仍要单独复核或拆分。

## 推荐查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/tables/clean_task_vehicle_metrics.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/clean_task_vehicle_metric_summary_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/A_instant2s_core_bad_samples_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_baselines_v0_1/figures/B_response3s_strict_core_bad_samples_test.png`
