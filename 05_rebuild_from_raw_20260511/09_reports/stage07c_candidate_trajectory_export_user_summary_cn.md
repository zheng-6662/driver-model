# Stage 7c 用户查看版：候选轨迹导出与差异审计 v0.1

## 这个阶段为什么做

前面已经看到 top-K / 多候选有 oracle 上限，但非 oracle selector 最后完全退回 RBF/KNN。这里先不训练新模型，而是把已有候选轨迹完整导出来，看清楚问题到底是“候选本身不够不同”，还是“候选有潜力但选择机制不会选”。

## 这个阶段检查了什么

- 样本：`B_response3s_strict_core`，3 秒响应覆盖严格核心失稳样本。
- 主参照：`RBF/KNN`，也就是当前最强车辆-only 部署基线。
- 候选：RBF/KNN、keypoint residual、top-K 的 3 个 branch、top-K top1。
- 上限：best-of-3、RBF+topK oracle、RBF+keypoint+topK broad oracle，只作为诊断，不当作可部署结果。
- 运行方式：只加载已有 checkpoint 和已有样本数组，不训练，不使用生理、脑电、连续风格或驾驶员 ID。

## 目前发现了什么

- RBF/KNN test RMSE = 0.533667。
- keypoint residual test RMSE = 0.548993。
- top-K top1 test RMSE = 0.587865。
- RBF+topK oracle test RMSE = 0.415652，比 RBF/KNN 好 0.118014，但这是事后用真实标签选候选。
- broad oracle test RMSE = 0.410957，比 RBF/KNN 好 0.122710，同样只是上限诊断。

RBF+topK oracle 在 test 上选择候选的比例：

```text
                                      model  rate
        rbf_kernel_ridge_context_no_subject 0.375
topk_vehicle_transformer_branch0_no_subject 0.250
topk_vehicle_transformer_branch1_no_subject 0.200
topk_vehicle_transformer_branch2_no_subject 0.175
```

## 哪些结果可信

可信的是：所有候选轨迹已经能从现有数据和 checkpoint 复现，并保存为一个 npz；RBF/KNN 仍然是当前部署主参照；oracle 上限只能说明候选池里存在潜在更好轨迹，不能说明当前 selector 可用。

## 哪些结果还不能下结论

不能把 best-of-K 或 broad oracle 说成模型性能；不能因为 oracle 好就进入生理/EEG；也不能说 top-K 已经超过 RBF/KNN，因为当前可部署 top1 和 Stage 7b selector 都没有超过 RBF/KNN。

## 下一阶段是否可以继续

可以继续 Stage 7，但下一步应该针对“选择机制”或“候选生成方式”改，而不是直接进入生理。优先做两件事：第一，利用这次导出的候选差异特征设计更严格的非 oracle selector；第二，如果候选差异太小，就重新设计候选生成，让不同候选覆盖方向、幅值、峰值时间和尾段模式。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_metric_summary_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_disagreement_vs_oracle_gain_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_fixed_predictions_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_oracle_gain_predictions_test.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/arrays/stage07c_candidate_trajectories.npz`
6. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_feature_and_label_diagnosis.csv`
