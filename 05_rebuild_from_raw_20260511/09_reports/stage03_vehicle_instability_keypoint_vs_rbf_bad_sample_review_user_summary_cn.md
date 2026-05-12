# 阶段 3 用户查看版：keypoint+residual 和 RBF 的坏样本差异

## 为什么做

keypoint+residual 的整体 RMSE 还没有超过 RBF，但错侧率和大幅响应召回更好。所以这一步不是再训练，而是逐个样本看：它到底救了哪些样本，又弄坏了哪些样本。

## 这次检查了什么

- 只看 B 轨道 test 的 40 个样本。
- 对比 keypoint+residual 和 RBF KRR 的逐样本 RMSE、错侧、大幅响应召回、幅值不足、峰值时间、启动延迟、尾段漂移和反向修正。
- 不使用生理、脑电、连续风格，也不连接服务器。

## 目前发现

- keypoint - RBF 的样本 RMSE 平均差：0.025325，说明整体上 keypoint 仍略差。
- RMSE 明显改善 11 个样本，明显退化 20 个样本。
- keypoint 修复错侧 5 个样本，新增错侧 1 个样本。
- keypoint 修复大幅响应召回 1 个样本，丢失大幅响应召回 0 个样本。

## 哪些结果可信

可信的是：keypoint+residual 的收益主要体现在方向和大幅响应召回；但它不是全局压倒 RBF，因为 RMSE 和困难样本仍没有赢。

## 哪些还不能下结论

还不能说结构模型已经解决车辆-only 问题，更不能说生理或风格有效。下一步如果继续模型，应看多假设或可靠性识别是否能保住 keypoint 的方向/大幅响应收益，同时减少退化样本。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/figures/keypoint_vs_rbf_rmse_delta_top_samples.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/figures/keypoint_vs_rbf_error_change_counts.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/tables/keypoint_vs_rbf_sample_delta.csv`
