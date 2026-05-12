# 阶段 3 用户查看版：响应分解标签 v0.1

## 这一步为什么做

上一轮已经看出，B 轨道车辆-only RBF KRR 的主要问题不是单纯 RMSE，而是反向修正、多段修正、峰值时间、幅值和尾段状态预测不好。因此下一步模型不能只输出整条轨迹，应该先把响应拆成几个能解释的物理目标。

## 这一步检查了什么

- 从已有 2 秒和 3 秒干净轨道标签轨迹里，提取主峰方向、主峰幅值、峰值时间、启动时间、尾段状态、零线穿越、反向修正次数和响应形态。
- 这些标签只作为训练目标或评估分组，不能作为模型输入。

## 目前发现了什么

- A 轨道有 84 个即时响应样本，但 test 只有 12 个，只适合做诊断。
- B 轨道有 270 个 3 秒响应覆盖严格核心样本，train/val/test=188/42/40。
- B 轨道里 reverse/multi 响应比例很高，合计 0.989；这解释了为什么普通轨迹回归容易在反向修正和多段修正上失败。

## 哪些结果可信

可信的是：B 轨道下一步应该优先做结构化车辆-only 响应分解，而不是直接进入风格或生理增量验证。

## 哪些结果还不能下结论

这些标签来自未来方向盘轨迹，所以不能作为推理输入，也不能证明生理、脑电或连续风格有效。

## 下一步是否可以继续

可以继续，但只继续到车辆-only 响应分解模型。等这个强车辆参考稳定后，才适合验证风格和生理是否提供额外信息。

## 推荐查看

1. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_response_decomposition_labels_v0_1\figures\response_decomposition_morphology_counts.png`
2. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_response_decomposition_labels_v0_1\figures\response_decomposition_peak_time_amp_scatter.png`
3. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_response_decomposition_labels_v0_1\figures\b_track_mean_gt_trajectories_by_morphology.png`
4. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_response_decomposition_labels_v0_1\tables\response_decomposition_sample_labels.csv`
