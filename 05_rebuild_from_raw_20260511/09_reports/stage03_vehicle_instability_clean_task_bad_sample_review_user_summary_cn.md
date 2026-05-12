# 阶段 3 用户查看版：B 轨道车辆-only 坏样本复查 v0.1

## 这一步为什么做

上一轮结果里，3 秒响应覆盖任务的 RBF KRR 是当前最稳的车辆-only 候选，但预测图里仍然能看到不少反向修正、多段修正和大幅动作没有预测好。这里先把这些失败类型数清楚，避免只看 RMSE 就进入生理或风格阶段。

## 这一步检查了什么

- 只检查 B 轨道 test 集 40 个样本。
- 只检查车辆-only RBF KRR，没有训练新模型。
- 检查错侧、严重幅值不足、大幅响应漏召回、尾段漂移、零线穿越、反向修正、多段修正、峰值时间和启动延迟。

## 目前发现了什么

- 最差 20% 的阈值是 RMSE >= 0.657，共有 8 个坏样本。
- 这 8 个坏样本里，平均真实主峰幅值是 1.623，说明坏样本并不只是微小噪声样本。
- 主要剩余问题仍集中在结构化响应：反向修正、多段修正、幅值不足/错侧和尾段回正，而不是简单调一个 RMSE 损失就能完全解决。

## 哪些结论可信

可信的是：当前 B 轨道车辆-only RBF KRR 比旧的混合样本车辆-only 对照更适合作为下一步参考，但它仍有明确物理错误。

## 哪些结果还不能下结论

还不能说连续风格、生理或 EEG 有效，也不能说 KNN/template 是主线。A 轨道样本太少，B 轨道仍有结构错误，所以还需要车辆-only 结构化建模。

## 下一步建议

下一步优先做车辆-only 的响应分解：先预测方向、幅值、峰值时间、反向/多段修正类型，再预测轨迹。只有这个强车辆参考稳定后，才适合进入风格和生理增量验证。

## 推荐查看

1. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_clean_task_bad_sample_review_v0_1\figures\b_track_rbf_failure_flag_rates.png`
2. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_clean_task_bad_sample_review_v0_1\figures\b_track_rbf_top_bad_rmse.png`
3. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_clean_task_bad_sample_review_v0_1\tables\b_track_rbf_top_bad_samples.csv`
