# 阶段 3 用户查看版：标签窗口覆盖审计 v0.1

## 为什么做

上一轮 Top 12 复发坏样本里，很多事件被自动归因为“标签窗口或样本规则需要复核”。这一步不训练新模型，只检查当前 2 秒标签窗口是否经常没有覆盖完整方向盘响应。

## 检查了什么

- 正式高置信失稳样本 `vehicle_instability_highconf_v0_1` 的 906 个事件。
- 当前主窗口 `pre2_label2_old_main`：事件前 2 秒车辆历史，预测事件后 2 秒方向盘响应。
- 诊断窗口 `pre3_label3_response_coverage`：事件前 3 秒车辆历史，预测事件后 3 秒方向盘响应。
- 是否出现 2 秒之后还有更大峰值、2 秒之后方向盘仍有明显变化、3 秒末端仍未稳定、事件持续时间超过标签窗口等情况。

## 目前发现

- 247/906 个事件在 3 秒标签里显示主峰出现在 2 秒之后，说明旧 2 秒标签可能漏掉后续更大响应。
- 635/906 个事件在 2 秒之后仍有明显方向盘变化。
- 822/906 个事件被标记为“2 秒标签需要复核”。
- 612/906 个事件即使用 3 秒标签仍需要复核，通常代表连续失稳、长事件或尾段没有回正。
- Top 复发坏样本中有 12/12 个需要复核 2 秒窗口，9/12 个即使 3 秒窗口也仍需复核。

## 哪些结果可信

这一步只读取已生成的 `samples_master.csv` 和处理后的车辆标签数组，不使用生理、脑电、连续风格、驾驶员 ID，也没有训练新模型。它适合用来决定下一步是否应该修样本规则和标签窗口。

## 哪些结果还不能下结论

尾段没有回到 0 不一定都是错误。某些真实驾驶响应可能本来就需要保持方向盘角度，或者事件本身持续超过 3 秒。因此这些旗标只能说明“需要复核”，不能直接说明样本无效。

## 下一阶段是否可以继续

建议暂时不要继续堆新模型。下一步应先决定正式主标签到底采用 2 秒即时响应、3 秒响应覆盖，还是把长失稳事件拆成“启动响应”和“持续控制”两个任务。这个决定会影响后续所有车辆、风格和生理增量实验。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_event_policy_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/tables/label_window_bad_event_overlay.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_policy_counts.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_peak_tail_scatter_pre3.png`
