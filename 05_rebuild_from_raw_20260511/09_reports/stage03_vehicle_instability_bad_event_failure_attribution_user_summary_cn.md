# 阶段 3 用户查看版：复发坏样本失败来源归因

## 为什么做

上一轮已经把 12 个反复失败事件画出来了。这一步把这些图和表整理成可执行判断：哪些样本可能是锚点或窗口问题，哪些更像车辆-only 模型真的预测不了复杂响应。

## 检查了什么

- 标签窗口是否可能太短。
- 事件锚点前是否已经出现方向盘响应或车辆动力学变化。
- 原始车辆核心信号是否有足够有效点。
- RBF/KNN/template 等车辆-only 候选是否共同幅值不足、错侧或漏反向修正。

## 目前发现

Top 12 中有 10 个事件优先归为“样本规则或原始信号需复核”，有 1 个事件优先归为“车辆-only 结构不足”。这意味着下一步不能直接进入生理或风格增量验证，要先把这些坏样本分清楚。

## 哪些结果可信

这一步没有训练新模型，没有使用生理、脑电、连续风格或驾驶员 ID，只读取已有车辆-only 误差表、样本清单和原始车辆片段。它适合决定下一步工程路线。

## 哪些还不能下结论

这些规则是自动初筛，不等于最终人工判定。特别是“锚点可能偏晚”和“窗口可能偏短”需要结合单事件曲线看。

## 下一阶段是否可以继续

可以继续阶段 3，但推荐先复核 `sample_rule_or_raw_signal_review` 的事件。若大部分确实是样本规则问题，应回到阶段 2 修 manifest；若不是，再进入响应分解、关键点残差或多假设车辆模型。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_failure_attribution_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/figures/bad_event_failure_attribution_flags.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/figures/bad_event_primary_attribution_counts.png`
4. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_bad_event_curve_review_v0_1\figures\bad_event_curve_contact_sheet.png`
