# 阶段 3 用户查看版：失稳样本响应任务定义决策 v0.1

## 为什么做

标签窗口覆盖审计显示，当前 2 秒标签经常没有覆盖完整方向盘响应，3 秒标签也有不少长事件仍未稳定。如果不先把任务定义拆清楚，继续训练模型会把“标签问题”和“模型能力问题”混在一起。

## 这个阶段做了什么

这一步没有改原始数据，也没有训练模型。它把 906 个高置信失稳事件分成几类：2 秒即时响应可用、应转 3 秒响应覆盖、2 秒尾段/锚点需复核、长事件或持续控制需回到阶段 2 复核。

## 当前决策数字

- 可作为 2 秒即时响应核心候选：84/906。
- 可作为 3 秒响应覆盖候选：294/906，其中严格核心候选 270/906。
- 需要长事件/持续控制复核：588/906。
- 需要人工窗口或锚点复核：636/906。
- 现有 2718 个窗口样本中，下一轮车辆-only 基线可优先使用的候选窗口样本为 462 个。

## 现在应该怎么理解

2 秒标签不适合再被说成“完整响应预测”。它可以作为“事件触发后的即时响应”任务。3 秒标签更接近完整响应，但仍有大量长事件需要拆分或标记为持续控制。后续如果要训练强车辆模型，应至少并行保留 2 秒即时响应和 3 秒响应覆盖两个任务定义。

## 哪些还不能下结论

这个决策表只是规则覆盖层，不等于人工最终真值；长事件不一定是坏样本，可能是驾驶员真实持续控制。它也不能说明连续风格、生理或 EEG 有效。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/event_response_task_decision_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/figures/response_task_decision_counts.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/figures/response_task_sample_roles_by_window.png`
