# episode-first 事件样本 v0.6 用户版说明

生成时间：2026-05-13 01:38:16

## 这一步为什么做

GPTPro 指出，我们真正需要的不是“哪个设计触发点是真事件”，而是先判断驾驶过程中是否真实出现了车辆动态异常、方向盘响应和回正/纠正的完整片段。因此本轮先从车辆动态 episode 出发，再回头贴场景触发点。

## 当前结果

本轮输入 908 个车辆动态高置信 episode，自动分出 19 个第一版最干净核心候选、246 个坐标需复核扩展候选、298 个弱响应/负样本、306 个连续任务复核、30 个场景暂缓复核和 9 个因果顺序不清复核。

另外，有 246 个片段满足车辆动态、方向盘响应和窗口完整条件，但横向偏移坐标存在跳变风险。它们不能直接混入最干净训练集，但也不能简单判废，建议作为第二批人工复核或扩展候选。

## 你优先看什么

1. 完整报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\episode_first_event_v0_6_cn.md`
2. 第一版可训练核心表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\primary_training_events_v0_6.csv`
3. 坐标需复核扩展候选表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\coordinate_flagged_expansion_events_v0_6.csv`
4. 分桶汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\episode_decision_summary_v0_6.csv`
5. 人工复核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\manual_review_events_v0_6.csv`
6. 概览图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\figures\episode_first_v0_6_summary.png`
7. 代表图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\figures\episode_review_panels`

## 当前不能直接下的结论

这还不是最终训练集。它是自动规则版 episode-first 清单。下一步要看代表图，确认 P1/P2 分类、锚点位置和坐标连续性，再决定是否训练纯车辆/道路 baseline。