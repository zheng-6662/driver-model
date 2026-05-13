# 阶段 2：事件候选筛选用户版说明 v0.5

生成时间：2026-05-12 21:35:53

## 这一步为什么做

现在我们已经知道每个场景大概有哪些设计点，但还不能把所有候选点直接拿去训练。因为有些点只是道路入口，有些是背景，有些是车辆已经响应后的峰值。为了避免样本继续错位，这一步先把事件候选筛一遍。

## 目前做到什么程度

我把 4519 个候选点都和原始车辆数据对齐，计算了触发点前后方向盘、横向加速度、横摆角速度、横向偏移、制动、车速、路面附着变化等指标。

自动筛完后，去重得到 534 个建议复核的事件，其中 314 个属于高置信复核候选。这里的“高置信”仍然只是进入复核，不等于最终训练样本。

## 你应该怎么看这个结果

优先看代表性复核图。每张图里红线是候选锚点，灰线如果出现则是最近旧锚点，橙线如果出现则是之前高置信车辆失稳点。我们要判断红线是不是比旧锚点更接近真实场景触发。

## 重点文件

1. 中文报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\09_reports\event_candidate_filter_v0_5_cn.md`
2. 复核清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\tables\event_candidates_for_review_v0_5.csv`
3. 高置信复核清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\tables\event_candidates_high_confidence_v0_5.csv`
4. 分场景汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\tables\event_candidate_module_summary_v0_5.csv`
5. 概览图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\figures\event_candidate_filter_overview_v0_5.png`
6. 代表性复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\figures\review_panels`

## 当前还不能下的结论

不能说这些事件已经是最终样本。下一步必须看图，把候选分成可保留、偏早、偏晚、无明显响应和语义不清。只有视觉和物理意义都合理的事件，才进入新的样本清单。