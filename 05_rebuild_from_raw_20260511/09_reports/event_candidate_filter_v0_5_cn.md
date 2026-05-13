# 事件候选筛选 v0.5

生成时间：2026-05-12 21:35:53

## 这次做了什么

本轮没有训练模型，而是对 v0.4 生成的候选锚点做第一轮自动筛选。筛选目标不是直接产出最终训练样本，而是把 4519 个候选点分成：优先复核、人工复核、只作响应确认、暂不进入。

筛选原则是：显式触发点和道路/任务设计点优先，但必须看触发点附近是否真的有被试车辆响应；车身姿态峰值只能作为确认点，不能直接当因果锚点。

## 输入

- 候选锚点表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\ego_direction_design_anchor_v0_4\tables\ego_direction_design_anchor_candidates_v0_4.csv`
- 旧锚点对齐表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\tables\old_new_anchor_alignment_v0_1.csv`
- 高置信车辆失稳事件表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_instability_highconf_v0_1\tables\event_anchor_table.csv`

## 输出

- 全部候选评分表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\tables\event_candidate_scores_v0_5.csv`
- 去重后复核清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\tables\event_candidates_for_review_v0_5.csv`
- 高置信复核清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\tables\event_candidates_high_confidence_v0_5.csv`
- 分场景汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\tables\event_candidate_module_summary_v0_5.csv`
- 分类型汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\tables\event_candidate_decision_summary_v0_5.csv`
- 图像索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\tables\event_candidate_review_panel_index_v0_5.csv`
- 概览图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\figures\event_candidate_filter_overview_v0_5.png`
- 代表性复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\figures\review_panels`

## 数量概览

- 输入候选：4519 行
- 去重后建议复核：534 行
- 去重后高置信复核：314 行
- 已生成代表性复核图：56 张

分场景汇总：

| 场景 | 候选总数 | 去重后复核 | 高置信复核 | 只作响应确认 | 暂不进入 |
|---|---:|---:|---:|---:|---:|
| `curve1` | 420 | 80 | 46 | 252 | 84 |
| `curve2` | 350 | 70 | 57 | 210 | 70 |
| `differentmu_road` | 514 | 80 | 80 | 0 | 64 |
| `fix_road` | 495 | 80 | 31 | 213 | 71 |
| `longstraight` | 255 | 80 | 20 | 0 | 85 |
| `middle_section` | 2260 | 80 | 80 | 1356 | 450 |
| `stop` | 225 | 64 | 0 | 97 | 64 |

## 当前可以怎么理解

1. `longstraight` 和 `fix_road` 的显式变道/停车触发已经进入候选审查，但仍不能直接当最终训练锚点。
2. `middle_section` 的连接段入口属于连续任务段候选；如果触发后没有明显横向动态，只能算弱响应样本。
3. 横向加速度峰值、横摆角速度峰值、横向偏移峰值等更适合确认响应是否发生，不适合单独作为因果触发点。
4. 事件筛选下一步应看复核图，把候选锚点分成“可进入样本清单、偏早、偏晚、无明显响应、语义不清”。

## 下一步建议

1. 先人工看本轮代表性复核图，重点看 `longstraight`、`fix_road`、`middle_section`、`differentmu_road` 和 `curve1/curve2`。
2. 对通过视觉复核的事件，再生成 v0.6 样本清单。
3. 在 v0.6 样本清单固定前，不建议继续训练风格/生理模型。