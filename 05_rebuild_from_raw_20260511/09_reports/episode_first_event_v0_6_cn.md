# episode-first 事件样本 v0.6

生成时间：2026-05-13 01:38:16

## 这次做了什么

本轮按照 GPTPro 的最新建议，不再从设计触发点出发，而是从原始车辆动态出发：先判定是否存在车辆动态 episode，再补方向盘响应、纠正过程和附近场景触发点。

核心思路是：先找到真实发生的“车辆动态-方向盘-纠正 episode”，再回头解释它和哪个触发或场景有关。

## 输入

- 全原始车辆动态高置信事件：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_instability_all_raw_rescreen_v0_1\tables\all_raw_vehicle_instability_primary_high_confidence_v0_1.csv`
- 场景触发时间表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\scene_trigger_audit_v0_2\tables\scene_trigger_session_times_v0_2.csv`
- v0.5 候选触发评分表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\event_candidate_filter_v0_5\tables\event_candidate_scores_v0_5.csv`

## 输出

- episode 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\episode_candidates_v0_6.csv`
- 第一版可训练核心表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\primary_training_events_v0_6.csv`
- 坐标需复核扩展候选表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\coordinate_flagged_expansion_events_v0_6.csv`
- 人工复核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\manual_review_events_v0_6.csv`
- 响应确认/正常弯道表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\response_confirm_only_v0_6.csv`
- 暂缓/排除表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\holdout_or_excluded_v0_6.csv`
- 弱响应/触发无效候选表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\trigger_no_effect_or_weak_response_v0_6.csv`
- 概览图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\figures\episode_first_v0_6_summary.png`
- 代表图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\episode_review_panel_index_v0_6.csv`

## 数量概览

- 输入车辆动态 episode：908
- 第一版可训练核心：19
- 坐标需复核但动态和方向盘响应成立的扩展候选：246
- 弱响应/负样本候选：298
- 连续超车任务复核：306
- 场景暂缓复核：30
- 因果顺序不清复核：9

按 episode 类型：

| 类型 | 置信级别 | 推荐去向 | 数量 |
|---|---|---|---:|
| U_continuous_episode / 连续超车/连续任务，需要拆子事件 | B | manual_review | 305 |
| N_vehicle_dynamic_no_steering_response / 车辆动态异常但方向盘响应不足 | B | manual_review | 298 |
| P2_driver_initiated_avoidance / 主动避让/转向后产生高横向动态 | S | primary_training | 122 |
| P1_vehicle_disturbance_correction / 车辆扰动/失稳后方向盘纠偏 | S | primary_training | 99 |
| P1_vehicle_disturbance_correction / 车辆扰动/失稳后方向盘纠偏 | A | primary_training | 32 |
| U_unclear_or_holdout_scene / 场景被试相关性或语义仍需复核 | B | manual_review | 30 |
| P2_driver_initiated_avoidance / 主动避让/转向后产生高横向动态 | A | primary_training | 12 |
| U_unclear / 因果顺序不清，需要复核 | A | manual_review | 5 |
| U_unclear / 因果顺序不清，需要复核 | S | manual_review | 4 |
| U_continuous_episode / 连续超车/连续任务，需要拆子事件 | C | manual_review | 1 |

## 当前判断

1. 这一步已经从“触发点是不是事件”转为“是否真实发生车辆动态-方向盘-纠正 episode”。
2. `middle_section` 和 `longstraight/stop` 当前仍主要进入复核或暂缓，不直接进入第一版核心训练。
3. 第一版可训练核心主要来自 `differentmu_road`、`fix_road`、`curve1/curve2` 中满足方向盘响应和纠正条件的 episode。
4. 这个 v0.6 仍是自动规则版，后续需要看代表图，确认是否存在锚点偏晚、坐标跳变或正常弯道误判。

## 下一步建议

1. 先看 episode 代表图，确认 P1/P2 分类是否合理。
2. 若 primary_training_events_v0_6 数量和质量可接受，再用它构建纯车辆/道路 baseline。
3. 如果 primary 数量太少，则先补人工复核，不要急着训练。