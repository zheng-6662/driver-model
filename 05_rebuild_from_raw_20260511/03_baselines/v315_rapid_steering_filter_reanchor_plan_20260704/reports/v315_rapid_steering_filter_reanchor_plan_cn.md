# 第315版方向盘快转过滤与重锚定候选方案

## 结论

- 本轮不训练模型，只把第314版来源审计转成下一轮训练前的数据处理策略。
- 全量事件：`1167`。
- 保留当前窗口训练：`1083`。
- 从当前窗口训练隔离：`84`。
- 其中候选重锚定：`77`。
- 第309版严重错误样本中需隔离：`4`。
- 用户截图样本中需隔离：`1`，即此前确认的 #020。

## 主要输出

- 全量处理策略表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v315_rapid_steering_filter_reanchor_plan_20260704\tables\v315_current_window_training_policy_all_delay0.csv`
- 当前任务保留清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v315_rapid_steering_filter_reanchor_plan_20260704\tables\v315_current_window_keep_manifest.csv`
- 当前任务隔离清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v315_rapid_steering_filter_reanchor_plan_20260704\tables\v315_current_window_isolate_manifest.csv`
- 重锚定候选表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v315_rapid_steering_filter_reanchor_plan_20260704\tables\v315_reanchor_candidate_manifest.csv`
- 按划分统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v315_rapid_steering_filter_reanchor_plan_20260704\tables\v315_split_filter_summary.csv`

## 按划分统计

| split | original_event_n | keep_current_window_n | isolate_current_window_n | reanchor_candidate_n | severe_n | screenshot_n | keep_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | 232 | 222 | 10 | 9 | 37 | 5 | 0.956897 |
| train | 702 | 650 | 52 | 47 | 0 | 0 | 0.925926 |
| val | 233 | 211 | 22 | 21 | 0 | 0 | 0.905579 |

## 处理策略分布

| v315_policy | v315_policy_cn | v315_next_action_cn | event_n | severe_n | screenshot_n |
| --- | --- | --- | --- | --- | --- |
| keep_current_window | 保留当前窗口样本 | 可进入当前窗口训练 | 1083 | 33 | 4 |
| isolate_late_fast_for_reanchor | 隔离：当前平缓但后续才快转，候选后移锚点 | 候选后移锚点后重新切窗 | 71 | 4 | 1 |
| exclude_weak_fast_source | 隔离：全程快转证据弱，候选剔除 | 从当前任务中剔除或单独归档 | 7 | 0 | 0 |
| isolate_pre_fast_for_reanchor | 隔离：锚点前已快转，候选前移锚点 | 候选前移锚点后重新切窗 | 6 | 0 | 0 |

## 重锚定候选统计

| split | v315_next_action_cn | event_n | mean_shift_s | median_shift_s | min_shift_s | max_shift_s |
| --- | --- | --- | --- | --- | --- | --- |
| test | 候选后移锚点后重新切窗 | 9 | 3.90389 | 4.36 | 2.06 | 5.5 |
| train | 候选前移锚点后重新切窗 | 5 | -0.841 | -0.775 | -1.21 | -0.61 |
| train | 候选后移锚点后重新切窗 | 42 | 3.52905 | 3.595 | 1.51 | 5.415 |
| val | 候选前移锚点后重新切窗 | 1 | -1.14 | -1.14 | -1.14 | -1.14 |
| val | 候选后移锚点后重新切窗 | 20 | 3.7945 | 3.7975 | 1.645 | 5.325 |

## 后续建议

- 下一轮若训练当前0到2秒任务，应先使用保留清单，隔离清单不参与当前窗口强动作监督。
- 重锚定候选需要重新切车辆窗口和目标曲线，不能只改表里的锚点时间后直接训练。
- 来源成立但仍预测差的严重样本，进入幅值、相位和极端动作跟随修正；来源不成立的样本不应再用于惩罚模型。
