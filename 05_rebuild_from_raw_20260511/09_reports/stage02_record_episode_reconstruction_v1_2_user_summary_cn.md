# 完整记录级 episode 样本集 v1.2：超长片段与上下马路/路外恢复筛除

生成时间：2026-05-21 10:18:38

## 这次为什么做

用户指出：真实单个事件通常最长也就十几秒；当前 v1.1 中出现 60 秒、80 秒、105 秒 episode，明显不应理解为真实事件持续这么久。这类片段很可能来自连续实验中驾驶员开下马路、重新开回马路、车身高度变化、车身抖动和恢复驾驶过程，被自动检测误合并为一个长 episode。

因此 v1.2 不训练模型，只在 v1.1 基础上增加一层清洗和分流：

- 保留短时、语义较清楚的目标极限事件；
- 保留保守/弱操作极限事件；
- 将疑似上下马路/路外恢复片段单独分出去；
- 将超过合理事件长度的片段单独分出去，后续需要拆分或人工复核；
- 不直接删除这些风险片段，而是保存表格和复核图。

## 输入与规则

- 输入表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_1_reviewed\tables\record_level_episodes_all_reviewed_v1_1.csv`
- 原始车辆 CSV：来自每个 episode 的 `vehicle_file`
- 新增检测信号：`zx|z` 高度、`zx|pitch` 俯仰角、`zx|vpitch` 俯仰角速度、横向偏移、车速、制动、横滚等。
- 正常主训练时长上限：15.0 秒。
- 15 到 20 秒：先进入复核。
- 超过 20 秒：默认不是单个干净事件，进入暂缓/拆分类。

## 阈值

```json
{
  "normal_duration_max_s": 15.0,
  "review_duration_max_s": 20.0,
  "z_range_thr": 5.460072299999988,
  "z_rate_thr": 3.185949999712134,
  "pitch_range_thr": 0.30223914,
  "pitch_rate_thr": 1.4324949999999912,
  "lat_offset_range_thr": 17.921939999999974,
  "speed_range_thr": 111.12798999999983
}
```

## 数量变化

- v1.1 全量 episode：1766
- v1.1 主训练候选：1383
- v1.2 主训练候选：1081
- v1.2 暂缓/复核：302
- v1.2 疑似上下马路/路外恢复：149
- v1.2 超过 20 秒片段：139

## v1.2 分类表

| v1_2_decision | v1_2_decision_cn | count |
| --- | --- | --- |
| train_target_extreme | 核心/次级目标极限事件，未触发超长或路外恢复风险，保留为训练候选 | 695 |
| train_conservative_extreme | 保守/弱操作极限样本，未触发超长或路外恢复风险，保留为训练候选 | 386 |
| discard_prior_review | v1.1 已经人工复核为舍弃/暂缓，本轮继续不进入训练 | 380 |
| review_duration_15_20s | 持续时间 15-20 秒，接近或超过常规事件上限，先进入复核而不直接训练 | 106 |
| defer_offroad_recovery_long | 持续时间超过 20 秒且存在高度/俯仰/横向偏移异常，疑似上下马路或路外恢复误合并 | 83 |
| defer_offroad_recovery | 高度/俯仰/横向偏移特征提示可能是上下马路或路外恢复，先不进入主训练 | 66 |
| defer_long_merged | 持续时间超过 20 秒，不符合单个事件通常只有十几秒的实验逻辑，疑似多个过程误合并 | 47 |
| control_normal_or_curve | 正常弯道或普通操控，仅保留为对照样本 | 3 |

## 时长变化

- v1.1 主训练候选时长中位数：8.000 秒；95% 分位：24.155 秒；最大：105.035 秒。
- v1.2 主训练候选时长中位数：6.213 秒；95% 分位：13.070 秒；最大：14.987 秒。

## 输出位置

- v1.2 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\record_level_episodes_all_v1_2.csv`
- v1.2 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\train_candidate_target_episodes_v1_2.csv`
- 疑似上下马路/路外恢复：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\suspected_offroad_or_road_recovery_episodes_v1_2.csv`
- 超长误合并：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\long_merged_episodes_v1_2.csv`
- 15 到 20 秒复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\duration_15_20s_review_episodes_v1_2.csv`
- 分类统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\tables\record_episode_v1_2_decision_summary.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_2_cleaned\figures\review_panels_v1_2`

## 当前结论

v1.2 比 v1.1 更适合作为下一轮训练入口，因为它不再把 30 秒、60 秒、100 秒的连续恢复过程直接当成单个目标事件。下一步建议先人工查看 v1.2 的三类图：

1. 主训练目标事件；
2. 疑似上下马路/路外恢复；
3. 超长误合并/需要拆分。

如果这些分类大体符合直觉，再用 `train_candidate_target_episodes_v1_2.csv` 重跑车辆-only。否则继续调整 v1.2 规则。
