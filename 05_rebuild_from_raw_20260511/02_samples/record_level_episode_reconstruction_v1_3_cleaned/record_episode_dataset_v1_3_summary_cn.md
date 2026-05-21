# 完整记录级 episode 样本集 v1.3：修正高度误判与路边恢复误判

生成时间：2026-05-21 11:14:03

## 这次为什么改

用户复核 v1.2 图片后指出两个错误：

1. 有些样本实际像上下马路/路边恢复，但 v1.2 因为高度变化不大，被误放进主训练目标极限事件；
2. 有些样本实际是弯道或道路趋势，但 v1.2 因为 `z_range` 很大，被误判为疑似上下马路。

所以 v1.3 的核心变化是：**不再把原始高度范围 `z_range` 当成上下马路的直接证据**。高度只作为辅助信号，并且要区分“平滑道路趋势”和“短时冲击/跳变残差”。

## 道路源文件检查

- 道路 cfg 目录：`F:\data_set_process\data_process\01_datasets\多模态数据\被试数据集合\道路信息\道路\Area2_extracted`
- 扫描 cfg 文件数：9
- cfg 中类似 `z/height/elevation/altitude` 的赋值数量：98
- cfg 中明确 `z0/z1/z` 数值数量：98，范围约 -7.0 到 0.0
- cfg 中类似坡度/横坡的赋值数量：0
- 中心线文件：`F:\data_set_process\data_process\01_datasets\多模态数据\被试数据集合\道路信息\道路\road_centerline_generated.csv`
- 中心线字段：`['s', 'kappa', 'x', 'y']`

当前检查说明：`curve1/curve2` 等道路 cfg 中确实存在 `z0/z1` 高程设置，这说明弯道或道路模块本身可能带有明显高度变化；但当前中心线表只有 `s/kappa/x/y`，还不能直接给每个车辆时刻扣除道路高程。因此 v1.3 不把车辆 `zx|z` 的绝对范围当成“上下马路”的唯一依据，而是把它拆成“平滑道路趋势”和“短时异常残差”两部分。

## v1.3 规则变化

- `z_range` 很大，但去掉线性趋势后的 `z_residual_range` 较小、变化方向比较单一时，优先标为“长弯道或平滑坡度，需要复核”，不直接标为上下马路。
- `z_range` 不大，但横向偏移跳变、车速大幅下降、制动明显、并且处在 middle/fix/long/低附着等上下文时，标为“疑似路边恢复或上下马路，暂缓”。
- 20 秒以上片段仍然不直接进入主训练；如果它像平滑弯道/坡度，就进入“长弯道或平滑坡度复核”，如果像路边恢复，就进入“疑似路边恢复或上下马路”。
- 用户点名的两个反例加入人工反馈覆盖规则，防止同类错误继续误导主训练样本。

## 数量变化

- 全量 episode：1766
- v1.2 主训练候选：1081
- v1.3 主训练候选：820
- v1.3 暂缓/复核：563
- v1.3 疑似路边恢复或上下马路：393
- v1.3 长弯道/平滑坡度/弯道高动态复核：128

v1.3 主训练候选时长中位数：5.379 秒；95% 分位：12.551 秒；最大：14.987 秒。

## v1.3 分类表

| v1_3_decision | v1_3_decision_cn | count |
| --- | --- | --- |
| train_target_extreme | 目标极限事件，未触发 v1.3 路边/超长风险，保留为训练候选 | 472 |
| discard_prior_review | v1.1 已经人工复核为舍弃/暂缓，本轮继续不进入训练 | 380 |
| train_conservative_extreme | 保守/弱操作极限样本，未触发 v1.3 路边/超长风险，保留为训练候选 | 348 |
| defer_roadedge_or_offroad | 车速/制动/横向偏移/高度残差组合提示可能是路边恢复或上下马路，暂不进入主训练 | 318 |
| review_long_curve_or_grade | 20 秒以上弯道/坡度趋势片段，不能仅因 z 范围大判为上下马路，先复核是否可拆成单事件 | 109 |
| defer_roadedge_or_offroad_long | 20 秒以上且存在路边/上下马路风险，暂不进入主训练 | 74 |
| review_duration_15_20s | 持续时间 15-20 秒，先进入复核而不直接训练 | 36 |
| review_curve_high_dynamics | 弯道内高动态样本，不能直接判为上下马路，但也需复核是否为干净极限工况 | 15 |
| defer_long_merged | 持续时间超过 20 秒，不适合作为单个干净事件直接训练 | 9 |
| control_normal_or_curve | 正常弯道或普通操控，仅保留为对照样本 | 3 |
| defer_roadedge_or_offroad_user_feedback | 用户复核指出：该样本更像上下马路/路边恢复，不应作为目标极限工况主训练样本 | 1 |
| review_long_curve_or_grade_user_feedback | 用户复核指出：该样本是弯道/道路趋势，不应仅因高度范围大判为上下马路 | 1 |

## 用户指出的两个反例在 v1.3 中的位置

| episode_uid | road_module_names | episode_duration_s | v1_2_decision | v1_3_decision | z_range_v1_2 | z_residual_range_v1_3 | lat_offset_range_raw_v1_2 | speed_drop_from_start_v1_3 | brake_peak_v1_3 | review_panel_v1_3_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rec_v1_byx_2025_09_28_17_05_51_0002 | middle_section | 7.3800 | train_target_extreme | defer_roadedge_or_offroad_user_feedback | 0.0166 | 0.0120 | 3.4947 | 25.1076 | 0.2585 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\figures\review_panels_v1_3\06_疑似路边恢复或上下马路_暂缓\0002_rec_v1_byx_2025_09_28_17_05_51_0002.png |
| rec_v1_gzj_2025_09_27_12_28_14_0004 | curve1 | 21.2400 | defer_offroad_recovery_long | review_long_curve_or_grade_user_feedback | 6.9977 | 1.6348 | 3.4984 | 109.9489 | 0.9700 | F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\figures\review_panels_v1_3\04_长弯道或平滑坡度_需要复核\0438_rec_v1_gzj_2025_09_27_12_28_14_0004.png |

## 输出位置

- v1.3 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\record_level_episodes_all_v1_3.csv`
- v1.3 主训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\train_candidate_target_episodes_v1_3.csv`
- 疑似路边恢复或上下马路：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\suspected_roadedge_or_offroad_episodes_v1_3.csv`
- 长弯道/平滑坡度/弯道高动态复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\review_curve_or_grade_episodes_v1_3.csv`
- 分类统计：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\tables\record_episode_v1_3_decision_summary.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_3_cleaned\figures\review_panels_v1_3`

## 当前结论

v1.3 比 v1.2 更符合你的反馈：它不再简单用高度范围判断上下马路，同时也能把“小高度变化但车速/制动/横向偏移明显异常”的路边恢复风险样本从主训练候选中分出来。

下一步建议先看两类图：

1. `06_疑似路边恢复或上下马路_暂缓`：确认这里是否主要是应排除/暂缓的样本；
2. `04_长弯道或平滑坡度_需要复核` 和 `05_弯道高动态_需要复核`：确认这些是否应该拆分后保留，还是直接作为弯道高动态样本。

本轮没有训练模型。
