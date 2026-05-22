# v2.0 全量无历史继承 episode 重审总结

生成时间：2026-05-22 17:14

## 这版为什么要做

用户指出：不能再沿用“历史上不是候选就不进入训练”的逻辑。此前 v1.9 虽然修正了弯道判定，但仍有一类 `discard_noncurve_prior_review`，本质上是历史非候选的继承结果。

因此 v2.0 做了一个更干净的重审：

- 1766 个 episode 全部重新审查；
- 历史 v1.8/v1.9 候选身份不参与新分类；
- 新分类只看当前可解释证据：道路坐标、车辆横滚/横摆/横向加速度、方向盘、速度/制动、高度/姿态异常和道路坐标映射质量；
- 历史标签只保留为审计对照字段。

## 总体数量

- 全部 episode：1766
- v2.0 训练候选：984
- 其中非弯道训练候选：746
- 其中弯道训练候选：238
- 待复核：463
- 正常/弱响应对照：319
- 从 v1.9 非训练集合中重新纳入训练：383
- 其中从 `discard_noncurve_prior_review` 中重新纳入训练：290

## v2.0 决策分布

| v2_0_decision                                | v2_0_decision_cn          |   count |   train_count |   review_count |   control_count |   recovered_from_v19_nontrain |
|:---------------------------------------------|:--------------------------|--------:|--------------:|---------------:|----------------:|------------------------------:|
| train_noncurve_secondary_dynamic             | 非弯道次级动态训练候选    |     409 |           409 |              0 |               0 |                           136 |
| train_noncurve_vehicle_dynamic               | 非弯道车辆动态训练候选    |     337 |           337 |              0 |               0 |                           247 |
| train_curve_normal_or_weak                   | 弯道普通/弱侧倾训练候选   |     208 |           208 |              0 |               0 |                             0 |
| train_curve_roll_dynamic                     | 弯道侧倾/动态训练候选     |      30 |            30 |              0 |               0 |                             0 |
| control_noncurve_weak_or_normal              | 非弯道弱响应/正常对照     |     319 |             0 |              0 |             319 |                             0 |
| review_curve_height_pose_abnormal            | 弯道高度/姿态异常复核     |     204 |             0 |            204 |               0 |                             0 |
| review_speed_brake_only                      | 速度/制动为主但横向动态弱 |     107 |             0 |            107 |               0 |                             0 |
| review_mapping_uncertain                     | 道路坐标映射不确定        |      79 |             0 |             79 |               0 |                             0 |
| review_fast_steer_weak_vehicle               | 快打方向但车辆响应弱      |      72 |             0 |             72 |               0 |                             0 |
| review_noncurve_height_abnormal_weak_dynamic | 非弯道高度异常但动态弱    |       1 |             0 |              1 |               0 |                             0 |

## v1.9 到 v2.0 的变化审计

下面这张表用于检查：哪些历史非候选在 v2.0 中被重新纳入或转入复核。

| v1_9_decision                                       | v2_0_decision                                |   count |   train_count |   recovered_from_v19_nontrain |
|:----------------------------------------------------|:---------------------------------------------|--------:|--------------:|------------------------------:|
| discard_noncurve_prior_review                       | train_noncurve_vehicle_dynamic               |     179 |           179 |                           179 |
| discard_noncurve_prior_review                       | train_noncurve_secondary_dynamic             |     111 |           111 |                           111 |
| review_noncurve_recovered_from_height_rule_conflict | train_noncurve_vehicle_dynamic               |      63 |            63 |                            63 |
| review_noncurve_false_curve_weak                    | train_noncurve_secondary_dynamic             |      15 |            15 |                            15 |
| defer_noncurve_prior_review                         | train_noncurve_secondary_dynamic             |      10 |            10 |                            10 |
| defer_noncurve_prior_review                         | train_noncurve_vehicle_dynamic               |       5 |             5 |                             5 |
| train_noncurve_target_extreme                       | train_noncurve_secondary_dynamic             |     260 |           260 |                             0 |
| train_curve_coord_valid_normal_or_weak              | train_curve_normal_or_weak                   |     182 |           182 |                             0 |
| train_noncurve_target_extreme                       | control_noncurve_weak_or_normal              |     175 |             0 |                             0 |
| discard_noncurve_prior_review                       | control_noncurve_weak_or_normal              |     137 |             0 |                             0 |
| train_curve_coord_valid_roll_candidate              | review_curve_height_pose_abnormal            |     126 |             0 |                             0 |
| discard_noncurve_prior_review                       | review_speed_brake_only                      |      95 |             0 |                             0 |
| review_road_coordinate_mapping_uncertain            | review_mapping_uncertain                     |      79 |             0 |                             0 |
| train_noncurve_target_extreme                       | train_noncurve_vehicle_dynamic               |      73 |            73 |                             0 |
| discard_curve_coord_height_or_pose_abnormal         | review_curve_height_pose_abnormal            |      59 |             0 |                             0 |
| train_noncurve_target_extreme                       | review_fast_steer_weak_vehicle               |      38 |             0 |                             0 |
| discard_noncurve_prior_review                       | review_fast_steer_weak_vehicle               |      34 |             0 |                             0 |
| train_curve_coord_valid_roll_candidate              | train_curve_roll_dynamic                     |      30 |            30 |                             0 |
| train_curve_coord_valid_roll_candidate              | train_curve_normal_or_weak                   |      26 |            26 |                             0 |
| train_curve_coord_valid_normal_or_weak              | review_curve_height_pose_abnormal            |      19 |             0 |                             0 |
| train_noncurve_recovered_from_false_curve_dynamic   | train_noncurve_vehicle_dynamic               |      17 |            17 |                             0 |
| train_noncurve_recovered_from_false_curve_dynamic   | train_noncurve_secondary_dynamic             |      13 |            13 |                             0 |
| train_noncurve_target_extreme                       | review_speed_brake_only                      |      11 |             0 |                             0 |
| review_noncurve_false_curve_weak                    | control_noncurve_weak_or_normal              |       6 |             0 |                             0 |
| discard_noncurve_prior_review                       | review_noncurve_height_abnormal_weak_dynamic |       1 |             0 |                             0 |
| review_noncurve_false_curve_weak                    | review_speed_brake_only                      |       1 |             0 |                             0 |
| train_noncurve_recovered_from_false_curve_dynamic   | control_noncurve_weak_or_normal              |       1 |             0 |                             0 |

## 道路模块分布

| road_coord_dominant_module_v1_9   | v2_0_decision                                |   count |
|:----------------------------------|:---------------------------------------------|--------:|
| curve1                            | review_curve_height_pose_abnormal            |     106 |
| curve1                            | train_curve_normal_or_weak                   |      79 |
| curve1                            | review_mapping_uncertain                     |      64 |
| curve1                            | train_curve_roll_dynamic                     |       4 |
| curve2                            | train_curve_normal_or_weak                   |      89 |
| curve2                            | review_curve_height_pose_abnormal            |      61 |
| curve2                            | train_curve_roll_dynamic                     |       9 |
| differentmu_road                  | control_noncurve_weak_or_normal              |     124 |
| differentmu_road                  | train_noncurve_vehicle_dynamic               |     114 |
| differentmu_road                  | train_noncurve_secondary_dynamic             |      95 |
| differentmu_road                  | review_speed_brake_only                      |      42 |
| differentmu_road                  | review_mapping_uncertain                     |      14 |
| differentmu_road                  | review_fast_steer_weak_vehicle               |      13 |
| fix_road                          | train_noncurve_secondary_dynamic             |      77 |
| fix_road                          | train_noncurve_vehicle_dynamic               |      50 |
| fix_road                          | control_noncurve_weak_or_normal              |      28 |
| fix_road                          | review_speed_brake_only                      |      18 |
| fix_road                          | review_fast_steer_weak_vehicle               |      17 |
| longstraight                      | train_noncurve_secondary_dynamic             |      15 |
| longstraight                      | control_noncurve_weak_or_normal              |       9 |
| longstraight                      | review_speed_brake_only                      |       7 |
| longstraight                      | train_noncurve_vehicle_dynamic               |       5 |
| longstraight                      | review_fast_steer_weak_vehicle               |       1 |
| middle_section                    | train_noncurve_secondary_dynamic             |     208 |
| middle_section                    | train_noncurve_vehicle_dynamic               |     162 |
| middle_section                    | control_noncurve_weak_or_normal              |     148 |
| middle_section                    | review_fast_steer_weak_vehicle               |      40 |
| middle_section                    | review_speed_brake_only                      |      40 |
| middle_section                    | train_curve_normal_or_weak                   |      40 |
| middle_section                    | review_curve_height_pose_abnormal            |      37 |
| middle_section                    | train_curve_roll_dynamic                     |      17 |
| middle_section                    | review_mapping_uncertain                     |       1 |
| middle_section                    | review_noncurve_height_abnormal_weak_dynamic |       1 |
| stop                              | train_noncurve_secondary_dynamic             |      14 |
| stop                              | control_noncurve_weak_or_normal              |      10 |
| stop                              | train_noncurve_vehicle_dynamic               |       6 |
| stop                              | review_fast_steer_weak_vehicle               |       1 |

## 当前解释

1. v2.0 不再使用“历史候选/历史非候选”作为分类依据。
2. 原先历史非候选并没有被直接舍弃，而是重新按车辆动态、道路坐标和姿态指标判断。
3. 快打方向但车辆动态弱的样本不直接纳入极限训练，先进入复核。
4. 车辆动态明显但驾驶员操作弱的样本可以进入训练，因为这符合“保守驾驶员/弱操作也可能处于极限工况”的研究目标。
5. 高度 z 仍然只作为异常辅助证据；直路/非弯道的小幅高度微动不作为排除依据。

## 输出文件

- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\record_level_episodes_all_v2_0.csv`
- 全部训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\train_candidate_all_episodes_v2_0.csv`
- 非弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\train_candidate_noncurve_episodes_v2_0.csv`
- 弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\train_candidate_curve_coord_episodes_v2_0.csv`
- 从 v1.9 非训练集合中重新纳入的样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\recovered_from_v1_9_nontrain_episodes_v2_0.csv`
- 待复核样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\manual_review_episodes_v2_0.csv`
- 正常/弱响应对照：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\tables\control_or_weak_episodes_v2_0.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_0_no_history_reaudit\figures\review_panels_v2_0`

## 下一步建议

先看 `00_全量重审新增训练候选_重点看` 这个复核图文件夹。如果这部分大多数确实合理，说明 v2.0 纠正了历史非候选带来的偏置；如果这里仍混入很多无效片段，就需要继续细化车辆动态阈值，而不是回到历史候选继承逻辑。
