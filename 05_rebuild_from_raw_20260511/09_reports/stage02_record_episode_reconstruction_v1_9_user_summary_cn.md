# v1.9 道路坐标版 episode 样本重分总结

生成时间：2026-05-22 12:55

## 这版为什么要做

用户指出两个关键问题：

1. 弯道不能只凭高度判断，因为不是所有弯道都是下坡。
2. 高度只能用于判断异常，例如疑似上斜坡、下路边或高度跳变；弯道本身应该根据道路坐标、道路中心线、道路曲率或道路模块来判断。

所以 v1.9 废弃了“高度下降约等于弯道”的错误思路。当前规则改为：

- 用车辆原始 `zx|x / zx|y` 在 episode 多个关键时刻匹配 `full_centerline_layout.csv`；
- 根据匹配到的 `curve1 / curve2 / curve3` 判断是否处在弯道道路坐标上；
- 同时保留车辆文件中的 `zx1|lanecurvatureXY` 作为辅助核对；
- 高度 `z`、高度残差、横滚、横向加速度只用于判断弯道内是否异常或是否有明显车身姿态响应。

## 总体数量

- 全部 episode：1766
- 道路坐标判定为弯道上下文：506
- 当前训练候选总数：971
- 其中弯道训练候选：383
- 其中非弯道训练候选：588
- 原先误判为弯道、道路坐标显示非弯道但车辆动态明显、被转为非弯道候选：31
- 道路坐标映射距离过大、需要复核：79

## v1.9 决策分布

| v1_9_decision                                       | v1_9_decision_cn                                                                       |   count |   train_count |
|:----------------------------------------------------|:---------------------------------------------------------------------------------------|--------:|--------------:|
| train_noncurve_target_extreme                       | 道路坐标显示非弯道，继承非弯道极限/近极限主训练候选                                    |     557 |           557 |
| discard_noncurve_prior_review                       | 道路坐标显示非弯道，且不属于当前训练候选                                               |     557 |             0 |
| train_curve_coord_valid_normal_or_weak              | 道路坐标确认在弯道，车辆动态较弱或更像正常过弯，作为弯道普通/弱侧倾训练候选            |     201 |           201 |
| train_curve_coord_valid_roll_candidate              | 道路坐标确认在弯道，且车辆侧倾/横滚/横向动态明显，可作为弯道极限或近极限候选           |     182 |           182 |
| review_road_coordinate_mapping_uncertain            | 道路坐标最近邻距离过大，先不直接决定弯道/非弯道，需要复核道路映射                      |      79 |             0 |
| review_noncurve_recovered_from_height_rule_conflict | 原先按弯道高度异常排除，但道路坐标显示非弯道；车辆动态明显，需复核是否为非弯道极限事件 |      63 |             0 |
| discard_curve_coord_height_or_pose_abnormal         | 道路坐标确认在弯道，但高度/姿态形态异常，疑似上斜坡、下路边或非正常过弯                |      59 |             0 |
| train_noncurve_recovered_from_false_curve_dynamic   | 原先按弯道纳入，但道路坐标显示非弯道；车辆动态明显，改为非弯道动态候选                 |      31 |            31 |
| review_noncurve_false_curve_weak                    | 原先按弯道纳入，但道路坐标显示非弯道且动态较弱，改为复核样本                           |      22 |             0 |
| defer_noncurve_prior_review                         | 道路坐标显示非弯道，继承历史待复核样本                                                 |      15 |             0 |

## 道路坐标模块分布

| road_coord_dominant_module_v1_9   | v1_9_decision                                       |   count |
|:----------------------------------|:----------------------------------------------------|--------:|
| curve1                            | train_curve_coord_valid_normal_or_weak              |      84 |
| curve1                            | train_curve_coord_valid_roll_candidate              |      66 |
| curve1                            | review_road_coordinate_mapping_uncertain            |      64 |
| curve1                            | discard_curve_coord_height_or_pose_abnormal         |      39 |
| curve2                            | train_curve_coord_valid_normal_or_weak              |      88 |
| curve2                            | train_curve_coord_valid_roll_candidate              |      57 |
| curve2                            | discard_curve_coord_height_or_pose_abnormal         |      14 |
| differentmu_road                  | train_noncurve_target_extreme                       |     165 |
| differentmu_road                  | discard_noncurve_prior_review                       |     106 |
| differentmu_road                  | review_noncurve_recovered_from_height_rule_conflict |      61 |
| differentmu_road                  | train_noncurve_recovered_from_false_curve_dynamic   |      31 |
| differentmu_road                  | review_noncurve_false_curve_weak                    |      21 |
| differentmu_road                  | review_road_coordinate_mapping_uncertain            |      14 |
| differentmu_road                  | defer_noncurve_prior_review                         |       4 |
| fix_road                          | discard_noncurve_prior_review                       |      94 |
| fix_road                          | train_noncurve_target_extreme                       |      91 |
| fix_road                          | defer_noncurve_prior_review                         |       4 |
| fix_road                          | review_noncurve_recovered_from_height_rule_conflict |       1 |
| longstraight                      | discard_noncurve_prior_review                       |      25 |
| longstraight                      | train_noncurve_target_extreme                       |      12 |
| middle_section                    | discard_noncurve_prior_review                       |     322 |
| middle_section                    | train_noncurve_target_extreme                       |     268 |
| middle_section                    | train_curve_coord_valid_roll_candidate              |      59 |
| middle_section                    | train_curve_coord_valid_normal_or_weak              |      29 |
| middle_section                    | defer_noncurve_prior_review                         |       7 |
| middle_section                    | discard_curve_coord_height_or_pose_abnormal         |       6 |
| middle_section                    | review_noncurve_false_curve_weak                    |       1 |
| middle_section                    | review_noncurve_recovered_from_height_rule_conflict |       1 |
| middle_section                    | review_road_coordinate_mapping_uncertain            |       1 |
| stop                              | train_noncurve_target_extreme                       |      21 |
| stop                              | discard_noncurve_prior_review                       |      10 |

## v1.8 与道路坐标判定的冲突审计

下面这个表用于看“旧规则/旧上下文判为弯道”的样本，在道路坐标下是否仍然是弯道。

| v1_8_decision                              | road_coord_is_curve_v1_9   | v1_9_decision                                       |   count |
|:-------------------------------------------|:---------------------------|:----------------------------------------------------|--------:|
| train_noncurve_target_extreme              | False                      | train_noncurve_target_extreme                       |     557 |
| discard_noncurve_prior_review              | False                      | discard_noncurve_prior_review                       |     555 |
| train_noncurve_target_extreme              | True                       | train_curve_coord_valid_normal_or_weak              |      76 |
| discard_curve_height_or_z_abnormal         | False                      | review_noncurve_recovered_from_height_rule_conflict |      63 |
| discard_curve_height_or_z_abnormal         | True                       | discard_curve_coord_height_or_pose_abnormal         |      56 |
| discard_curve_height_or_z_abnormal         | True                       | train_curve_coord_valid_roll_candidate              |      50 |
| train_curve_smooth_downhill_roll_candidate | True                       | train_curve_coord_valid_roll_candidate              |      48 |
| discard_noncurve_prior_review              | True                       | train_curve_coord_valid_normal_or_weak              |      46 |
| train_curve_smooth_downhill_normal_or_weak | True                       | train_curve_coord_valid_roll_candidate              |      40 |
| train_noncurve_target_extreme              | True                       | review_road_coordinate_mapping_uncertain            |      32 |
| train_curve_unclear_or_weak                | True                       | train_curve_coord_valid_normal_or_weak              |      32 |
| discard_curve_height_or_z_abnormal         | True                       | train_curve_coord_valid_normal_or_weak              |      32 |
| train_curve_unclear_or_weak                | False                      | train_noncurve_recovered_from_false_curve_dynamic   |      31 |
| train_curve_unclear_or_weak                | False                      | review_noncurve_false_curve_weak                    |      22 |
| train_noncurve_target_extreme              | True                       | train_curve_coord_valid_roll_candidate              |      16 |
| defer_noncurve_prior_review                | False                      | defer_noncurve_prior_review                         |      15 |
| train_curve_smooth_downhill_normal_or_weak | True                       | train_curve_coord_valid_normal_or_weak              |      14 |
| discard_noncurve_prior_review              | True                       | review_road_coordinate_mapping_uncertain            |      12 |
| train_curve_unclear_or_weak                | True                       | train_curve_coord_valid_roll_candidate              |      11 |
| discard_curve_height_or_z_abnormal         | True                       | review_road_coordinate_mapping_uncertain            |      11 |
| discard_noncurve_prior_review              | False                      | review_road_coordinate_mapping_uncertain            |      11 |
| train_curve_unclear_profile_roll_candidate | True                       | train_curve_coord_valid_roll_candidate              |      11 |
| train_curve_unclear_or_weak                | True                       | review_road_coordinate_mapping_uncertain            |       7 |
| discard_noncurve_prior_review              | True                       | train_curve_coord_valid_roll_candidate              |       5 |
| train_noncurve_target_extreme              | False                      | review_road_coordinate_mapping_uncertain            |       4 |
| discard_curve_height_or_z_abnormal         | False                      | discard_noncurve_prior_review                       |       2 |
| defer_noncurve_prior_review                | True                       | review_road_coordinate_mapping_uncertain            |       2 |
| train_noncurve_target_extreme              | True                       | discard_curve_coord_height_or_pose_abnormal         |       2 |
| discard_noncurve_prior_review              | True                       | discard_curve_coord_height_or_pose_abnormal         |       1 |
| defer_noncurve_prior_review                | True                       | train_curve_coord_valid_normal_or_weak              |       1 |
| defer_noncurve_prior_review                | True                       | train_curve_coord_valid_roll_candidate              |       1 |

## 当前解释

1. 弯道判断已经改成道路坐标判断，不再由高度下降或高度起伏决定。
2. 平路弯道会被保留，因为是否弯道来自 `curve1/curve2/curve3` 道路模块，而不是 `z` 是否下降。
3. 下坡直道不会因为高度下降被判为弯道；如果道路坐标显示它不是弯道，它会进入非弯道候选、复核或排除。
4. 高度仍然有用，但用途变了：它只用于判断弯道内是否出现非正常高度变化，例如疑似上斜坡、下路边、坐标/路面异常。
5. 道路中心线最近邻距离有时较大，所以本版保留了 `road_coord_mapping_quality_v1_9`。距离很大的样本不直接作为强结论，进入复核。

## 输出文件

- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\record_level_episodes_all_v1_9.csv`
- 全部训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\train_candidate_all_episodes_v1_9.csv`
- 道路坐标弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\train_candidate_curve_coord_episodes_v1_9.csv`
- 非弯道训练候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\train_candidate_noncurve_episodes_v1_9.csv`
- 原误判弯道但道路坐标显示非弯道的动态候选：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\false_curve_recovered_noncurve_dynamic_episodes_v1_9.csv`
- 待复核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\manual_review_episodes_v1_9.csv`
- 舍弃表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\discarded_episodes_v1_9.csv`
- 冲突审计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\metadata_vs_coord_curve_audit_v1_9.csv`
- 道路模块统计表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\tables\road_coord_module_summary_v1_9.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v1_9_coord_curve_revised\figures\review_panels_v1_9`

## 下一步建议

先看 v1.9 的复核图，重点看三类：

1. 道路坐标确认弯道的训练候选；
2. 道路坐标确认弯道但高度/姿态异常的排除样本；
3. 原先被当弯道、现在道路坐标显示非弯道但车辆动态明显的样本。

确认这三类的语义后，再决定是否用 v1.9 训练车辆-only 模型。当前没有训练模型。
