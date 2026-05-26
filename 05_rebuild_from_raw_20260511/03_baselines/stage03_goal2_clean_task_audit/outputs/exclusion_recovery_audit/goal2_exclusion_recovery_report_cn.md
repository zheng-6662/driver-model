# Goal2 被排除样本原因拆解与恢复优先级

## 目的

这份审计只解释 Goal2 中 1407 个被排除样本为什么被排除，并按当前理解给出恢复复核优先级。它不重新训练模型，也不把旧版本结论作为最终事实。

## 总量

- 全部 Goal2 样本：`1766`
- Goal2 被标为 slope/offroad/height 排除：`1407`

## 实际触发排除的原因

| reason                                                |   count |
|:------------------------------------------------------|--------:|
| Goal2 episode_type 已映射为 excluded_slope_or_offroad |    1407 |
| 旧版本文字包含路边/下马路/斜坡类关键词                |    1302 |
| 旧v1.3路边/路外标记                                   |     450 |
| 旧v1.3高度瞬态标记                                    |     331 |
| 当前v2.0标记明显高度异常                              |     270 |
| 当前v2.0文字包含高度异常类关键词                      |     266 |
| z_residual_range_v1_3 >= 0.50                         |     236 |
| 当前v2.0文字包含路边/下马路/斜坡类关键词              |     205 |
| 旧v1.2路外/回路标记                                   |     167 |
| 旧v1.2高度跳变标记                                    |     129 |
| z_rise_from_start_v1_4 >= 0.50                        |      47 |

## 建议恢复优先级

| recovery_priority         |   count |
|:--------------------------|--------:|
| A_优先人工恢复复核        |     792 |
| B_较可能可恢复            |     265 |
| C1_弯道高度变化重点复核   |       3 |
| C2_高度姿态重点复核       |     323 |
| D_暂不恢复_疑似路边或路外 |      16 |
| U_原因不清_需要复核       |       8 |

## 高度字段与恢复优先级交叉表

| v2_0_height_pose_issue   | recovery_priority         |   count |
|:-------------------------|:--------------------------|--------:|
| 明显高度异常             | C1_弯道高度变化重点复核   |       3 |
| 明显高度异常             | C2_高度姿态重点复核       |     267 |
| 轻度高度/姿态复核        | C2_高度姿态重点复核       |      56 |
| 轻度高度/姿态复核        | D_暂不恢复_疑似路边或路外 |      16 |
| 轻度高度/姿态复核        | U_原因不清_需要复核       |       8 |
| 高度小幅复核             | A_优先人工恢复复核        |      36 |
| 高度小幅复核             | B_较可能可恢复            |      24 |
| 高度微动正常             | A_优先人工恢复复核        |     739 |
| 高度微动正常             | B_较可能可恢复            |     202 |
| 高度轻微变化             | A_优先人工恢复复核        |      17 |
| 高度轻微变化             | B_较可能可恢复            |      39 |

## 解释

- `A_优先人工恢复复核`：当前高度接近正常，主要是旧版本文字触发排除，最可能是误伤。
- `B_较可能可恢复`：当前高度不属于明显异常，z 指标没有越过 0.50，建议看图后恢复。
- `C1_弯道高度变化重点复核`：道路坐标显示弯道且高度变化较大，需要区分正常坡道/弯道高程与上斜坡或路边。
- `C2_高度姿态重点复核`：高度或姿态指标明显，不建议直接恢复，必须结合道路源文件和图像。
- `D_暂不恢复_疑似路边或路外`：有路边/下马路/斜坡类证据，除非人工确认仍在道路内，否则暂不进入主训练。

## 重要提醒

旧版本文字和旧标记只能作为复核提示，不能继续作为硬排除规则。下一版样本规则应优先使用当前道路坐标、道路设计源文件、当前车辆轨迹和人工复核结论。

## 输出文件

- 逐样本拆解：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\goal2_exclusion_reason_breakdown.csv`
- 排除原因汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\goal2_exclusion_reason_summary.csv`
- 恢复优先级汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\goal2_recovery_priority_summary.csv`
- 每档抽查样本：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal2_clean_task_audit\outputs\exclusion_recovery_audit\goal2_manual_review_sample_30_each_priority.csv`