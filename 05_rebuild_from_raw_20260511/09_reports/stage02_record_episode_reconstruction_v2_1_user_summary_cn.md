# v2.1 横向偏移参考系与道路高程修正后样本表

生成时间：2026-05-26

## 这版修正了什么

1. `SILAB` 中横向偏移在换道/跨道路参考线时可能出现跳变，因此横向偏移突变不再作为硬排除条件，只作为“参考系切换风险提示”。
2. 道路设计文件和道路中心线显示：`curve1/curve2` 本身存在米级高程变化，因此不能用原始 `z` 范围直接判断“上斜坡/下马路”。
3. 车辆侧倾、车身姿态和仿真噪声可能带来厘米级或十几厘米级高度变化。v2.1 中：
   - `z_residual < 0.20 m`：不作为排除依据；
   - `0.20-0.50 m`：进入复核；
   - `0.50-1.00 m`：重点复核；
   - `>=1.00 m`：强复核，只有结合当前道路/高度证据才暂不训练。
4. 旧版本文字里的“路边/下马路/上斜坡/高度异常”不再直接继承为硬排除，只作为历史提示。

## 总体数量

| 项目 | 数量 |
|---|---:|
| 全部 episode | 1766 |
| Goal2 严格排除样本 | 1407 |
| v2.1 可进入训练池/复核训练池 | 1753 |
| 从 Goal2 严格排除集中恢复 | 1394 |
| 其中 Goal2 排除但 v2.1 恢复到训练池 | 1394 |
| v2.1 硬排除 | 13 |
| 横向偏移参考系风险提示 | 1580 |
| 小幅高度变化且不作为排除依据 | 1459 |

## v2.1 角色分布

| v2_1_role                       |   count |
|:--------------------------------|--------:|
| main_train_candidate_v2_1       |     971 |
| review_recovered_candidate_v2_1 |     463 |
| control_or_weak_candidate_v2_1  |     319 |
| hard_excluded_v2_1              |      13 |

## 按数据划分统计

| split   | v2_1_role                       |   count |
|:--------|:--------------------------------|--------:|
| test    | main_train_candidate_v2_1       |     117 |
| test    | review_recovered_candidate_v2_1 |      91 |
| test    | control_or_weak_candidate_v2_1  |      67 |
| test    | hard_excluded_v2_1              |       1 |
| train   | main_train_candidate_v2_1       |     629 |
| train   | review_recovered_candidate_v2_1 |     300 |
| train   | control_or_weak_candidate_v2_1  |     203 |
| train   | hard_excluded_v2_1              |       9 |
| val     | main_train_candidate_v2_1       |     225 |
| val     | review_recovered_candidate_v2_1 |      72 |
| val     | control_or_weak_candidate_v2_1  |      49 |
| val     | hard_excluded_v2_1              |       3 |

## 高度规则与角色分布

| v2_1_height_level           | v2_1_role                       |   count |
|:----------------------------|:--------------------------------|--------:|
| 中等高度变化_需要复核       | review_recovered_candidate_v2_1 |      46 |
| 中等高度变化_需要复核       | main_train_candidate_v2_1       |      25 |
| 大幅高度变化_强复核         | review_recovered_candidate_v2_1 |      91 |
| 大幅高度变化_强复核         | hard_excluded_v2_1              |      13 |
| 小幅高度变化_不作为排除依据 | main_train_candidate_v2_1       |     874 |
| 小幅高度变化_不作为排除依据 | control_or_weak_candidate_v2_1  |     319 |
| 小幅高度变化_不作为排除依据 | review_recovered_candidate_v2_1 |     266 |
| 较大高度变化_重点复核       | main_train_candidate_v2_1       |      72 |
| 较大高度变化_重点复核       | review_recovered_candidate_v2_1 |      60 |

## Goal2 排除样本恢复情况

| v2_1_restored_from_goal2_exclusion   | v2_1_role                       |   count |
|:-------------------------------------|:--------------------------------|--------:|
| True                                 | main_train_candidate_v2_1       |     842 |
| True                                 | review_recovered_candidate_v2_1 |     365 |
| True                                 | control_or_weak_candidate_v2_1  |     187 |
| False                                | control_or_weak_candidate_v2_1  |     132 |
| False                                | main_train_candidate_v2_1       |     129 |
| False                                | review_recovered_candidate_v2_1 |      98 |
| False                                | hard_excluded_v2_1              |      13 |

## 弯道/非弯道上下文分布

| v2_1_curve_context   | v2_1_role                       |   count |
|:---------------------|:--------------------------------|--------:|
| True                 | review_recovered_candidate_v2_1 |     268 |
| True                 | main_train_candidate_v2_1       |     238 |
| False                | main_train_candidate_v2_1       |     733 |
| False                | control_or_weak_candidate_v2_1  |     319 |
| False                | review_recovered_candidate_v2_1 |     195 |
| False                | hard_excluded_v2_1              |      13 |

## 当前建议

- 这版只是样本规则修正和候选表，不训练模型。
- 下一步应先从 `manifest_training_pool_v2_1.csv` 中按角色抽样看图，尤其看：
  - `height_or_curve_review_v2_1`
  - `review_recovered_candidate_v2_1`
  - `manifest_lateral_reference_switch_review_v2_1.csv`
- 如果人工确认这些恢复样本多数合理，再基于 v2.1 生成新的 vehicle-only 数据集和共同评价集。
- 不建议再把“横向偏移突变”直接当作车辆真实横向突变；它应和方向盘、横摆、横滚、车速、制动、道路坐标一起解释。
