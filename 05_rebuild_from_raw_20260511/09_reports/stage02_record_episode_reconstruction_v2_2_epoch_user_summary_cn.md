# v2.2 epoch 边界精修审计

生成时间：2026-05-26 16:15:38

## 这版解决什么

v2.1 解决的是“样本是否被异常规则误删”；v2.2 解决的是“每个 epoch 的开始、结束和模型锚点是否切得合适”。

本轮把三个概念分开：

1. **完整事件段**：从驾驶员或车辆持续活动开始，到主要响应后稳定下来。
2. **模型锚点**：后续模型真正对齐的 `t0`，优先用驾驶员动作起点，其次用车辆动态起点。
3. **建模窗口**：`t0` 前 2 秒作为历史，`t0` 后 0.5 秒可作为早期观察，之后 5 秒作为预测标签。

## 总体结果

| 项目 | 数量 |
|---|---:|
| 全部 episode | 1766 |
| v2.2 边界基本一致 | 398 |
| v2.2 需要重划边界 | 1360 |
| 活动弱或边界不清楚 | 8 |
| v2.2 可进入边界训练池 | 1721 |
| 复核图数量 | 114 |

## 状态统计

| v2_2_epoch_status    |   count |
|:---------------------|--------:|
| boundary_ok          |     398 |
| boundary_reworked    |    1360 |
| low_activity_unclear |       8 |

## 边界问题统计

| flag                 |   count |
|:---------------------|--------:|
| old_start_too_early  |     846 |
| old_anchor_too_early |     614 |
| old_end_too_late     |     459 |
| old_end_too_early    |     449 |
| boundary_ok          |     398 |
| old_start_too_late   |     154 |
| old_anchor_too_late  |     136 |

## 边界偏移幅度统计

正数表示 v2.2 比旧版本更晚，负数表示 v2.2 比旧版本更早。

| metric          |   mean |   median |    p10 |    p90 |
|:----------------|-------:|---------:|-------:|-------:|
| 新开始 - 旧开始 |  1.177 |    0.735 | -0.08  |  4.222 |
| 新结束 - 旧结束 | -1.225 |    0.106 | -7.492 |  4.551 |
| 新锚点 - 旧锚点 |  0.764 |    0.237 | -0.33  |  3.426 |
| 新episode时长   |  6.528 |    4.88  |  1.8   | 15     |

## v2.1 角色与 v2.2 边界状态

| v2_1_role                       | v2_2_epoch_status    |   count |
|:--------------------------------|:---------------------|--------:|
| control_or_weak_candidate_v2_1  | boundary_ok          |      74 |
| control_or_weak_candidate_v2_1  | boundary_reworked    |     243 |
| control_or_weak_candidate_v2_1  | low_activity_unclear |       2 |
| hard_excluded_v2_1              | boundary_ok          |       1 |
| hard_excluded_v2_1              | boundary_reworked    |      12 |
| main_train_candidate_v2_1       | boundary_ok          |     234 |
| main_train_candidate_v2_1       | boundary_reworked    |     733 |
| main_train_candidate_v2_1       | low_activity_unclear |       4 |
| review_recovered_candidate_v2_1 | boundary_ok          |      89 |
| review_recovered_candidate_v2_1 | boundary_reworked    |     372 |
| review_recovered_candidate_v2_1 | low_activity_unclear |       2 |

## 输出文件

- 全量表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\record_level_episodes_all_v2_2_epoch_refined.csv`
- v2.2 训练池：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\training_pool_epoch_refined_v2_2.csv`
- 需要重划边界表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\epoch_boundary_rework_needed_v2_2.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\tables\epoch_boundary_review_figure_index_v2_2.csv`
- 复核图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\record_level_episode_reconstruction_v2_2_epoch_refined\figures\epoch_boundary_review_v2_2`

## 当前建议

- 后续训练不要再直接使用旧 `episode_start_s` 或旧 `episode_end_s`。
- 训练输入应优先使用 `v2_2_model_anchor_s`、`v2_2_obs_start_s`、`v2_2_obs_end_s`、`v2_2_label_start_s`、`v2_2_label_end_s`。
- 人工复核优先看 `00_旧开始偏早`、`02_旧结束偏早`、`03_旧结束偏晚` 和 `04_旧锚点偏晚` 四类图。
