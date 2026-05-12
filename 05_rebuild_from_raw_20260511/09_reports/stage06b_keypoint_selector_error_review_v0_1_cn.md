# Stage 6b：RBF/keypoint 选择器错误复盘 v0.1

生成时间：2026-05-13 05:57

## 输入和边界

- 输入：`rbf_keypoint_selector_training_table.csv`、`rbf_keypoint_selector_decisions.csv`、`rbf_keypoint_selector_selected_per_sample_metrics.csv`
- 只读已有车辆-only结果，不训练新模型。
- 不使用生理、脑电、连续风格、驾驶员 ID 或服务器。

## Test 混淆表

| selected_keypoint | oracle_keypoint_better | n_samples | mean_regret_vs_oracle | mean_delta_vs_rbf |
| ----------------- | ---------------------- | --------- | --------------------- | ----------------- |
| 0.000000          | 0.000000               | 17.000000 | 0.000000              | 0.000000          |
| 0.000000          | 1.000000               | 12.000000 | 0.130873              | 0.000000          |
| 1.000000          | 0.000000               | 6.000000  | 0.132400              | 0.132400          |
| 1.000000          | 1.000000               | 5.000000  | 0.000000              | -0.103321         |

## 高 regret 分组

| group_type              | group_value        | n_samples | selected_keypoint_rate | oracle_keypoint_better_rate | mean_regret_vs_oracle |
| ----------------------- | ------------------ | --------- | ---------------------- | --------------------------- | --------------------- |
| event_level             | weak               | 3.000000  | 0.333333               | 0.666667                    | 0.212028              |
| subject                 | zx                 | 4.000000  | 0.000000               | 0.750000                    | 0.195795              |
| subject                 | gf                 | 4.000000  | 0.500000               | 0.500000                    | 0.135158              |
| road_design_module_name | curve1             | 7.000000  | 0.142857               | 0.714286                    | 0.133155              |
| road_design_module_name | curve2             | 2.000000  | 0.000000               | 1.000000                    | 0.105193              |
| is_large_response       | 0                  | 32.000000 | 0.312500               | 0.500000                    | 0.073902              |
| is_difficult_peak_top20 | 0                  | 33.000000 | 0.303030               | 0.484848                    | 0.071663              |
| event_level             | instability_medium | 1.000000  | 0.000000               | 1.000000                    | 0.064030              |
| event_level             | strong_active      | 21.000000 | 0.380952               | 0.428571                    | 0.062492              |
| event_level             | medium_active      | 4.000000  | 0.250000               | 0.750000                    | 0.061843              |
| subject                 | gzj                | 9.000000  | 0.444444               | 0.333333                    | 0.059718              |
| subject                 | zxy                | 2.000000  | 0.500000               | 0.500000                    | 0.058142              |

## 下一步动作

| priority | action                       | why                                                                     |
| -------- | ---------------------------- | ----------------------------------------------------------------------- |
| 1.000000 | 先复盘FN_missed_keypoint_gain样本 | 当前 test 中漏选 keypoint 的样本数为 12，这是 selector 没吃到 oracle/keypoint 上限的主要来源。  |
| 2.000000 | 控制FP_select_keypoint_hurts样本 | 当前 test 中错选 keypoint 的样本数为 6，这类样本直接拉高 selector RMSE。                    |
| 3.000000 | 加入可靠性/不确定性特征                 | 当前 selector probability 与 keypoint 是否真的更好仍有重叠，需要引入候选间差异、历史稳定性或响应形态风险特征。 |
| 4.000000 | 把选择目标从纯RMSE改成物理错误多目标         | selector 虽然 RMSE 基本持平，但方向、大幅响应和困难样本有信号；下一版应显式惩罚错侧、严重幅值不足和尾段漂移。          |
| 5.000000 | 继续阻塞生理/EEG                   | selector 还未形成稳定可部署车辆-only提升，不能把剩余错误归因给新模态。                              |

## 图

- 混淆矩阵：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_confusion_matrix.png`
- top regret 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_top_regret_samples.png`
- probability vs actual gain：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06b_keypoint_selector_error_review_v0_1/figures/keypoint_selector_probability_vs_gain.png`
