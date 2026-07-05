# v300 within-subject 完整重训报告

## 这一步做了什么

v300 不是 v299 那种固定旧 v249 预测后的 residual 校准，而是把 v299 的同被试事件级划分映射回全部 rolling 样本后，从原始车辆/道路/phase 输入重新训练 joint curve decoder。

训练了两类输入候选：

- `no_subject`：只使用现有车辆历史、道路和 phase 输入。
- `subject_onehot`：在现有输入上加入被试身份 one-hot，用来检验驾驶员身份/风格信息是否能补上锚点前车辆信息不足。

旧 v249 预测只作为结果诊断参照，没有参与 scaler fit、训练、validation 选择或 test 调参。

## 划分与防泄漏

| split   |   rolling_rows |   unique_events_from_rolling |   unique_events_from_v299_table |   unique_subjects |   event_in_multiple_splits_n |   event_without_6_delay_rows_n |   duplicate_event_delay_rows_n |
|:--------|---------------:|-----------------------------:|--------------------------------:|------------------:|-----------------------------:|-------------------------------:|-------------------------------:|
| train   |           4212 |                          702 |                             702 |                18 |                          nan |                            nan |                            nan |
| val     |           1398 |                          233 |                             233 |                18 |                          nan |                            nan |                            nan |
| test    |           1392 |                          232 |                             232 |                18 |                          nan |                            nan |                            nan |
| audit   |           7002 |                         1167 |                            1167 |                18 |                            0 |                              0 |                              0 |

核心约束：同一个 `event_uid` 的 6 个 delay 样本全部跟随同一个 `within_subject_split`。

## Validation-only 选择

| model_name                         | input_variant   | uses_subject_onehot   | test_used_for_selection   | selected_by                        |   best_epoch |   best_val_loss |   training_seconds | config_json                                                                                                                                                                                                                               |   validation_selection_score |   val_sample_rmse_weighted |   val_tail_rmse_weighted |   val_strong_under_rate_weighted |   val_peak_ratio_weighted |   validation_rank |
|:-----------------------------------|:----------------|:----------------------|:--------------------------|:-----------------------------------|-------------:|----------------:|-------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------:|---------------------------:|-------------------------:|---------------------------------:|--------------------------:|------------------:|
| v300_full_joint_h64_no_subject     | no_subject      | False                 | False                     | validation_original_remaining_only |           40 |        0.44749  |            37.5019 | {"batch_size": 384, "dropout": 0.08, "hidden_dim": 64, "lr": 0.0006, "max_epochs": 80, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 96, "n_heads": 4, "n_layers": 3, "patience": 12, "smooth_weight": 0.02, "weight_decay": 0.0003}  |                     0.75651  |                   0.478918 |                 0.53231  |                        0.0762485 |                   4.02999 |                 1 |
| v300_full_joint_h64_subject_onehot | subject_onehot  | True                  | False                     | validation_original_remaining_only |           33 |        0.54987  |            29.1636 | {"batch_size": 384, "dropout": 0.08, "hidden_dim": 64, "lr": 0.0006, "max_epochs": 80, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 112, "n_heads": 4, "n_layers": 3, "patience": 12, "smooth_weight": 0.02, "weight_decay": 0.0003} |                     0.844965 |                   0.52887  |                 0.582103 |                        0.166952  |                   2.97715 |                 2 |
| v300_full_joint_h96_subject_onehot | subject_onehot  | True                  | False                     | validation_original_remaining_only |           15 |        0.501738 |            33.7332 | {"batch_size": 256, "dropout": 0.11, "hidden_dim": 96, "lr": 0.0005, "max_epochs": 90, "min_lr": 1e-05, "mixer_layers": 3, "mlp_hidden": 144, "n_heads": 4, "n_layers": 4, "patience": 14, "smooth_weight": 0.04, "weight_decay": 0.0005} |                     0.859528 |                   0.540669 |                 0.598402 |                        0.131051  |                   2.98542 |                 3 |

validation 选择出的 v300 模型是：`v300_full_joint_h64_no_subject`。

## delay0 test 关键结果

- v300 选择模型 test/all：n=232, RMSE=0.5198, tail=0.6141
- 旧 v249 诊断参照 test/all：n=232, RMSE=0.3246, tail=0.3633
- v300 选择模型 test/within_bad_top10：n=24, RMSE=0.8600, tail=1.0473
- 旧 v249 诊断参照 test/within_bad_top10：n=24, RMSE=1.0383, tail=1.2935

注意：旧 v249 在 within-subject test 中有原始 split 暴露风险，不能当作正式公平基线；它这里只用于判断旧路线预测形状和当前完整重训之间的差距。

## 防线结论

- 同一事件跨 split 数：`0`。
- 旧 v249 是否参与训练：`False`。
- 模型选择是否看 test：`False`。
- within test 中旧 v249 原 train 暴露比例：`0.5819`。

## 产物

- `tables/v300_model_selection_validation.csv`：validation-only 选择表。
- `tables/v300_metrics_by_delay_and_bucket.csv`：完整分层指标。
- `tables/v300_delay0_group_summary.csv`：delay0 差样本组汇总。
- `tables/v300_delay0_event_wide_comparison.csv`：每个 delay0 事件的宽表，适合人工审查差样本。
- `figures/v300_test_selected_bad_top6_curves.png`：选择模型 test delay0 最差曲线。
- `v300_within_subject_full_predictions.npz`：完整预测数组。