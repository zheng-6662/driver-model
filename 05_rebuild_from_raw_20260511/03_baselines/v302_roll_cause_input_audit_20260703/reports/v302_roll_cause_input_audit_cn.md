# v302 侧倾诱因输入审计

## 这一步回答的问题

用户指出：车辆一开始发生侧倾的行为诱因，本来就应该作为输入。这个判断是成立的。v302 因此不再讨论“未来事件标签能不能直接输入”，而是检查当前输入中是否已有侧倾诱因，以及显式聚合这些因果可见信号后是否有增益。

当前 v300 参照模型：`v300_full_joint_h64_no_subject`。

## 当前输入是否已经包含侧倾诱因

- v236 preinput 总特征数：`609`。
- 侧倾/横摆/转向/道路等原始相关列数：`392`。
- roll 相关列数：`95`。
- ay 相关列数：`34`。
- yaw 相关列数：`63`。
- steer 相关列数：`33`。
- 是否包含 `current_roll_abs`：`True`。
- 是否包含 `current_roll_rate_abs`：`True`。
- 是否包含 `current_ay_abs`：`True`。
- 是否包含 `current_yaw_rate_abs`：`True`。

结论：当前 v236/v300 输入并不是没有看到侧倾诱因；roll、roll_rate、ay、yaw_rate、steering、road curvature 等信号已经在历史序列或 current 特征中出现。v302 的新增部分是把这些信号显式聚合成更容易被浅层模型利用的 summary。

## 信号覆盖

| signal                | source          |   feature_count |   time_min |   time_max | available   |
|:----------------------|:----------------|----------------:|-----------:|-----------:|:------------|
| steering              | v236_hist       |              31 |         -3 |          0 | True        |
| speed_kmh             | v236_hist       |              31 |         -3 |          0 | True        |
| ay                    | v236_hist       |              31 |         -3 |          0 | True        |
| yaw_rate              | v236_hist       |              31 |         -3 |          0 | True        |
| roll                  | v236_hist       |              31 |         -3 |          0 | True        |
| yaw                   | v236_hist       |              31 |         -3 |          0 | True        |
| roll_rate             | v236_hist       |              31 |         -3 |          0 | True        |
| roll_acc              | v236_hist       |              31 |         -3 |          0 | True        |
| brake                 | v236_hist       |              31 |         -3 |          0 | True        |
| lane_curvature        | v236_hist       |              31 |         -3 |          0 | True        |
| lateral_distance      | v236_hist       |              31 |         -3 |          0 | True        |
| road_curvature        | v236_known_road |              21 |          0 |          2 | True        |
| road_lateral_distance | v236_known_road |              21 |          0 |          2 | True        |

## 事件类型识别结果

| feature_set                     |   feature_n | chosen_classifier   |   val_macro_f1 |   test_accuracy |   test_balanced_accuracy |   test_macro_f1 |   test_weighted_f1 |
|:--------------------------------|------------:|:--------------------|---------------:|----------------:|-------------------------:|----------------:|-------------------:|
| base_plus_engineered_roll_cause |         910 | extra_trees_d10     |       0.280689 |        0.543103 |                 0.440003 |        0.390576 |           0.537704 |
| engineered_roll_cause_summary   |         301 | extra_trees_d10     |       0.313349 |        0.49569  |                 0.426138 |        0.353992 |           0.520328 |
| raw_roll_cause_subset           |         392 | extra_trees_d10     |       0.235598 |        0.456897 |                 0.375604 |        0.303736 |           0.456048 |
| base_all_v236_preinput          |         609 | extra_trees_d6      |       0.249391 |        0.340517 |                 0.349276 |        0.228417 |           0.376792 |

## 差样本识别结果

| feature_set                     |   feature_n | chosen_classifier   |   val_roc_auc |   test_roc_auc |   test_average_precision |   test_balanced_accuracy |   test_f1 |
|:--------------------------------|------------:|:--------------------|--------------:|---------------:|-------------------------:|-------------------------:|----------:|
| engineered_roll_cause_summary   |         301 | random_forest_d8    |      0.684809 |       0.635417 |                 0.159344 |                 0.497596 |         0 |
| base_plus_engineered_roll_cause |         910 | random_forest_d8    |      0.691786 |       0.622796 |                 0.138366 |                 0.5      |         0 |
| raw_roll_cause_subset           |         392 | random_forest_d8    |      0.672847 |       0.58153  |                 0.128735 |                 0.5      |         0 |
| base_all_v236_preinput          |         609 | random_forest_d8    |      0.651715 |       0.573518 |                 0.128677 |                 0.5      |         0 |

## 残差修正结果

| feature_set                     |   feature_n | regressor          |   selected_shrink |   val_rmse_mean |
|:--------------------------------|------------:|:-------------------|------------------:|----------------:|
| base_plus_engineered_roll_cause |         910 | extra_trees_reg_d6 |              1    |        0.521058 |
| engineered_roll_cause_summary   |         301 | extra_trees_reg_d6 |              1    |        0.521329 |
| base_all_v236_preinput          |         609 | extra_trees_reg_d6 |              1    |        0.52452  |
| raw_roll_cause_subset           |         392 | extra_trees_reg_d6 |              1    |        0.524722 |
| base_plus_engineered_roll_cause |         910 | ridge_alpha100     |              0.25 |        0.530271 |
| base_all_v236_preinput          |         609 | ridge_alpha100     |              0.25 |        0.531013 |
| engineered_roll_cause_summary   |         301 | ridge_alpha100     |              0.5  |        0.532034 |
| raw_roll_cause_subset           |         392 | ridge_alpha100     |              0.25 |        0.532525 |
| engineered_roll_cause_summary   |         301 | ridge_alpha10      |              0    |        0.534133 |
| raw_roll_cause_subset           |         392 | ridge_alpha10      |              0    |        0.534133 |
| base_all_v236_preinput          |         609 | ridge_alpha10      |              0    |        0.534133 |
| base_plus_engineered_roll_cause |         910 | ridge_alpha10      |              0    |        0.534133 |

| method                                                         | group            |   n |   baseline_rmse_mean |   method_rmse_mean |   delta_vs_v300_mean |   improved_rate |
|:---------------------------------------------------------------|:-----------------|----:|---------------------:|-------------------:|---------------------:|----------------:|
| engineered_roll_cause_summary::extra_trees_reg_d6::shrink1.0   | all              | 232 |             0.519805 |           0.510968 |         -0.00883663  |        0.607759 |
| base_plus_engineered_roll_cause::extra_trees_reg_d6::shrink1.0 | all              | 232 |             0.519805 |           0.511185 |         -0.00861995  |        0.568966 |
| raw_roll_cause_subset::extra_trees_reg_d6::shrink1.0           | all              | 232 |             0.519805 |           0.512449 |         -0.00735615  |        0.560345 |
| base_all_v236_preinput::extra_trees_reg_d6::shrink1.0          | all              | 232 |             0.519805 |           0.513068 |         -0.00673722  |        0.560345 |
| engineered_roll_cause_summary::ridge_alpha100::shrink0.5       | all              | 232 |             0.519805 |           0.516718 |         -0.00308718  |        0.568966 |
| raw_roll_cause_subset::ridge_alpha100::shrink0.25              | all              | 232 |             0.519805 |           0.518694 |         -0.00111097  |        0.616379 |
| base_all_v236_preinput::ridge_alpha10::shrink0.0               | all              | 232 |             0.519805 |           0.519805 |          0           |        0        |
| raw_roll_cause_subset::ridge_alpha10::shrink0.0                | all              | 232 |             0.519805 |           0.519805 |          0           |        0        |
| engineered_roll_cause_summary::ridge_alpha10::shrink0.0        | all              | 232 |             0.519805 |           0.519805 |          0           |        0        |
| base_plus_engineered_roll_cause::ridge_alpha10::shrink0.0      | all              | 232 |             0.519805 |           0.519805 |          0           |        0        |
| base_all_v236_preinput::ridge_alpha100::shrink0.25             | all              | 232 |             0.519805 |           0.519829 |          2.42498e-05 |        0.603448 |
| base_plus_engineered_roll_cause::ridge_alpha100::shrink0.25    | all              | 232 |             0.519805 |           0.519838 |          3.3142e-05  |        0.586207 |
| base_all_v236_preinput::ridge_alpha10::shrink0.0               | within_bad_top10 |  24 |             0.859987 |           0.859987 |          0           |        0        |
| raw_roll_cause_subset::ridge_alpha10::shrink0.0                | within_bad_top10 |  24 |             0.859987 |           0.859987 |          0           |        0        |
| engineered_roll_cause_summary::ridge_alpha10::shrink0.0        | within_bad_top10 |  24 |             0.859987 |           0.859987 |          0           |        0        |
| base_plus_engineered_roll_cause::ridge_alpha10::shrink0.0      | within_bad_top10 |  24 |             0.859987 |           0.859987 |          0           |        0        |
| engineered_roll_cause_summary::extra_trees_reg_d6::shrink1.0   | within_bad_top10 |  24 |             0.859987 |           0.862325 |          0.00233781  |        0.416667 |
| raw_roll_cause_subset::ridge_alpha100::shrink0.25              | within_bad_top10 |  24 |             0.859987 |           0.864003 |          0.00401517  |        0.458333 |
| base_plus_engineered_roll_cause::extra_trees_reg_d6::shrink1.0 | within_bad_top10 |  24 |             0.859987 |           0.864584 |          0.00459634  |        0.333333 |
| raw_roll_cause_subset::extra_trees_reg_d6::shrink1.0           | within_bad_top10 |  24 |             0.859987 |           0.866704 |          0.00671711  |        0.416667 |
| base_all_v236_preinput::extra_trees_reg_d6::shrink1.0          | within_bad_top10 |  24 |             0.859987 |           0.867828 |          0.00784018  |        0.333333 |
| base_plus_engineered_roll_cause::ridge_alpha100::shrink0.25    | within_bad_top10 |  24 |             0.859987 |           0.86799  |          0.00800275  |        0.291667 |
| base_all_v236_preinput::ridge_alpha100::shrink0.25             | within_bad_top10 |  24 |             0.859987 |           0.869014 |          0.00902619  |        0.25     |
| engineered_roll_cause_summary::ridge_alpha100::shrink0.5       | within_bad_top10 |  24 |             0.859987 |           0.870648 |          0.0106607   |        0.333333 |
| base_all_v236_preinput::ridge_alpha10::shrink0.0               | within_bad_top20 |  47 |             0.690942 |           0.690942 |          0           |        0        |
| raw_roll_cause_subset::ridge_alpha10::shrink0.0                | within_bad_top20 |  47 |             0.690942 |           0.690942 |          0           |        0        |
| engineered_roll_cause_summary::ridge_alpha10::shrink0.0        | within_bad_top20 |  47 |             0.690942 |           0.690942 |          0           |        0        |
| base_plus_engineered_roll_cause::ridge_alpha10::shrink0.0      | within_bad_top20 |  47 |             0.690942 |           0.690942 |          0           |        0        |
| engineered_roll_cause_summary::extra_trees_reg_d6::shrink1.0   | within_bad_top20 |  47 |             0.690942 |           0.691629 |          0.000686837 |        0.489362 |
| base_plus_engineered_roll_cause::extra_trees_reg_d6::shrink1.0 | within_bad_top20 |  47 |             0.690942 |           0.692476 |          0.00153382  |        0.446809 |
| raw_roll_cause_subset::ridge_alpha100::shrink0.25              | within_bad_top20 |  47 |             0.690942 |           0.693214 |          0.00227251  |        0.489362 |
| raw_roll_cause_subset::extra_trees_reg_d6::shrink1.0           | within_bad_top20 |  47 |             0.690942 |           0.694008 |          0.00306609  |        0.468085 |
| base_plus_engineered_roll_cause::ridge_alpha100::shrink0.25    | within_bad_top20 |  47 |             0.690942 |           0.694863 |          0.00392066  |        0.425532 |
| base_all_v236_preinput::extra_trees_reg_d6::shrink1.0          | within_bad_top20 |  47 |             0.690942 |           0.69505  |          0.0041085   |        0.446809 |
| base_all_v236_preinput::ridge_alpha100::shrink0.25             | within_bad_top20 |  47 |             0.690942 |           0.69583  |          0.00488832  |        0.382979 |
| engineered_roll_cause_summary::ridge_alpha100::shrink0.5       | within_bad_top20 |  47 |             0.690942 |           0.69749  |          0.00654757  |        0.446809 |

## 当前判断

- 用户关于“侧倾诱因应作为输入”的判断是对的；严格说，这些因果可见信号已经在当前输入里。
- 如果 v302 显示 base_plus_engineered 只有很小改善，说明问题不是没有输入 roll-cause，而是模型没有从这些锚点前信号中稳定推断未来分叉行为。
- 如果 raw_roll_cause_subset 或 engineered_roll_cause_summary 单独接近 base_all，说明侧倾诱因是关键输入组；后续可以围绕这组信号做专门编码，而不是盲目增加所有通道。
- 事件类型标签仍建议作为辅助监督/分层诊断，而不是直接作为未来标签输入。

## 产物

- `tables/v302_roll_cause_raw_feature_audit.csv`：当前输入中侧倾诱因相关列数量。
- `tables/v302_roll_cause_signal_coverage.csv`：各类历史/道路信号覆盖情况。
- `tables/v302_roll_cause_summary_features.csv`：逐事件 roll-cause summary 特征。
- `tables/v302_multiclass_predictability_by_input.csv`：不同输入集合的事件类型识别结果。
- `tables/v302_bad_sample_binary_by_input.csv`：不同输入集合的差样本识别结果。
- `tables/v302_residual_regression_summary.csv`：不同输入集合的残差修正结果。
- `figures/v302_event_type_macro_f1_by_input.png`
- `figures/v302_badtop10_auc_by_input.png`
- `figures/v302_residual_delta_by_input.png`