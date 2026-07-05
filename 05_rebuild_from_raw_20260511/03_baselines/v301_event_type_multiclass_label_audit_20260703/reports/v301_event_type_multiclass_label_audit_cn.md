# v301 事件类型多分类标签草稿与有效性审计

## 这一步做了什么

本轮给 1167 个 delay0 事件生成了一版自动事件类型草稿标签，例如强减速/急停、紧急连续变道/避让、急左转、急右转、多段修正、晚响应/长事件等。

这些标签主要由 anchor 后 0-2s 的真实车辆行为和真实轨迹曲线派生，因此当前不能直接当作预测输入。它们的合理用途是：人工标注草稿、分层评估、辅助监督目标、以及后续建立锚点前可知事件条件标签的候选字典。

当前 v300 参照模型：`v300_full_joint_h64_no_subject`。

## 标签阈值

|   strong_steer_abs |   extreme_steer_abs |   high_steer_rate_abs |   high_yaw_abs |   high_ay_abs |   long_line_length |   large_lat_delta_abs |   large_lat_range |   speed_drop_large |   speed_drop_emergency |   brake_abs_peak_high |   ax_min_strong_decel |
|-------------------:|--------------------:|----------------------:|---------------:|--------------:|-------------------:|----------------------:|------------------:|-------------------:|-----------------------:|----------------------:|----------------------:|
|                  2 |                 2.6 |                     7 |           0.42 |       6.59338 |                  4 |               2.22752 |            3.4935 |            12.6436 |                19.5233 |              0.264304 |              -3.74024 |

## 标签分布和误差

| event_primary_type   |   n |   within_bad_top10_rate |   v300_rmse_mean |   v300_rmse_p90 |
|:---------------------|----:|------------------------:|-----------------:|----------------:|
| 复合急制动转向       |  17 |               0.294118  |         1.02137  |        1.54553  |
| 急左转               |   8 |               0.25      |         0.972372 |        2.16506  |
| 急右转               |   9 |               0.111111  |         0.711478 |        1.1877   |
| 强减速/急停          |  21 |               0.0952381 |         0.545063 |        0.77735  |
| 连续变道/横向避让    |  38 |               0.0789474 |         0.520678 |        1.07053  |
| 晚响应/长事件        |   6 |               0.166667  |         0.468044 |        0.705885 |
| 多段修正             | 122 |               0.0819672 |         0.425068 |        0.705484 |
| 紧急连续变道/避让    |  11 |               0         |         0.286421 |        0.459504 |

## 锚点前输入能否预测标签

validation 选择的标签分类器：`extra_trees_d6`。

| classifier     | split   |   n |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |
|:---------------|:--------|----:|-----------:|--------------------:|-----------:|--------------:|
| extra_trees_d6 | val     | 233 |   0.360515 |            0.374504 |   0.249391 |      0.402886 |

| classifier     | split   |   n |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |
|:---------------|:--------|----:|-----------:|--------------------:|-----------:|--------------:|
| extra_trees_d6 | test    | 232 |   0.340517 |            0.349276 |   0.228417 |      0.376792 |

如果 test macro-F1 / balanced accuracy 较低，说明这些事件类型虽然能解释未来行为，但锚点前车辆输入并不容易提前识别它们。

## 标签已知时的理论收益

| method                                | split   | group            |   n |   baseline_rmse_mean |   method_rmse_mean |   delta_vs_v300_mean |   delta_vs_v300_median |   improved_rate |
|:--------------------------------------|:--------|:-----------------|----:|---------------------:|-------------------:|---------------------:|-----------------------:|----------------:|
| oracle_true_label_residual_shrink0.75 | test    | all              | 232 |             0.519805 |           0.519097 |         -0.000707857 |           -0.00308321  |        0.568966 |
| oracle_true_label_residual_shrink0.75 | test    | within_bad_top10 |  24 |             0.859987 |           0.86592  |          0.00593224  |            0.00518681  |        0.416667 |
| oracle_true_label_residual_shrink0.75 | test    | within_bad_top20 |  47 |             0.690942 |           0.693344 |          0.00240227  |            0.00109762  |        0.446809 |
| predicted_label_residual_shrink0.5    | test    | all              | 232 |             0.519805 |           0.51873  |         -0.00107485  |           -0.000443459 |        0.512931 |
| predicted_label_residual_shrink0.5    | test    | within_bad_top10 |  24 |             0.859987 |           0.868272 |          0.00828452  |            0.00475445  |        0.291667 |
| predicted_label_residual_shrink0.5    | test    | within_bad_top20 |  47 |             0.690942 |           0.693423 |          0.00248076  |            0.000524759 |        0.425532 |

解释：`oracle_true_label_residual` 使用真实事件类型，属于理论上限；`predicted_label_residual` 使用锚点前输入预测出的事件类型，更接近可部署但通常更难。

## 当前判断

- 多分类事件标签值得保留为人工复核和辅助监督方向。
- 但当前自动标签是未来行为派生，不可直接作为正式输入。
- 下一步如果要进模型，应先让用户人工复核一小批高误差样本，确认标签定义是否符合驾驶语义。
- 只有能在锚点前被可靠识别的标签，才适合作为预测模型输入；否则只能作为训练辅助或报告分层。

## 产物

- `tables/v301_event_type_labels.csv`：每个事件的自动标签草稿。
- `tables/v301_manual_review_pack.csv`：建议人工优先复核的样本。
- `tables/v301_label_predictability_summary.csv`：标签可预测性。
- `tables/v301_label_residual_correction_summary.csv`：标签残差修正理论收益。
- `figures/v301_event_type_distribution.png`：标签分布。
- `figures/v301_event_type_test_rmse.png`：各标签 test RMSE。