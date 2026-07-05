# v252 输入相似样本的未来分叉审计

## 本轮问题

本轮只回答一个问题：锚点前输入相似的样本，锚点后的真实 steering_delta 是否会明显分叉。
如果答案是肯定的，那么当前差样本并不只是模型结构问题，而是存在同输入多未来或可观测信息不足。

## 固定边界

- 固定 v250 validation-only 选出的 `v250_minimal_lateral7` 输入口径。
- 不重新训练，不调通道，不删样本，不做 anchor selector / gate / router。
- 每个 test sample 只在同 delay 的 train sample 中找近邻，避免 delay 口径混在一起。
- 近邻搜索使用标准化后的 `hist + road + phase` sample-level 输入。

## v250 选择来源

| model_name            |   n_hist_channels | channels                                                            |   best_epoch |   best_val_loss | accepted_as_channel_candidate   |
|:----------------------|------------------:|:--------------------------------------------------------------------|-------------:|----------------:|:--------------------------------|
| v250_minimal_lateral7 |                 7 | steering|speed_kmh|ay|yaw_rate|roll|lane_curvature|lateral_distance |           19 |        0.487806 | True                            |

## 总体摘要

- 全 test rolling sample：N=1104，近邻未来两两 RMSE 均值=0.707，query-vs-neighbor 未来 RMSE 均值=0.686，高近邻分歧率=0.250。
- 当前 v250 bad_top10 样本：N=111，近邻未来两两 RMSE 均值=0.837，query-vs-neighbor 未来 RMSE 均值=1.063，高近邻分歧率=0.441。
- 0ms 原始锚点：N=184，近邻未来两两 RMSE 均值=0.836，query-vs-neighbor 未来 RMSE 均值=0.783。

## Bucket / Delay 摘要

| bucket             | delay_ms   |    n |   event_n |   neighbor_input_distance_mean |   neighbor_future_pairwise_rmse_mean |   neighbor_future_to_query_mean_rmse |   high_neighbor_divergence_q75_rate |   tail_rmse_v250_mean |
|:-------------------|:-----------|-----:|----------:|-------------------------------:|-------------------------------------:|-------------------------------------:|------------------------------------:|----------------------:|
| all                | all_delays | 1104 |       184 |                       0.471445 |                             0.707214 |                             0.685933 |                           0.25      |              0.323335 |
| all                | 0          |  184 |       184 |                       0.426964 |                             0.835706 |                             0.78284  |                           0.440217  |              0.396888 |
| all                | 600        |  184 |       184 |                       0.472855 |                             0.71494  |                             0.694314 |                           0.255435  |              0.318    |
| all                | 1000       |  184 |       184 |                       0.497878 |                             0.566391 |                             0.562144 |                           0.076087  |              0.236594 |
| normal_predictable | all_delays |  594 |        99 |                       0.533819 |                             0.651821 |                             0.575246 |                           0.16835   |              0.237091 |
| normal_predictable | 0          |   99 |        99 |                       0.531265 |                             0.750757 |                             0.634701 |                           0.343434  |              0.292139 |
| normal_predictable | 600        |   99 |        99 |                       0.519871 |                             0.656791 |                             0.594809 |                           0.141414  |              0.232909 |
| normal_predictable | 1000       |   99 |        99 |                       0.535727 |                             0.543718 |                             0.475575 |                           0.0505051 |              0.177867 |
| observe_later_like | all_delays |  162 |        27 |                       0.496346 |                             0.737814 |                             0.905219 |                           0.271605  |              0.520208 |
| observe_later_like | 0          |   27 |        27 |                       0.234439 |                             0.838037 |                             0.987318 |                           0.407407  |              0.639918 |
| observe_later_like | 600        |   27 |        27 |                       0.560067 |                             0.796472 |                             0.908924 |                           0.37037   |              0.511828 |
| observe_later_like | 1000       |   27 |        27 |                       0.616449 |                             0.575679 |                             0.793157 |                           0.111111  |              0.375296 |
| strong_steer       | all_delays |  480 |        80 |                       0.335323 |                             0.780994 |                             0.825665 |                           0.35625   |              0.422244 |
| strong_steer       | 0          |   80 |        80 |                       0.30997  |                             0.937814 |                             0.955306 |                           0.55      |              0.504266 |
| strong_steer       | 600        |   80 |        80 |                       0.336195 |                             0.791984 |                             0.823024 |                           0.4125    |              0.422398 |
| strong_steer       | 1000       |   80 |        80 |                       0.372516 |                             0.604277 |                             0.677419 |                           0.1125    |              0.307555 |
| bad_top10_v250     | all_delays |  111 |        41 |                       0.778253 |                             0.837383 |                             1.06271  |                           0.441441  |              0.878316 |
| bad_top10_v250     | 0          |   30 |        30 |                       0.465177 |                             0.96286  |                             1.14826  |                           0.566667  |              0.918189 |
| bad_top10_v250     | 600        |   17 |        17 |                       0.955019 |                             0.794635 |                             1.05943  |                           0.352941  |              0.840224 |
| bad_top10_v250     | 1000       |   10 |        10 |                       1.18714  |                             0.655128 |                             0.980105 |                           0.2       |              0.742203 |

## 误差与未来分叉相关

| subset     | x_metric                           | y_metric       |    n |    pearson |    spearman |
|:-----------|:-----------------------------------|:---------------|-----:|-----------:|------------:|
| all_delays | neighbor_future_pairwise_rmse_mean | tail_rmse_v250 | 1104 |  0.260621  |  0.287306   |
| all_delays | neighbor_future_pairwise_rmse_mean | tail_rmse_v241 | 1104 |  0.175363  |  0.203907   |
| all_delays | neighbor_future_to_query_mean_rmse | tail_rmse_v250 | 1104 |  0.543446  |  0.495131   |
| all_delays | neighbor_future_to_query_mean_rmse | tail_rmse_v241 | 1104 |  0.568839  |  0.503109   |
| all_delays | neighbor_mean_curve_to_query_rmse  | tail_rmse_v250 | 1104 |  0.517887  |  0.451228   |
| all_delays | neighbor_mean_curve_to_query_rmse  | tail_rmse_v241 | 1104 |  0.591116  |  0.508232   |
| all_delays | neighbor_input_distance_mean       | tail_rmse_v250 | 1104 |  0.130024  |  0.047415   |
| all_delays | neighbor_input_distance_mean       | tail_rmse_v241 | 1104 |  0.0215454 | -0.00599922 |
| 0ms        | neighbor_future_pairwise_rmse_mean | tail_rmse_v250 |  184 |  0.177777  |  0.195517   |
| 0ms        | neighbor_future_pairwise_rmse_mean | tail_rmse_v241 |  184 |  0.148256  |  0.225874   |
| 0ms        | neighbor_future_to_query_mean_rmse | tail_rmse_v250 |  184 |  0.565941  |  0.490961   |
| 0ms        | neighbor_future_to_query_mean_rmse | tail_rmse_v241 |  184 |  0.550366  |  0.502664   |
| 0ms        | neighbor_mean_curve_to_query_rmse  | tail_rmse_v250 |  184 |  0.559124  |  0.480436   |
| 0ms        | neighbor_mean_curve_to_query_rmse  | tail_rmse_v241 |  184 |  0.591014  |  0.509428   |
| 0ms        | neighbor_input_distance_mean       | tail_rmse_v250 |  184 |  0.0590543 |  0.0223652  |
| 0ms        | neighbor_input_distance_mean       | tail_rmse_v241 |  184 | -0.0264449 | -0.00943345 |

## 高误差与高分叉重叠

| subset     | ambiguity_group   |    n |   ambiguity_n |   bad_top10_v250_n |   overlap_bad250_and_ambiguity_n |   bad250_covered_by_ambiguity_rate |   bad250_rate_inside_ambiguity |   bad250_rate_outside_ambiguity |   bad_top10_v241_n |   overlap_bad241_and_ambiguity_n |   bad241_covered_by_ambiguity_rate |   bad241_rate_inside_ambiguity |   bad241_rate_outside_ambiguity |
|:-----------|:------------------|-----:|--------------:|-------------------:|---------------------------------:|-----------------------------------:|-------------------------------:|--------------------------------:|-------------------:|---------------------------------:|-----------------------------------:|-------------------------------:|--------------------------------:|
| all_delays | high_q75          | 1104 |           276 |                111 |                               49 |                           0.441441 |                       0.177536 |                       0.0748792 |                111 |                               41 |                           0.369369 |                       0.148551 |                       0.0845411 |
| all_delays | very_high_q90     | 1104 |           111 |                111 |                               23 |                           0.207207 |                       0.207207 |                       0.0886203 |                111 |                               17 |                           0.153153 |                       0.153153 |                       0.0946626 |
| delay0     | high_q75          |  184 |            81 |                 30 |                               17 |                           0.566667 |                       0.209877 |                       0.126214  |                 24 |                               15 |                           0.625    |                       0.185185 |                       0.0873786 |
| delay0     | very_high_q90     |  184 |            49 |                 30 |                               11 |                           0.366667 |                       0.22449  |                       0.140741  |                 24 |                                8 |                           0.333333 |                       0.163265 |                       0.118519  |

## 人工审查优先样本

| event_uid                                         | subject   |   delay_ms | casebook_reason                             |   tail_rmse_v250 |   neighbor_future_pairwise_rmse_mean |   neighbor_future_to_query_mean_rmse |   neighbor_input_distance_mean |
|:--------------------------------------------------|:----------|-----------:|:--------------------------------------------|-----------------:|-------------------------------------:|-------------------------------------:|-------------------------------:|
| rjy_Entity_Recording_2025_09_28_19_51_44_v108_039 | rjy       |          0 | bad_top10_v250_and_high_neighbor_divergence |         2.36684  |                             0.824684 |                             1.65839  |                       0.277794 |
| rjy_Entity_Recording_2025_09_28_20_02_20_v108_038 | rjy       |          0 | bad_top10_v250_and_high_neighbor_divergence |         0.926919 |                             1.60853  |                             1.70473  |                       0.594125 |
| tyy_Entity_Recording_2025_09_28_14_40_01_v108_012 | tyy       |          0 | bad_top10_v250_and_high_neighbor_divergence |         1.84297  |                             1.38008  |                             0.977216 |                       1.32754  |
| rjy_Entity_Recording_2025_09_28_20_02_20_v108_010 | rjy       |        200 | bad_top10_v250_and_high_neighbor_divergence |         0.780256 |                             1.49215  |                             1.34088  |                       2.54208  |
| rjy_Entity_Recording_2025_09_28_20_02_20_v108_010 | rjy       |          0 | bad_top10_v250_and_high_neighbor_divergence |         0.629123 |                             1.56751  |                             1.37296  |                       2.39056  |
| rjy_Entity_Recording_2025_09_28_20_02_20_v108_021 | rjy       |          0 | bad_top10_v250_and_high_neighbor_divergence |         0.860862 |                             1.59554  |                             1.12477  |                       0.999836 |
| tyy_Entity_Recording_2025_09_28_14_23_43_v108_037 | tyy       |          0 | high_neighbor_divergence_or_mismatch        |         0.49713  |                             1.51953  |                             1.59689  |                       0.241095 |
| tyy_Entity_Recording_2025_09_28_14_40_01_v108_001 | tyy       |          0 | high_neighbor_divergence_or_mismatch        |         0.517921 |                             1.63766  |                             1.39387  |                       0.205442 |
| tyy_Entity_Recording_2025_09_28_14_57_17_v108_022 | tyy       |          0 | bad_top10_v250_and_high_neighbor_divergence |         0.646749 |                             1.67469  |                             1.13933  |                       0.246871 |
| rjy_Entity_Recording_2025_09_28_20_02_20_v108_038 | rjy       |        200 | bad_top10_v250_and_high_neighbor_divergence |         0.79637  |                             1.42751  |                             1.2889   |                       0.572132 |
| rjy_Entity_Recording_2025_09_28_20_02_20_v108_041 | rjy       |          0 | bad_top10_v250                              |         0.998858 |                             0.79619  |                             2.15414  |                       0.716888 |
| rjy_Entity_Recording_2025_09_28_20_02_20_v108_010 | rjy       |        400 | high_neighbor_divergence_or_mismatch        |         0.474456 |                             1.30369  |                             1.33868  |                       2.557    |

## 解释

若 casebook 图中左侧锚点前 steering 历史高度相似，但右侧近邻真实未来呈扇形分叉，则说明单条确定性曲线预测会自然学成折中曲线。此时继续增强 MLP/TCN/attention 只能有限改善，更合理的下一步是概率预测、多模态候选轨迹或显式不确定性建模。

## 关键图

- `figures\v252_error_vs_neighbor_future_divergence.png`
- `figures\v252_neighbor_divergence_by_error_group.png`
- `figures\v252_delay_future_divergence_summary.png`
- `figures\v252_casebook_high_error_high_ambiguity.png`
- `figures\v252_casebook_worst_regression_neighbors.png`

## 关键表

- `tables/v252_neighbor_divergence_by_sample.csv`
- `tables/v252_neighbor_detail.csv`
- `tables/v252_summary_by_delay_bucket.csv`
- `tables/v252_error_ambiguity_correlation.csv`
- `tables/v252_high_ambiguity_error_overlap.csv`
- `tables/v252_casebook_index.csv`
