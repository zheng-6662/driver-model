# v253a 驾驶风格/生理信号消歧审计

## 本轮问题

v252 已经证明一部分差样本属于“锚点前输入相似，但锚点后真实未来分叉”。本轮检查驾驶风格和生理信号是否能让近邻未来更集中。

## 固定边界

- 不训练新预测模型，不改 v250/v251/v252。
- 不使用 test 选择模型；所有结果只作为状态信号是否值得进入下一步模型的审计证据。
- 旧 stage04 style 表与当前 v252 样本不匹配，本轮只记录其不可直接复用，不直接使用。
- 驾驶风格从当前 raw vehicle 重新提取 `last60_guard3`，窗口截止到 `observation_s - 3s`。
- 生理信号从 1Hz 表提取 `pre5_pre2` 和 `pre2_0`，窗口均不超过 `observation_s`。

## 旧 style 表匹配审计

| check                                   |   value |
|:----------------------------------------|--------:|
| old_style_table_rows                    |     270 |
| sample_id_intersection_count            |       0 |
| event_uid_intersection_count            |       0 |
| subject_session_anchor_round_match_rows |       0 |

## 特征可用性

| feature_block                  |   n_features | source                                          |
|:-------------------------------|-------------:|:------------------------------------------------|
| vehicle_base_v250_minimal      |          268 | v250_minimal_lateral7 hist+road+phase           |
| driving_style_last60_guard3    |          127 | raw vehicle recomputed for current v252 samples |
| physio_recent_pre2_0_and_delta |          198 | physio_features_1hz.csv causal pre-observation  |
| physio_guard_pre5_pre2         |          170 | physio_features_1hz.csv guarded baseline        |

## 驾驶风格提取摘要

|   n_samples |   ok_rate |   short_or_missing_rate |   mean_row_count | post_observation_any   | overlap_direct_input_any   |
|------------:|----------:|------------------------:|-----------------:|:-----------------------|:---------------------------|
|        7002 |         1 |                       0 |          23498.3 | False                  | False                      |

## 生理提取摘要

|   n_samples |   recording_inventory_match_rate |   has_1hz_rate |   recent_pre2_0_rows_mean |   guard_pre5_pre2_rows_mean | post_observation_any   |
|------------:|---------------------------------:|---------------:|--------------------------:|----------------------------:|:-----------------------|
|        7002 |                         0.919452 |       0.919452 |                    2.0977 |                      3.0977 | False                  |

## 关键对比

| feature_group                     | bucket             | delay_ms   |    n |   neighbor_future_pairwise_rmse_mean |   neighbor_future_to_query_mean_rmse |   delta_query_neighbor_vs_vehicle_only |   high_neighbor_divergence_q75_rate |   neighbor_same_subject_rate_mean |   neighbor_same_recording_rate_mean |
|:----------------------------------|:-------------------|:-----------|-----:|-------------------------------------:|-------------------------------------:|---------------------------------------:|------------------------------------:|----------------------------------:|------------------------------------:|
| vehicle_only                      | all                | all_delays | 1104 |                             0.707214 |                             0.685933 |                            0           |                            0.25     |                                 0 |                                   0 |
| vehicle_only                      | all                | 0          |  184 |                             0.835706 |                             0.78284  |                            0           |                            0.440217 |                                 0 |                                   0 |
| vehicle_only                      | observe_later_like | all_delays |  162 |                             0.737814 |                             0.905219 |                            0           |                            0.271605 |                                 0 |                                   0 |
| vehicle_only                      | observe_later_like | 0          |   27 |                             0.838037 |                             0.987318 |                            0           |                            0.407407 |                                 0 |                                   0 |
| vehicle_only                      | strong_steer       | all_delays |  480 |                             0.780994 |                             0.825665 |                            0           |                            0.35625  |                                 0 |                                   0 |
| vehicle_only                      | strong_steer       | 0          |   80 |                             0.937814 |                             0.955306 |                            0           |                            0.55     |                                 0 |                                   0 |
| vehicle_only                      | bad_top10_v250     | all_delays |  111 |                             0.837383 |                             1.06271  |                            0           |                            0.441441 |                                 0 |                                   0 |
| vehicle_only                      | bad_top10_v250     | 0          |   30 |                             0.96286  |                             1.14826  |                            0           |                            0.566667 |                                 0 |                                   0 |
| vehicle_plus_style_w0.25          | all                | all_delays | 1104 |                             0.714131 |                             0.691839 |                            0.0059054   |                            0.25     |                                 0 |                                   0 |
| vehicle_plus_style_w0.25          | all                | 0          |  184 |                             0.852212 |                             0.79052  |                            0.00768038  |                            0.472826 |                                 0 |                                   0 |
| vehicle_plus_style_w0.25          | observe_later_like | all_delays |  162 |                             0.757057 |                             0.922876 |                            0.0176568   |                            0.339506 |                                 0 |                                   0 |
| vehicle_plus_style_w0.25          | observe_later_like | 0          |   27 |                             0.878232 |                             1.00578  |                            0.0184593   |                            0.555556 |                                 0 |                                   0 |
| vehicle_plus_style_w0.25          | strong_steer       | all_delays |  480 |                             0.784437 |                             0.829385 |                            0.00372069  |                            0.3625   |                                 0 |                                   0 |
| vehicle_plus_style_w0.25          | strong_steer       | 0          |   80 |                             0.946948 |                             0.957071 |                            0.00176512  |                            0.6125   |                                 0 |                                   0 |
| vehicle_plus_style_w0.25          | bad_top10_v250     | all_delays |  111 |                             0.846338 |                             1.06371  |                            0.000995196 |                            0.432432 |                                 0 |                                   0 |
| vehicle_plus_style_w0.25          | bad_top10_v250     | 0          |   30 |                             0.970165 |                             1.13859  |                           -0.00966147  |                            0.533333 |                                 0 |                                   0 |
| vehicle_plus_style_w0.50          | all                | all_delays | 1104 |                             0.728042 |                             0.71461  |                            0.0286767   |                            0.25     |                                 0 |                                   0 |
| vehicle_plus_style_w0.50          | all                | 0          |  184 |                             0.9101   |                             0.842561 |                            0.0597211   |                            0.570652 |                                 0 |                                   0 |
| vehicle_plus_style_w0.50          | observe_later_like | all_delays |  162 |                             0.759709 |                             0.952312 |                            0.0470927   |                            0.290123 |                                 0 |                                   0 |
| vehicle_plus_style_w0.50          | observe_later_like | 0          |   27 |                             0.96755  |                             1.06583  |                            0.0785164   |                            0.62963  |                                 0 |                                   0 |
| vehicle_plus_style_w0.50          | strong_steer       | all_delays |  480 |                             0.792193 |                             0.863211 |                            0.0375458   |                            0.341667 |                                 0 |                                   0 |
| vehicle_plus_style_w0.50          | strong_steer       | 0          |   80 |                             1.01378  |                             1.04391  |                            0.0886042   |                            0.7125   |                                 0 |                                   0 |
| vehicle_plus_style_w0.50          | bad_top10_v250     | all_delays |  111 |                             0.846534 |                             1.08676  |                            0.0240446   |                            0.441441 |                                 0 |                                   0 |
| vehicle_plus_style_w0.50          | bad_top10_v250     | 0          |   30 |                             0.987123 |                             1.14092  |                           -0.00733069  |                            0.6      |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.25  | all                | all_delays | 1104 |                             0.735568 |                             0.700936 |                            0.0150024   |                            0.25     |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.25  | all                | 0          |  184 |                             0.908263 |                             0.824889 |                            0.0420489   |                            0.543478 |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.25  | observe_later_like | all_delays |  162 |                             0.799426 |                             0.928248 |                            0.023029    |                            0.358025 |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.25  | observe_later_like | 0          |   27 |                             1.01984  |                             1.058    |                            0.0706783   |                            0.703704 |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.25  | strong_steer       | all_delays |  480 |                             0.811756 |                             0.848317 |                            0.0226525   |                            0.360417 |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.25  | strong_steer       | 0          |   80 |                             1.01858  |                             1.0127   |                            0.0573991   |                            0.725    |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.25  | bad_top10_v250     | all_delays |  111 |                             0.874798 |                             1.07029  |                            0.0075755   |                            0.405405 |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.25  | bad_top10_v250     | 0          |   30 |                             1.03043  |                             1.15086  |                            0.00260762  |                            0.633333 |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.50  | all                | all_delays | 1104 |                             0.821357 |                             0.772371 |                            0.0864373   |                            0.25     |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.50  | all                | 0          |  184 |                             1.06106  |                             0.954834 |                            0.171995    |                            0.61413  |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.50  | observe_later_like | all_delays |  162 |                             0.904534 |                             1.0343   |                            0.129085    |                            0.345679 |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.50  | observe_later_like | 0          |   27 |                             1.27408  |                             1.27554  |                            0.288219    |                            0.777778 |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.50  | strong_steer       | all_delays |  480 |                             0.905553 |                             0.94959  |                            0.123925    |                            0.345833 |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.50  | strong_steer       | 0          |   80 |                             1.17985  |                             1.19212  |                            0.236811    |                            0.7625   |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.50  | bad_top10_v250     | all_delays |  111 |                             0.923184 |                             1.13111  |                            0.0683987   |                            0.396396 |                                 0 |                                   0 |
| vehicle_plus_physio_recent_w0.50  | bad_top10_v250     | 0          |   30 |                             1.1203   |                             1.21582  |                            0.0675692   |                            0.7      |                                 0 |                                   0 |
| vehicle_plus_physio_guarded_w0.50 | all                | all_delays | 1104 |                             0.840805 |                             0.775638 |                            0.0897045   |                            0.25     |                                 0 |                                   0 |
| vehicle_plus_physio_guarded_w0.50 | all                | 0          |  184 |                             1.06769  |                             0.95007  |                            0.16723     |                            0.61413  |                                 0 |                                   0 |
| vehicle_plus_physio_guarded_w0.50 | observe_later_like | all_delays |  162 |                             0.901026 |                             1.02218  |                            0.116959    |                            0.296296 |                                 0 |                                   0 |
| vehicle_plus_physio_guarded_w0.50 | observe_later_like | 0          |   27 |                             1.16967  |                             1.23161  |                            0.24429     |                            0.666667 |                                 0 |                                   0 |
| vehicle_plus_physio_guarded_w0.50 | strong_steer       | all_delays |  480 |                             0.911737 |                             0.936304 |                            0.110639    |                            0.327083 |                                 0 |                                   0 |
| vehicle_plus_physio_guarded_w0.50 | strong_steer       | 0          |   80 |                             1.15795  |                             1.16112  |                            0.205812    |                            0.725    |                                 0 |                                   0 |
| vehicle_plus_physio_guarded_w0.50 | bad_top10_v250     | all_delays |  111 |                             0.940581 |                             1.1323   |                            0.0695936   |                            0.414414 |                                 0 |                                   0 |
| vehicle_plus_physio_guarded_w0.50 | bad_top10_v250     | 0          |   30 |                             1.10729  |                             1.22972  |                            0.0814664   |                            0.666667 |                                 0 |                                   0 |
| vehicle_plus_style_physio_w0.50   | all                | all_delays | 1104 |                             0.819837 |                             0.772623 |                            0.0866902   |                            0.25     |                                 0 |                                   0 |
| vehicle_plus_style_physio_w0.50   | all                | 0          |  184 |                             1.04802  |                             0.955264 |                            0.172425    |                            0.61413  |                                 0 |                                   0 |
| vehicle_plus_style_physio_w0.50   | observe_later_like | all_delays |  162 |                             0.929368 |                             1.05971  |                            0.154489    |                            0.364198 |                                 0 |                                   0 |
| vehicle_plus_style_physio_w0.50   | observe_later_like | 0          |   27 |                             1.24904  |                             1.28428  |                            0.296961    |                            0.814815 |                                 0 |                                   0 |
| vehicle_plus_style_physio_w0.50   | strong_steer       | all_delays |  480 |                             0.89952  |                             0.946524 |                            0.12086     |                            0.329167 |                                 0 |                                   0 |
| vehicle_plus_style_physio_w0.50   | strong_steer       | 0          |   80 |                             1.17131  |                             1.19747  |                            0.242168    |                            0.75     |                                 0 |                                   0 |
| vehicle_plus_style_physio_w0.50   | bad_top10_v250     | all_delays |  111 |                             0.948194 |                             1.14659  |                            0.083876    |                            0.432432 |                                 0 |                                   0 |
| vehicle_plus_style_physio_w0.50   | bad_top10_v250     | 0          |   30 |                             1.13925  |                             1.23729  |                            0.08903     |                            0.7      |                                 0 |                                   0 |

## 相关性摘要

| feature_group                     | subset     | x_metric                           | y_metric       |    n |   pearson |   spearman |
|:----------------------------------|:-----------|:-----------------------------------|:---------------|-----:|----------:|-----------:|
| vehicle_only                      | all_delays | neighbor_future_pairwise_rmse_mean | tail_rmse_v250 | 1104 | 0.260621  |  0.287306  |
| vehicle_only                      | all_delays | neighbor_future_to_query_mean_rmse | tail_rmse_v250 | 1104 | 0.543446  |  0.495131  |
| vehicle_only                      | all_delays | neighbor_input_distance_mean       | tail_rmse_v250 | 1104 | 0.130024  |  0.047415  |
| vehicle_plus_style_w0.25          | all_delays | neighbor_future_pairwise_rmse_mean | tail_rmse_v250 | 1104 | 0.269663  |  0.271557  |
| vehicle_plus_style_w0.25          | all_delays | neighbor_future_to_query_mean_rmse | tail_rmse_v250 | 1104 | 0.540184  |  0.488442  |
| vehicle_plus_style_w0.25          | all_delays | neighbor_input_distance_mean       | tail_rmse_v250 | 1104 | 0.107329  |  0.0240117 |
| vehicle_plus_style_w0.50          | all_delays | neighbor_future_pairwise_rmse_mean | tail_rmse_v250 | 1104 | 0.254282  |  0.245246  |
| vehicle_plus_style_w0.50          | all_delays | neighbor_future_to_query_mean_rmse | tail_rmse_v250 | 1104 | 0.514955  |  0.468837  |
| vehicle_plus_style_w0.50          | all_delays | neighbor_input_distance_mean       | tail_rmse_v250 | 1104 | 0.0735041 |  0.0103509 |
| vehicle_plus_physio_recent_w0.25  | all_delays | neighbor_future_pairwise_rmse_mean | tail_rmse_v250 | 1104 | 0.273038  |  0.275409  |
| vehicle_plus_physio_recent_w0.25  | all_delays | neighbor_future_to_query_mean_rmse | tail_rmse_v250 | 1104 | 0.521036  |  0.474386  |
| vehicle_plus_physio_recent_w0.25  | all_delays | neighbor_input_distance_mean       | tail_rmse_v250 | 1104 | 0.135788  |  0.0947384 |
| vehicle_plus_physio_recent_w0.50  | all_delays | neighbor_future_pairwise_rmse_mean | tail_rmse_v250 | 1104 | 0.200552  |  0.219046  |
| vehicle_plus_physio_recent_w0.50  | all_delays | neighbor_future_to_query_mean_rmse | tail_rmse_v250 | 1104 | 0.450905  |  0.423268  |
| vehicle_plus_physio_recent_w0.50  | all_delays | neighbor_input_distance_mean       | tail_rmse_v250 | 1104 | 0.142712  |  0.130494  |
| vehicle_plus_physio_guarded_w0.50 | all_delays | neighbor_future_pairwise_rmse_mean | tail_rmse_v250 | 1104 | 0.198926  |  0.232173  |
| vehicle_plus_physio_guarded_w0.50 | all_delays | neighbor_future_to_query_mean_rmse | tail_rmse_v250 | 1104 | 0.475812  |  0.430405  |
| vehicle_plus_physio_guarded_w0.50 | all_delays | neighbor_input_distance_mean       | tail_rmse_v250 | 1104 | 0.135679  |  0.0547074 |
| vehicle_plus_style_physio_w0.50   | all_delays | neighbor_future_pairwise_rmse_mean | tail_rmse_v250 | 1104 | 0.201196  |  0.210935  |
| vehicle_plus_style_physio_w0.50   | all_delays | neighbor_future_to_query_mean_rmse | tail_rmse_v250 | 1104 | 0.45173   |  0.406815  |
| vehicle_plus_style_physio_w0.50   | all_delays | neighbor_input_distance_mean       | tail_rmse_v250 | 1104 | 0.0882656 |  0.105382  |

## 判读方式

- `delta_query_neighbor_vs_vehicle_only < 0`：加入状态信号后，同输入近邻的真实未来更接近 query，说明有消歧价值。
- 如果 delta 很小或为正，说明该状态信号当前表示没有帮助消歧。
- 如果 same subject / same recording rate 明显升高，需要警惕状态信号只是把近邻检索推向身份或 session 匹配。

## 关键图

- `figures\v253a_state_signal_badtop10_disambiguation.png`
- `figures\v253a_state_signal_delta_vs_vehicle_only.png`
