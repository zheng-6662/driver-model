# v253b 生理/驾驶风格状态 tie-break 审计

## 本轮问题

本轮不再把生理/风格简单拼进全局输入距离，而是先用 vehicle-only 找同 delay 的车辆相似候选池，再看生理/风格能否在这个池内挑出未来更接近 query 的样本。

## 关键边界

- 不训练预测模型，不修改 v250/v252/v253a。
- 当前 split 是 subject-disjoint：test 被试不在 train 中，因此不能验证同一驾驶员个体记忆，只能验证跨驾驶员状态相似性。
- 未来 RMSE 只用于诊断评价 tie-break 是否挑对，不作为部署输入。

## Subject Split

| subject   |   test |   train |   val |
|:----------|-------:|--------:|------:|
| byx       |      0 |     612 |     0 |
| cwh       |    276 |       0 |     0 |
| gf        |      0 |     216 |     0 |
| gzj       |      0 |       0 |   630 |
| hzh       |      0 |     708 |     0 |
| jy        |      0 |     252 |     0 |
| lx        |     78 |       0 |     0 |
| lxy       |      0 |       0 |   390 |
| rjy       |    492 |       0 |     0 |
| txj       |      0 |       0 |   546 |
| tyy       |    258 |       0 |     0 |
| xst       |      0 |      36 |     0 |
| yyl       |      0 |     522 |     0 |
| yzy       |      0 |     474 |     0 |
| zdq       |      0 |       0 |   288 |
| zt        |      0 |      90 |     0 |
| zx        |      0 |     918 |     0 |
| zxy       |      0 |     216 |     0 |

## 关键结果

| bucket             | delay_ms   | strategy                               |    n |   selected_future_rmse_mean |   delta_selected_minus_vehicle_mean |   improve_rate_vs_vehicle_rank1 |   selected_neighbor_vehicle_rank_mean |
|:-------------------|:-----------|:---------------------------------------|-----:|----------------------------:|------------------------------------:|--------------------------------:|--------------------------------------:|
| all                | all_delays | vehicle_rank1                          | 1104 |                    0.581431 |                           0         |                        0        |                                1      |
| all                | all_delays | style_nearest_in_vehicle_pool          | 1104 |                    0.775657 |                           0.194226  |                        0.344203 |                               27.1612 |
| all                | all_delays | physio_recent_nearest_in_vehicle_pool  | 1104 |                    0.789481 |                           0.20805   |                        0.35779  |                               28.5969 |
| all                | all_delays | physio_guarded_nearest_in_vehicle_pool | 1104 |                    0.803351 |                           0.22192   |                        0.315217 |                               28.2645 |
| all                | all_delays | style_physio_nearest_in_vehicle_pool   | 1104 |                    0.795264 |                           0.213833  |                        0.351449 |                               29.3895 |
| all                | all_delays | oracle_best_future_in_vehicle_pool     | 1104 |                    0.184621 |                          -0.39681   |                        0.962862 |                               25.1422 |
| all                | 0          | vehicle_rank1                          |  184 |                    0.656528 |                           0         |                        0        |                                1      |
| all                | 0          | style_nearest_in_vehicle_pool          |  184 |                    0.921527 |                           0.264999  |                        0.353261 |                               27.8261 |
| all                | 0          | physio_recent_nearest_in_vehicle_pool  |  184 |                    0.941001 |                           0.284473  |                        0.293478 |                               29.7717 |
| all                | 0          | physio_guarded_nearest_in_vehicle_pool |  184 |                    0.916356 |                           0.259828  |                        0.282609 |                               29.288  |
| all                | 0          | style_physio_nearest_in_vehicle_pool   |  184 |                    0.958091 |                           0.301563  |                        0.309783 |                               30.7446 |
| all                | 0          | oracle_best_future_in_vehicle_pool     |  184 |                    0.239365 |                          -0.417163  |                        0.951087 |                               22.288  |
| bad_top10_v250     | all_delays | vehicle_rank1                          |  111 |                    0.993414 |                           0         |                        0        |                                1      |
| bad_top10_v250     | all_delays | style_nearest_in_vehicle_pool          |  111 |                    1.10434  |                           0.110927  |                        0.369369 |                               25.8649 |
| bad_top10_v250     | all_delays | physio_recent_nearest_in_vehicle_pool  |  111 |                    1.10764  |                           0.11423   |                        0.468468 |                               29.3874 |
| bad_top10_v250     | all_delays | physio_guarded_nearest_in_vehicle_pool |  111 |                    1.19295  |                           0.199539  |                        0.333333 |                               33.9279 |
| bad_top10_v250     | all_delays | style_physio_nearest_in_vehicle_pool   |  111 |                    1.27327  |                           0.279856  |                        0.414414 |                               30.3694 |
| bad_top10_v250     | all_delays | oracle_best_future_in_vehicle_pool     |  111 |                    0.367753 |                          -0.625662  |                        0.972973 |                               27.4505 |
| bad_top10_v250     | 0          | vehicle_rank1                          |   30 |                    1.06663  |                           0         |                        0        |                                1      |
| bad_top10_v250     | 0          | style_nearest_in_vehicle_pool          |   30 |                    1.0418   |                          -0.0248301 |                        0.4      |                               27.7667 |
| bad_top10_v250     | 0          | physio_recent_nearest_in_vehicle_pool  |   30 |                    1.12103  |                           0.0543943 |                        0.433333 |                               29.0667 |
| bad_top10_v250     | 0          | physio_guarded_nearest_in_vehicle_pool |   30 |                    1.30187  |                           0.235235  |                        0.266667 |                               32.4    |
| bad_top10_v250     | 0          | style_physio_nearest_in_vehicle_pool   |   30 |                    1.38352  |                           0.316888  |                        0.433333 |                               33.3333 |
| bad_top10_v250     | 0          | oracle_best_future_in_vehicle_pool     |   30 |                    0.405899 |                          -0.660732  |                        1        |                               30.6333 |
| strong_steer       | all_delays | vehicle_rank1                          |  480 |                    0.692386 |                           0         |                        0        |                                1      |
| strong_steer       | all_delays | style_nearest_in_vehicle_pool          |  480 |                    0.962411 |                           0.270024  |                        0.285417 |                               28.5354 |
| strong_steer       | all_delays | physio_recent_nearest_in_vehicle_pool  |  480 |                    0.942905 |                           0.250519  |                        0.360417 |                               28.3333 |
| strong_steer       | all_delays | physio_guarded_nearest_in_vehicle_pool |  480 |                    0.976856 |                           0.284469  |                        0.272917 |                               28.2667 |
| strong_steer       | all_delays | style_physio_nearest_in_vehicle_pool   |  480 |                    0.996094 |                           0.303707  |                        0.327083 |                               31.3896 |
| strong_steer       | all_delays | oracle_best_future_in_vehicle_pool     |  480 |                    0.252303 |                          -0.440084  |                        0.95625  |                               24.1562 |
| strong_steer       | 0          | vehicle_rank1                          |   80 |                    0.805673 |                           0         |                        0        |                                1      |
| strong_steer       | 0          | style_nearest_in_vehicle_pool          |   80 |                    1.17776  |                           0.372091  |                        0.3      |                               30.2125 |
| strong_steer       | 0          | physio_recent_nearest_in_vehicle_pool  |   80 |                    1.13168  |                           0.326003  |                        0.3125   |                               28.4875 |
| strong_steer       | 0          | physio_guarded_nearest_in_vehicle_pool |   80 |                    1.12396  |                           0.318291  |                        0.2375   |                               31.275  |
| strong_steer       | 0          | style_physio_nearest_in_vehicle_pool   |   80 |                    1.21894  |                           0.413263  |                        0.275    |                               32.3    |
| strong_steer       | 0          | oracle_best_future_in_vehicle_pool     |   80 |                    0.326444 |                          -0.479229  |                        0.95     |                               23.275  |
| observe_later_like | all_delays | vehicle_rank1                          |  162 |                    0.846188 |                           0         |                        0        |                                1      |
| observe_later_like | all_delays | style_nearest_in_vehicle_pool          |  162 |                    1.07262  |                           0.226436  |                        0.277778 |                               29.3148 |
| observe_later_like | all_delays | physio_recent_nearest_in_vehicle_pool  |  162 |                    0.858254 |                           0.0120666 |                        0.567901 |                               26.8951 |
| observe_later_like | all_delays | physio_guarded_nearest_in_vehicle_pool |  162 |                    1.00365  |                           0.157461  |                        0.407407 |                               29.9259 |
| observe_later_like | all_delays | style_physio_nearest_in_vehicle_pool   |  162 |                    1.04948  |                           0.203297  |                        0.388889 |                               31.3333 |
| observe_later_like | all_delays | oracle_best_future_in_vehicle_pool     |  162 |                    0.31283  |                          -0.533358  |                        0.969136 |                               23.5432 |
| observe_later_like | 0          | vehicle_rank1                          |   27 |                    0.825921 |                           0         |                        0        |                                1      |
| observe_later_like | 0          | style_nearest_in_vehicle_pool          |   27 |                    1.23029  |                           0.404374  |                        0.259259 |                               27.6667 |
| observe_later_like | 0          | physio_recent_nearest_in_vehicle_pool  |   27 |                    0.978774 |                           0.152853  |                        0.444444 |                               24.4074 |
| observe_later_like | 0          | physio_guarded_nearest_in_vehicle_pool |   27 |                    1.13089  |                           0.304965  |                        0.296296 |                               30.0741 |
| observe_later_like | 0          | style_physio_nearest_in_vehicle_pool   |   27 |                    1.31551  |                           0.489587  |                        0.296296 |                               37      |
| observe_later_like | 0          | oracle_best_future_in_vehicle_pool     |   27 |                    0.383459 |                          -0.442461  |                        0.962963 |                               23.5926 |

## 候选池内距离-未来误差相关

| bucket             | distance_block   |   n_query |   mean_spearman_distance_vs_future_rmse |   median_spearman_distance_vs_future_rmse |   positive_rate |
|:-------------------|:-----------------|----------:|----------------------------------------:|------------------------------------------:|----------------:|
| all                | physio_guard     |      1104 |                            -0.0197923   |                               -0.0208805  |        0.444746 |
| all                | physio_recent    |      1104 |                             0.0205976   |                                0.0205647  |        0.549819 |
| all                | style            |      1104 |                             0.0390136   |                                0.0357599  |        0.598732 |
| all                | style_physio     |      1104 |                             0.0364387   |                                0.0329536  |        0.580616 |
| all                | vehicle          |      1104 |                             0.179498    |                                0.180856   |        0.82337  |
| bad_top10_v250     | physio_guard     |       111 |                            -0.0215032   |                               -0.0310172  |        0.369369 |
| bad_top10_v250     | physio_recent    |       111 |                            -0.00803183  |                               -0.0245136  |        0.45045  |
| bad_top10_v250     | style            |       111 |                             0.0261424   |                                0.0333426  |        0.585586 |
| bad_top10_v250     | style_physio     |       111 |                            -0.00233849  |                               -0.0192831  |        0.468468 |
| bad_top10_v250     | vehicle          |       111 |                             0.151956    |                                0.141539   |        0.702703 |
| strong_steer       | physio_guard     |       480 |                            -0.012047    |                               -0.0209005  |        0.45     |
| strong_steer       | physio_recent    |       480 |                             0.00670549  |                                0.007783   |        0.51875  |
| strong_steer       | style            |       480 |                             0.0158299   |                                0.0174493  |        0.545833 |
| strong_steer       | style_physio     |       480 |                             0.00668612  |                                0.00166713 |        0.504167 |
| strong_steer       | vehicle          |       480 |                             0.178637    |                                0.186441   |        0.833333 |
| observe_later_like | physio_guard     |       162 |                            -0.0131092   |                               -0.0276949  |        0.41358  |
| observe_later_like | physio_recent    |       162 |                            -0.0042852   |                               -0.0232374  |        0.475309 |
| observe_later_like | style            |       162 |                             0.000999935 |                                0.0147263  |        0.537037 |
| observe_later_like | style_physio     |       162 |                            -0.0204885   |                               -0.0216449  |        0.475309 |
| observe_later_like | vehicle          |       162 |                             0.216365    |                                0.213309   |        0.858025 |

## 判读

- 如果生理/风格能提供驾驶员状态区别，应该看到 tie-break 策略的 `delta_selected_minus_vehicle_mean < 0`，尤其在 bad_top10_v250 上。
- 如果距离-未来误差相关为正，说明生理/风格距离越近，未来也越近；若接近 0 或负值，则当前状态表示没有提供有效排序。
- oracle 只表示车辆相似候选池内还存在更好未来上限，不代表可部署。

## 关键图

- `figures\v253b_badtop10_tiebreak_selected_future_rmse.png`
- `figures\v253b_tiebreak_delta_vs_vehicle_rank1.png`
