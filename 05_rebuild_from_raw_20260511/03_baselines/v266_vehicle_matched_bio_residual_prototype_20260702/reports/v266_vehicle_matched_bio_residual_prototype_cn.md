# v266 vehicle-matched bio residual prototype

## 本轮问题

- GPTPro phase02 的路线 1（wait-benefit / CATE-style）已经基本由 v265 覆盖，结果没有形成 bio 增量。
- v266 因此验证路线 2：在车辆历史相似的局部区域，train 事件的最佳残差/锚点模式是否能给 query 事件提供少量候选，bio260 是否能在这些候选内部重排序。
- 如果 candidate oracle 本身不能低于 fixed wait-latest `0.6950`，说明这个候选库没有足够 headroom；如果有 headroom 但 bio reranker 追不上，则问题在可部署选择信号。

## 方法边界

- prototype 只来自 train split；val/test 驾驶员历史完全不参与检索。
- query 只用 0ms 车辆上下文与 floor 0ms 的 bio260_sp64 特征。
- 生理不直接预测轨迹，只在 vehicle topK prototype 内部重排。
- K/lambda 只由 val bad_top10 选择；test 不调参。

## 特征与覆盖

|   event_n |   train_event_n |   val_event_n |   test_event_n |   vehicle_feature_n |   bio260_sp64_feature_n |   max_k |   bio260_uses_post_observation_max |
|----------:|----------------:|--------------:|---------------:|--------------------:|------------------------:|--------:|-----------------------------------:|
|      1167 |             674 |           309 |            184 |                  31 |                      65 |      40 |                                  0 |

## Test bad_top10 关键对照

| strategy                                          | strategy_family   | deployable   |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |   selected_latest_rate |   prototype_unique_delay_n_mean |
|:--------------------------------------------------|:------------------|:-------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|-------------------------:|-----------------------:|--------------------------------:|
| policy_keep_0ms_anchor                            | baseline          | True         |  19 |                  1.19771  |                          0        |                          0.502658  |                    0     |              0         |                         1       |
| policy_wait_to_latest_anchor                      | baseline          | True         |  19 |                  0.695048 |                         -0.502658 |                          0         |                 1000     |              1         |                         1       |
| oracle_best_anchor_upper_bound                    | oracle            | False        |  19 |                  0.612475 |                         -0.585231 |                         -0.082573  |                  818.421 |              0.368421  |                        21       |
| prototype_candidate_oracle_k40                    | candidate_oracle  | False        |  19 |                  0.616603 |                         -0.581103 |                         -0.0784456 |                  831.579 |              0.368421  |                        16.4737  |
| prototype_candidate_oracle_k20                    | candidate_oracle  | False        |  19 |                  0.625011 |                         -0.572695 |                         -0.0700371 |                  813.158 |              0.315789  |                        12.4211  |
| prototype_candidate_oracle_k10                    | candidate_oracle  | False        |  19 |                  0.642559 |                         -0.555148 |                         -0.0524897 |                  818.421 |              0.263158  |                         7.89474 |
| prototype_vehicle_vote_k40                        | vehicle_only      | True         |  19 |                  0.788777 |                         -0.40893  |                          0.0937282 |                  771.053 |              0.263158  |                        16.4737  |
| prototype_vehicle_vote_k20                        | vehicle_only      | True         |  19 |                  0.861457 |                         -0.336249 |                          0.166409  |                  544.737 |              0.157895  |                        12.4211  |
| prototype_vehicle_nearest                         | vehicle_only      | True         |  19 |                  0.878536 |                         -0.31917  |                          0.183488  |                  521.053 |              0.157895  |                         1       |
| prototype_bio_closest_k40                         | vehicle_bio       | True         |  19 |                  0.798872 |                         -0.398834 |                          0.103824  |                  642.105 |              0.0526316 |                        16.4737  |
| prototype_bio_closest_k20                         | vehicle_bio       | True         |  19 |                  0.807803 |                         -0.389904 |                          0.112754  |                  665.789 |              0.105263  |                        12.4211  |
| prototype_vehicle_bio_k20_lam0.50                 | vehicle_bio       | True         |  19 |                  0.828984 |                         -0.368722 |                          0.133936  |                  589.474 |              0.105263  |                        12.4211  |
| val_best_vehicle_only: prototype_vehicle_vote_k10 | vehicle_only      | True         |  19 |                  0.888994 |                         -0.308712 |                          0.193946  |                  481.579 |              0.157895  |                       nan       |
| val_best_vehicle_bio: prototype_bio_closest_k3    | vehicle_bio       | True         |  19 |                  0.837371 |                         -0.360335 |                          0.142323  |                  560.526 |              0.0526316 |                       nan       |

## Val 选择的可部署策略

| chosen_label          | chosen_strategy            | chosen_family   | split   | event_group   |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |
|:----------------------|:---------------------------|:----------------|:--------|:--------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|-------------------------:|
| val_best_vehicle_only | prototype_vehicle_vote_k10 | vehicle_only    | train   | all           | 674 |                  0.126783 |                        -0.0165511 |                         0.00179063 |                  608.086 |
| val_best_vehicle_only | prototype_vehicle_vote_k10 | vehicle_only    | train   | bad_top10     |  68 |                  0.184827 |                        -0.0948876 |                        -0.00193863 |                  625     |
| val_best_vehicle_only | prototype_vehicle_vote_k10 | vehicle_only    | val     | all           | 309 |                  0.57563  |                        -0.167676  |                         0.0977709  |                  606.311 |
| val_best_vehicle_only | prototype_vehicle_vote_k10 | vehicle_only    | val     | bad_top10     |  31 |                  1.57775  |                        -0.593817  |                         0.504963   |                  496.774 |
| val_best_vehicle_only | prototype_vehicle_vote_k10 | vehicle_only    | test    | all           | 184 |                  0.37453  |                        -0.100523  |                         0.0699146  |                  598.37  |
| val_best_vehicle_only | prototype_vehicle_vote_k10 | vehicle_only    | test    | bad_top10     |  19 |                  0.888994 |                        -0.308712  |                         0.193946   |                  481.579 |
| val_best_vehicle_bio  | prototype_bio_closest_k3   | vehicle_bio     | train   | all           | 674 |                  0.127771 |                        -0.0155623 |                         0.00277936 |                  539.837 |
| val_best_vehicle_bio  | prototype_bio_closest_k3   | vehicle_bio     | train   | bad_top10     |  68 |                  0.205619 |                        -0.0740957 |                         0.0188533  |                  532.353 |
| val_best_vehicle_bio  | prototype_bio_closest_k3   | vehicle_bio     | val     | all           | 309 |                  0.581508 |                        -0.161799  |                         0.103649   |                  583.657 |
| val_best_vehicle_bio  | prototype_bio_closest_k3   | vehicle_bio     | val     | bad_top10     |  31 |                  1.53889  |                        -0.632679  |                         0.466102   |                  508.065 |
| val_best_vehicle_bio  | prototype_bio_closest_k3   | vehicle_bio     | test    | all           | 184 |                  0.375562 |                        -0.0994916 |                         0.0709461  |                  551.63  |
| val_best_vehicle_bio  | prototype_bio_closest_k3   | vehicle_bio     | test    | bad_top10     |  19 |                  0.837371 |                        -0.360335  |                         0.142323   |                  560.526 |

## 判读

- vehicle-matched candidate oracle 最好为 `0.6166` (prototype_candidate_oracle_k40)；fixed wait-latest 是 `0.6950`。
- 这说明相似车辆 prototype 候选库理论上有一点 headroom，值得继续看可部署 reranker。
- val 选出的 vehicle-only prototype 在 test bad_top10 为 `0.8890`；val 选出的 vehicle+bio prototype 为 `0.8374`。
- bio 在可部署重排上比 vehicle-only 低 `0.0516`。
- vehicle+bio 未低于 fixed wait-latest，不能算差样本本质性改善。

## 关键图

- `figures\v266_test_badtop10_main_comparison.png`
- `figures\v266_candidate_oracle_headroom_by_k.png`

## 输入合并审计

|   candidate_rows |   event_n |   bio260_source_rows |   bio260_source_event_n |   bio260_feature_n |   bio260_merge_ok_rate |   bio260_feature_missing_rate_after_merge |   bio260_uses_post_observation_max |   candidate_delay_min |   candidate_delay_max |
|-----------------:|----------:|---------------------:|------------------------:|-------------------:|-----------------------:|------------------------------------------:|-----------------------------------:|----------------------:|----------------------:|
|            24507 |      1167 |                 7002 |                    1167 |                233 |               0.919452 |                                  0.256043 |                                  0 |                     0 |                  1000 |
