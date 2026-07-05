# v255 生理状态条件化候选轨迹选择实验

## 本轮问题

- v254b 说明：200Hz 生理直接拼接到车辆输入后，正式 subject-disjoint 轨迹行为诊断没有增量。
- v253b 说明：车辆相似候选池内 oracle 上限很高，但简单生理最近邻没有选中好未来。
- v255 因此改成学习式候选重排序：车辆先给候选池，生理状态只负责在候选未来原型中参与选择。

## 方法边界

- 不使用 query 的未来作为部署输入；未来 RMSE 只用于训练 pair 监督和离线评价。
- 候选未来摘要来自训练库候选样本，因为 retrieval 预测本身就是从训练库选未来原型。
- 不做删样本、不做 residual 修正、不做 v222a 式 gate；这里是候选轨迹选择模型。
- subject-disjoint 是正式泛化口径；subject-aware 只表示同一驾驶员有历史样本时的个体化潜力。

## 特征与阈值

| protocol         | model                              |   n_features |   chosen_threshold |
|:-----------------|:-----------------------------------|-------------:|-------------------:|
| subject_disjoint | learned_vehicle_context_guarded    |           10 |              1e+09 |
| subject_disjoint | learned_physio_state_guarded       |           28 |              1e+09 |
| subject_disjoint | learned_physio_badweighted_guarded |           28 |              1e+09 |
| subject_aware    | learned_vehicle_context_guarded    |           10 |              1e+09 |
| subject_aware    | learned_physio_state_guarded       |           28 |              1e+09 |
| subject_aware    | learned_physio_badweighted_guarded |           28 |              1e+09 |

## Test 关键结果

| protocol         | bucket             | strategy                           |    n |   selected_future_rmse_mean |   delta_selected_minus_vehicle_mean |   improve_rate_vs_vehicle_rank1 |   selected_neighbor_vehicle_rank_mean |
|:-----------------|:-------------------|:-----------------------------------|-----:|----------------------------:|------------------------------------:|--------------------------------:|--------------------------------------:|
| subject_disjoint | all                | vehicle_rank1                      | 1104 |                    0.581431 |                            0        |                        0        |                                1      |
| subject_disjoint | all                | learned_vehicle_context_guarded    | 1104 |                    0.581431 |                            0        |                        0        |                                1      |
| subject_disjoint | all                | learned_physio_state_guarded       | 1104 |                    0.581431 |                            0        |                        0        |                                1      |
| subject_disjoint | all                | learned_physio_badweighted_guarded | 1104 |                    0.581431 |                            0        |                        0        |                                1      |
| subject_disjoint | all                | oracle_best_future                 | 1104 |                    0.184621 |                           -0.39681  |                        0.962862 |                               25.1422 |
| subject_disjoint | bad_top10_v250     | vehicle_rank1                      |  111 |                    0.993414 |                            0        |                        0        |                                1      |
| subject_disjoint | bad_top10_v250     | learned_vehicle_context_guarded    |  111 |                    0.993414 |                            0        |                        0        |                                1      |
| subject_disjoint | bad_top10_v250     | learned_physio_state_guarded       |  111 |                    0.993414 |                            0        |                        0        |                                1      |
| subject_disjoint | bad_top10_v250     | learned_physio_badweighted_guarded |  111 |                    0.993414 |                            0        |                        0        |                                1      |
| subject_disjoint | bad_top10_v250     | oracle_best_future                 |  111 |                    0.367753 |                           -0.625662 |                        0.972973 |                               27.4505 |
| subject_disjoint | strong_steer       | vehicle_rank1                      |  480 |                    0.692386 |                            0        |                        0        |                                1      |
| subject_disjoint | strong_steer       | learned_vehicle_context_guarded    |  480 |                    0.692386 |                            0        |                        0        |                                1      |
| subject_disjoint | strong_steer       | learned_physio_state_guarded       |  480 |                    0.692386 |                            0        |                        0        |                                1      |
| subject_disjoint | strong_steer       | learned_physio_badweighted_guarded |  480 |                    0.692386 |                            0        |                        0        |                                1      |
| subject_disjoint | strong_steer       | oracle_best_future                 |  480 |                    0.252303 |                           -0.440084 |                        0.95625  |                               24.1562 |
| subject_disjoint | observe_later_like | vehicle_rank1                      |  162 |                    0.846188 |                            0        |                        0        |                                1      |
| subject_disjoint | observe_later_like | learned_vehicle_context_guarded    |  162 |                    0.846188 |                            0        |                        0        |                                1      |
| subject_disjoint | observe_later_like | learned_physio_state_guarded       |  162 |                    0.846188 |                            0        |                        0        |                                1      |
| subject_disjoint | observe_later_like | learned_physio_badweighted_guarded |  162 |                    0.846188 |                            0        |                        0        |                                1      |
| subject_disjoint | observe_later_like | oracle_best_future                 |  162 |                    0.31283  |                           -0.533358 |                        0.969136 |                               23.5432 |
| subject_aware    | all                | vehicle_rank1                      | 1398 |                    0.651627 |                            0        |                        0        |                                1      |
| subject_aware    | all                | learned_vehicle_context_guarded    | 1398 |                    0.651627 |                            0        |                        0        |                                1      |
| subject_aware    | all                | learned_physio_state_guarded       | 1398 |                    0.651627 |                            0        |                        0        |                                1      |
| subject_aware    | all                | learned_physio_badweighted_guarded | 1398 |                    0.651627 |                            0        |                        0        |                                1      |
| subject_aware    | all                | oracle_best_future                 | 1398 |                    0.22852  |                           -0.423107 |                        0.95422  |                               25.0594 |
| subject_aware    | bad_top10_v250     | vehicle_rank1                      |  140 |                    0.983806 |                            0        |                        0        |                                1      |
| subject_aware    | bad_top10_v250     | learned_vehicle_context_guarded    |  140 |                    0.983806 |                            0        |                        0        |                                1      |
| subject_aware    | bad_top10_v250     | learned_physio_state_guarded       |  140 |                    0.983806 |                            0        |                        0        |                                1      |
| subject_aware    | bad_top10_v250     | learned_physio_badweighted_guarded |  140 |                    0.983806 |                            0        |                        0        |                                1      |
| subject_aware    | bad_top10_v250     | oracle_best_future                 |  140 |                    0.440309 |                           -0.543498 |                        0.964286 |                               27.6643 |
| subject_aware    | strong_steer       | vehicle_rank1                      |  756 |                    0.779096 |                            0        |                        0        |                                1      |
| subject_aware    | strong_steer       | learned_vehicle_context_guarded    |  756 |                    0.779096 |                            0        |                        0        |                                1      |
| subject_aware    | strong_steer       | learned_physio_state_guarded       |  756 |                    0.779096 |                            0        |                        0        |                                1      |
| subject_aware    | strong_steer       | learned_physio_badweighted_guarded |  756 |                    0.779096 |                            0        |                        0        |                                1      |
| subject_aware    | strong_steer       | oracle_best_future                 |  756 |                    0.30761  |                           -0.471486 |                        0.952381 |                               25.3995 |
| subject_aware    | observe_later_like | vehicle_rank1                      |  174 |                    0.743502 |                            0        |                        0        |                                1      |
| subject_aware    | observe_later_like | learned_vehicle_context_guarded    |  174 |                    0.743502 |                            0        |                        0        |                                1      |
| subject_aware    | observe_later_like | learned_physio_state_guarded       |  174 |                    0.743502 |                            0        |                        0        |                                1      |
| subject_aware    | observe_later_like | learned_physio_badweighted_guarded |  174 |                    0.743502 |                            0        |                        0        |                                1      |
| subject_aware    | observe_later_like | oracle_best_future                 |  174 |                    0.295103 |                           -0.448399 |                        0.91954  |                               19.5575 |

## 关键判读

- subject_disjoint / bad_top10：vehicle rank1=0.9934，最佳非 oracle=learned_physio_state_guarded 0.9934 (delta=+0.0000)，oracle=0.3678。
- subject_aware / bad_top10：vehicle rank1=0.9838，最佳非 oracle=learned_physio_state_guarded 0.9838 (delta=+0.0000)，oracle=0.4403。
- 如果 learned_physio_state 明显优于 learned_vehicle_context，才说明生理状态真正提供了候选选择增量。
- 如果只在 subject-aware 改善，说明生理更适合作为个体化校准信号；若 subject-disjoint 仍无改善，就不能把它宣称为跨驾驶员通用行为信息。

## 关键图

- `figures\v255_badtop10_candidate_selection_rmse.png`
- `figures\v255_test_delta_vs_vehicle_rank1.png`