# v250 历史通道精简消融报告

## 本轮边界

- 只精简 `X_hist` 的 18 个车辆历史通道；历史长度仍为 -3.0s 到 0.0s，共 31 个时间点。
- 道路预瞄 `X_road`、phase/current 特征和 point query 不变。
- 每个精简通道组都从头训练 v241 TCN + multi-head query attention；不加载 v241 checkpoint。
- validation-only 选择；test 只做 locked report。
- 不做 anchor selector、gate/router、response-type hard routing，不删除样本。

## Validation 选择

- `v250_minimal_lateral7`: n_channels=7, score=0.6097, no_major_harm=True, hard_gain=True, accepted=True, best_epoch=19。
- `v250_lateral_core10`: n_channels=10, score=0.7478, no_major_harm=True, hard_gain=True, accepted=True, best_epoch=18。
- `v250_drop_attitude_noise13`: n_channels=13, score=0.7701, no_major_harm=True, hard_gain=True, accepted=True, best_epoch=17。

当前 best validation diagnostic model：`v250_minimal_lateral7`。

## Test 对照摘要

| bucket             |   delay_ms |   steer_tail_rmse_mean__v236_joint_ridge_existing |   steer_tail_rmse_mean__v241_tcn_mha_h96 |   steer_tail_rmse_mean__v250_drop_attitude_noise13 |   steer_tail_rmse_mean__v250_lateral_core10 |   steer_tail_rmse_mean__v250_minimal_lateral7 |   delta_steer_tail_rmse_mean__v250_drop_attitude_noise13_minus_v241 |   delta_steer_tail_rmse_mean__v250_lateral_core10_minus_v241 |   delta_steer_tail_rmse_mean__v250_minimal_lateral7_minus_v241 |
|:-------------------|-----------:|--------------------------------------------------:|-----------------------------------------:|---------------------------------------------------:|--------------------------------------------:|----------------------------------------------:|--------------------------------------------------------------------:|-------------------------------------------------------------:|---------------------------------------------------------------:|
| all                |          0 |                                          0.777846 |                                 0.475053 |                                           0.440619 |                                    0.447337 |                                      0.396888 |                                                          -0.0344344 |                                                  -0.0277165  |                                                     -0.0781656 |
| all                |        600 |                                          0.569821 |                                 0.38532  |                                           0.350571 |                                    0.368331 |                                      0.318    |                                                          -0.0347493 |                                                  -0.0169894  |                                                     -0.0673206 |
| all                |       1000 |                                          0.400838 |                                 0.304616 |                                           0.260139 |                                    0.281923 |                                      0.236594 |                                                          -0.0444762 |                                                  -0.0226927  |                                                     -0.0680217 |
| normal_predictable |          0 |                                          0.638458 |                                 0.371164 |                                           0.334542 |                                    0.344443 |                                      0.292139 |                                                          -0.0366219 |                                                  -0.0267209  |                                                     -0.0790247 |
| normal_predictable |        600 |                                          0.445911 |                                 0.297837 |                                           0.256646 |                                    0.292811 |                                      0.232909 |                                                          -0.0411901 |                                                  -0.00502512 |                                                     -0.0649279 |
| normal_predictable |       1000 |                                          0.317906 |                                 0.220595 |                                           0.190863 |                                    0.22349  |                                      0.177867 |                                                          -0.0297314 |                                                   0.00289516 |                                                     -0.0427276 |
| observe_later_like |          0 |                                          1.1004   |                                 0.792468 |                                           0.757546 |                                    0.709289 |                                      0.639918 |                                                          -0.0349224 |                                                  -0.0831795  |                                                     -0.15255   |
| observe_later_like |        600 |                                          0.960424 |                                 0.588433 |                                           0.601174 |                                    0.572924 |                                      0.511828 |                                                           0.0127417 |                                                  -0.0155087  |                                                     -0.0766044 |
| observe_later_like |       1000 |                                          0.744582 |                                 0.50421  |                                           0.4482   |                                    0.453981 |                                      0.375296 |                                                          -0.0560097 |                                                  -0.0502286  |                                                     -0.128914  |
| strong_steer       |          0 |                                          0.961224 |                                 0.590904 |                                           0.555114 |                                    0.555944 |                                      0.504266 |                                                          -0.03579   |                                                  -0.0349603  |                                                     -0.0866382 |
| strong_steer       |        600 |                                          0.69145  |                                 0.494278 |                                           0.467574 |                                    0.457476 |                                      0.422398 |                                                          -0.0267036 |                                                  -0.0368018  |                                                     -0.0718795 |
| strong_steer       |       1000 |                                          0.446245 |                                 0.405783 |                                           0.344479 |                                    0.35051  |                                      0.307555 |                                                          -0.0613033 |                                                  -0.0552723  |                                                     -0.0982276 |

## Shape 摘要

| event_group        |   delay_ms |   n |   mean_rmse |   mean_range_ratio |   mean_slope_ratio |   delta_rmse_candidate_minus_v241 |   delta_range_ratio_candidate_minus_v241 |   delta_slope_ratio_candidate_minus_v241 |
|:-------------------|-----------:|----:|------------:|-------------------:|-------------------:|----------------------------------:|-----------------------------------------:|-----------------------------------------:|
| all                |          0 | 184 |    0.340761 |           1.14547  |           0.823348 |                        -0.0573361 |                              -0.0855774  |                              -0.0145713  |
| all                |        600 | 184 |    0.288198 |           1.22415  |           0.906086 |                        -0.0604629 |                              -0.0577978  |                              -0.0172045  |
| all                |       1000 | 184 |    0.236594 |           1.29029  |           0.910792 |                        -0.0680216 |                              -0.344285   |                              -0.0148508  |
| normal             |          0 |  99 |    0.254916 |           1.38332  |           0.90021  |                        -0.0601449 |                              -0.22357    |                              -0.0857899  |
| normal             |        600 |  99 |    0.211812 |           1.52579  |           1.04606  |                        -0.0562934 |                              -0.164184   |                              -0.033596   |
| normal             |       1000 |  99 |    0.177867 |           1.72284  |           1.1177   |                        -0.0427276 |                              -0.700936   |                              -0.112426   |
| strong_steer       |          0 |  80 |    0.432051 |           0.855092 |           0.73105  |                        -0.0596733 |                               0.0778664  |                               0.0813607  |
| strong_steer       |        600 |  80 |    0.38221  |           0.87301  |           0.740108 |                        -0.067391  |                               0.06944    |                               0.00114956 |
| strong_steer       |       1000 |  80 |    0.307555 |           0.787355 |           0.658179 |                        -0.0982275 |                               0.0933607  |                               0.103132   |
| observe_later_like |          0 |  27 |    0.532047 |           0.76109  |           0.594242 |                        -0.100892  |                               0.0326462  |                               0.0333531  |
| observe_later_like |        600 |  27 |    0.46548  |           0.792113 |           0.643684 |                        -0.0682823 |                               0.0182036  |                              -0.0286375  |
| observe_later_like |       1000 |  27 |    0.375296 |           0.698279 |           0.60352  |                        -0.128914  |                               0.013238   |                               0.0822961  |
| bad_top10_v241     |          0 |  24 |    0.657487 |           0.801903 |           0.648705 |                        -0.208412  |                              -0.00778932 |                               0.0124293  |
| bad_top10_v241     |        600 |  18 |    0.667654 |           0.71708  |           0.658582 |                        -0.229208  |                               0.0477181  |                              -0.0275456  |
| bad_top10_v241     |       1000 |  13 |    0.509487 |           0.719714 |           0.611475 |                        -0.355224  |                               0.198819   |                               0.222572   |

## 输入邻域歧义

| model_name                 |   n_cases |   input_ambiguous_rate |   neighbor_future_pairwise_rmse_mean |   neighbor_peak_abs_std_mean |   neighbor_slope_abs_std_mean |   query_vs_neighbor_best_rmse_mean |
|:---------------------------|----------:|-----------------------:|-------------------------------------:|-----------------------------:|------------------------------:|-----------------------------------:|
| v250_lateral_core10        |        19 |                      1 |                             0.89063  |                     0.530606 |                      0.179377 |                           0.58819  |
| v250_drop_attitude_noise13 |        19 |                      1 |                             0.92474  |                     0.530196 |                      0.188453 |                           0.608534 |
| v250_minimal_lateral7      |        19 |                      1 |                             0.927086 |                     0.575736 |                      0.200561 |                           0.669133 |

## 下一步决策

| decision_item                            | decision                                                               | reason                                                                                                |
|:-----------------------------------------|:-----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| best_validation_channel_model            | v250_minimal_lateral7                                                  | Best by validation-only channel-ablation score; test was not used for selection.                      |
| accept_reduced_channel_as_next_candidate | True                                                                   | At least one reduced-channel candidate passed validation checks; still needs locked robustness audit. |
| accepted_model_name                      | v250_minimal_lateral7                                                  | Empty means v250 remains diagnostic only.                                                             |
| formal_replacement_allowed               | False                                                                  | v250 is an input-channel ablation; formal claim needs robustness and target-line consistency.         |
| input_ambiguity_note                     | Lowest neighbor future pairwise RMSE is 0.891 for v250_lateral_core10. | Use this to decide whether reduced channels actually make hard samples more distinguishable.          |
| recommended_next_task                    | v250_review_channel_ablation_or_try_multimodal_if_ambiguity_persists   | Do not tune on test; inspect validation + ambiguity evidence before changing model family.            |

## 关键图

- `figures\v250_tail_delta_by_channel_group.png`
- `figures\v250_neighbor_ambiguity_by_channel_group.png`

## 关键产物

- `tables/v250_model_selection_validation_channel_ablation.csv`
- `tables/v250_compare_vs_v241_original_remaining.csv`
- `tables/v250_shape_summary.csv`
- `tables/v250_input_neighborhood_ambiguity_by_channel.csv`
- `tables/v250_input_neighborhood_ambiguity_summary.csv`
- ZIP：`v250_history_channel_ablation_20260630_pack.zip`
