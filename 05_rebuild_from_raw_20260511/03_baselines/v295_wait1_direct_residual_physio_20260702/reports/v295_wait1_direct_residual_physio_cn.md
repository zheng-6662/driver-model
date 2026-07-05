# v295 wait1 direct residual + physiology 审计

## 结论
- v295 没达到“本质性改善差样本”的标准；它更像是一个方向性探针。
- 当前 best_val_physio_deployable test bad_top10 delta = -0.001106，test all delta = 0.010394。
- 负 delta 表示比 v249 wait1 rolling baseline 更好；正 delta 表示变差。

## 方法
- baseline: v249 `delay_ms=1000` 的 rolling 预测。
- target: `y_true - baseline` 的 wait1 残差曲线，只在共同有效 horizon 点上训练和评估。
- inputs: baseline 曲线形态、原锚点后 0-1s 已观测车辆响应、v293 `post0_1` 生理特征、subject one-hot。
- selection: residual 模型只用 train；风险 gate 阈值只用 val；test 不参与筛选。

## 数据口径
- event_n=1167, eval_point_n=11, eval_grid=[0.0, 0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5, 0.6000000238418579, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0], target_cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].
| split   |   n |
|:--------|----:|
| test    | 184 |
| train   | 674 |
| val     | 309 |

## chosen selectors
| choice_name                         | feature_block                                  | model_name   |   shrinkage | risk_tag                                         |   threshold |   risk_val_auc |   risk_test_auc |   val_all_delta_vs_baseline_mean |   val_bad_top10_delta_vs_baseline_mean |   test_all_delta_vs_baseline_mean |   test_bad_top10_delta_vs_baseline_mean |   test_bad_top10_vehicle_ambiguous_delta_vs_baseline_mean |   test_bad_top10_override_rate |
|:------------------------------------|:-----------------------------------------------|:-------------|------------:|:-------------------------------------------------|------------:|---------------:|----------------:|---------------------------------:|---------------------------------------:|----------------------------------:|----------------------------------------:|----------------------------------------------------------:|-------------------------------:|
| fallback_no_correction              | base_curve_only                                | ridge_a10    |        0.25 | no_override                                      |  inf        |                |                 |                        -0        |                              -0        |                         -0        |                                0        |                                                  0        |                       0        |
| best_val_overall_deployable         | base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |    0.357928 |       0.528777 |         0.35949 |                         0.000397 |                              -0.008889 |                          0.010394 |                               -0.001106 |                                                 -0.001401 |                       0.210526 |
| best_val_physio_deployable          | base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |    0.357928 |       0.528777 |         0.35949 |                         0.000397 |                              -0.008889 |                          0.010394 |                               -0.001106 |                                                 -0.001401 |                       0.210526 |
| best_val_nonphysio_ablation         | base_curve_only                                | ridge_a10    |        1    | always_apply                                     | -inf        |                |                 |                        -0.002813 |                              -0.005556 |                         -0.00508  |                               -0.010736 |                                                 -0.011401 |                       1        |
| test_best_diagnostic_not_deployable | base_plus_post01_physio_subject                | ridge_a10    |        1    | always_apply                                     | -inf        |                |                 |                         0.003777 |                              -0.003817 |                          0.019167 |                               -0.024859 |                                                 -0.03124  |                       1        |

## risk classifier audit
| risk_tag                                                   |   val_auc |   test_auc |   feature_n |
|:-----------------------------------------------------------|----------:|-----------:|------------:|
| base_post01_physio_subject__extra_trees_cls_d3             |  0.550476 |   0.651675 |         141 |
| post01_physio_subject__extra_trees_cls_d3                  |  0.610118 |   0.567145 |         119 |
| vehicle_prefix_post01_physio_subject__extra_trees_cls_d3   |  0.708285 |   0.517065 |         141 |
| post01_physio_subject__logreg_balanced_c025                |  0.55094  |   0.435726 |         119 |
| vehicle_prefix_post01_physio_subject__logreg_balanced_c025 |  0.577048 |   0.406699 |         141 |
| base_post01_physio_subject__logreg_balanced_c025           |  0.528777 |   0.35949  |         141 |

## top validation candidates
| feature_block                                  | model_name   |   shrinkage | risk_tag                                         |   risk_val_auc |   val_all_delta_vs_baseline_mean |   val_bad_top10_delta_vs_baseline_mean |   test_all_delta_vs_baseline_mean |   test_bad_top10_delta_vs_baseline_mean |
|:-----------------------------------------------|:-------------|------------:|:-------------------------------------------------|---------------:|---------------------------------:|---------------------------------------:|----------------------------------:|----------------------------------------:|
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000397 |                              -0.008889 |                          0.010394 |                               -0.001106 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000669 |                              -0.008395 |                          0.009778 |                               -0.001033 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.001593 |                              -0.007532 |                          0.012783 |                               -0.002778 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        0.75 | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -0.000259 |                              -0.007257 |                          0.006225 |                               -0.001019 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.001133 |                              -0.007187 |                          0.011771 |                               -0.001106 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000576 |                              -0.007187 |                          0.010974 |                               -0.001106 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000853 |                              -0.007187 |                          0.012501 |                               -0.002778 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.001048 |                              -0.007187 |                          0.011867 |                               -0.002778 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.001121 |                              -0.007187 |                          0.011486 |                               -0.001106 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000454 |                              -0.007142 |                          0.008364 |                               -0.001033 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        0.75 | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -3.3e-05  |                              -0.006845 |                          0.005822 |                               -0.000923 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000919 |                              -0.006843 |                          0.008867 |                               -0.001033 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000269 |                              -0.006843 |                          0.008841 |                               -0.001033 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000375 |                              -0.006843 |                          0.008928 |                               -0.001033 |
| base_plus_observed_vehicle_prefix              | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -0.001498 |                              -0.006519 |                         -0.001355 |                               -0.000593 |
| base_plus_observed_vehicle_prefix              | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -0.001912 |                              -0.006482 |                         -0.000841 |                                0.000613 |
| base_plus_observed_vehicle_prefix              | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -0.001699 |                              -0.006482 |                         -0.000924 |                                0.000613 |
| base_plus_observed_vehicle_prefix              | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -0.001881 |                              -0.006482 |                         -0.000996 |                                0.000613 |
| base_plus_observed_vehicle_prefix              | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -0.001897 |                              -0.006482 |                         -0.001316 |                               -0.000593 |
| base_plus_observed_vehicle_prefix              | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -0.002    |                              -0.006482 |                         -0.001464 |                               -0.000593 |
| base_plus_observed_vehicle_prefix              | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -0.001737 |                              -0.006451 |                         -0.000874 |                                0.000613 |
| base_plus_post01_physio_subject                | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000569 |                              -0.006438 |                          0.010439 |                               -0.000873 |
| base_plus_observed_vehicle_prefix              | ridge_a10    |        1    | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -0.001382 |                              -0.006349 |                         -0.000658 |                                0.001096 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        0.75 | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000452 |                              -0.006286 |                          0.007585 |                               -0.002319 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        1    | post01_physio_subject__logreg_balanced_c025      |       0.55094  |                         0.000407 |                              -0.006111 |                          0.012506 |                               -0.009128 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        0.75 | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -0.000171 |                              -0.006018 |                          0.006591 |                               -0.001019 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        0.75 | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                        -7.9e-05  |                              -0.006018 |                          0.007395 |                               -0.002319 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        0.75 | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         9e-05    |                              -0.006018 |                          0.006956 |                               -0.002319 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        0.75 | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000221 |                              -0.006018 |                          0.006929 |                               -0.001019 |
| base_plus_vehicle_prefix_post01_physio_subject | ridge_a10    |        0.75 | base_post01_physio_subject__logreg_balanced_c025 |       0.528777 |                         0.000206 |                              -0.006018 |                          0.007028 |                               -0.001019 |

## guardrail
```json
{
  "pass": true,
  "event_n": 1167,
  "wait_ms": 1000,
  "wait_s": 1.0,
  "eval_point_n": 11,
  "eval_grid_s": [
    0.0,
    0.10000000149011612,
    0.20000000298023224,
    0.30000001192092896,
    0.4000000059604645,
    0.5,
    0.6000000238418579,
    0.699999988079071,
    0.800000011920929,
    0.8999999761581421,
    1.0
  ],
  "baseline": "v249_shape_conditioned_residual_delay1000",
  "uses_post_observation": true,
  "post_features_are_wait_policy_only": true,
  "post_window_used": "v293_post0_1",
  "test_used_for_feature_screen_model_or_threshold": false,
  "chosen_physio_exists": true,
  "chosen_overall_exists": true,
  "best_physio_test_badtop10_delta": -0.001105991050377485,
  "best_physio_test_all_delta": 0.010394378433541778,
  "best_nonphysio_test_badtop10_delta": -0.010735767569183076,
  "physio_increment_vs_nonphysio_badtop10_delta": 0.009629776518805592,
  "best_overall_test_badtop10_delta": -0.001105991050377485,
  "best_diagnostic_test_badtop10_delta": -0.02485921308334546,
  "best_risk_test_auc": 0.6516746411483253,
  "route_viable_now": false,
  "weak_physio_residual_signal_exists": false,
  "goal_achieved_now": false,
  "requirement_for_route_viable": "physio deployable must improve test bad_top10 by at least 0.05 RMSE with test all delta <=0.005 and val no-harm"
}
```