# v298 event label explanatory audit

## 结论
- v298 的核心问题不是继续堆模型，而是检查“事件/响应标签”这条线有没有上限价值。
- 当前 future response label-known 修正对 test bad_top10 的上限收益仍不足，标签路线需要谨慎。
- 历史规则标签 1s 时间匹配覆盖率：all=0.227, test=0.283，覆盖不足，不能直接当成当前全量标签。
- 当前没有足够覆盖、可部署、锚点前可知的事件标签；下一步应做当前事件级人工/实验条件标签，而不是直接把 oracle 标签输入模型。

## 边界
- `oracle_strength/timing/shape/direction` 来自未来真实轨迹，只能作为 auxiliary target、分层审计和上限分析。
- `oracle_error_label` 直接来自 v249 误差，是泄漏诊断，只能看理论极限，不能参与模型输入或部署决策。
- 历史规则标签来自旧事件版本，本轮只按 subject + session + anchor time 做最近邻匹配；覆盖不足时只作为线索。

## decision
| check                                      | requirement                                                                               |     value | pass   |
|:-------------------------------------------|:------------------------------------------------------------------------------------------|----------:|:-------|
| oracle_response_label_upper_bound_badtop10 | best future response label-known correction improves test bad_top10 by at least 0.05 RMSE | -0.009262 | False  |
| oracle_response_label_all_no_big_harm      | same label-known correction has test all delta <= 0.01                                    | -0.001286 | True   |
| history_rule_label_current_coverage        | historical rule labels match >= 80% current test events within 1s                         |  0.282609 | False  |
| deployable_current_label_available         | pre-anchor/current labels have coverage>=0.8, train-seen>=0.8, and useful test signal     |  0.575521 | False  |
| coarse_response_label_risk_only            | future response labels may identify risk, but risk AUC alone is not trajectory correction |  0.773525 | True   |

## label catalog
| label_family                | label_type                                |   coverage_all |   coverage_test |   label_n_all |   test_seen_in_train_key_rate |
|:----------------------------|:------------------------------------------|---------------:|----------------:|--------------:|------------------------------:|
| oracle_error_label          | future_error_leakage_diagnostic_only      |       1        |        1        |             3 |                             1 |
| oracle_strength_label       | future_response_auxiliary_oracle          |       1        |        1        |             3 |                             1 |
| oracle_timing_label         | future_response_auxiliary_oracle          |       1        |        1        |             2 |                             1 |
| oracle_shape_label          | future_response_auxiliary_oracle          |       1        |        1        |             4 |                             1 |
| oracle_direction_label      | future_response_auxiliary_oracle          |       1        |        1        |             2 |                             1 |
| hist_response_task_track    | historical_rule_label_time_matched_subset |       0.227078 |        0.282609 |             5 |                             1 |
| hist_response_task_class    | historical_rule_label_time_matched_subset |       0.227078 |        0.282609 |             6 |                             1 |
| hist_event_level            | historical_rule_label_time_matched_subset |       0.227078 |        0.282609 |             7 |                             1 |
| hist_road_design_risk_class | historical_rule_label_time_matched_subset |       0.227078 |        0.282609 |             4 |                             1 |
| hist_road_type_anchor       | historical_rule_label_time_matched_subset |       0.227078 |        0.282609 |             5 |                             1 |
| meta_observation_bin        | pre_anchor_metadata_proxy                 |       1        |        1        |             4 |                             1 |
| meta_event_order_bin        | pre_anchor_metadata_proxy                 |       1        |        1        |             4 |                             1 |
| meta_subject                | pre_anchor_metadata_proxy                 |       1        |        1        |            18 |                             0 |
| meta_recording              | pre_anchor_metadata_proxy                 |       1        |        1        |            79 |                             0 |

## test label risk AUC
| label_family                | label_type                                |   n |   coverage_in_split |   bad_top10_auc_from_train_label_rate |   bad_top10_vehicle_ambiguous_auc_from_train_label_rate |   test_or_split_seen_key_rate |
|:----------------------------|:------------------------------------------|----:|--------------------:|--------------------------------------:|--------------------------------------------------------:|------------------------------:|
| oracle_strength_label       | future_response_auxiliary_oracle          | 184 |            1        |                              0.773525 |                                                0.426036 |                             1 |
| oracle_error_label          | future_error_leakage_diagnostic_only      | 184 |            1        |                              0.654545 |                                                0.650888 |                             1 |
| hist_response_task_track    | historical_rule_label_time_matched_subset |  52 |            0.282609 |                              0.611979 |                                                0.734694 |                             1 |
| hist_response_task_class    | historical_rule_label_time_matched_subset |  52 |            0.282609 |                              0.611979 |                                                0.734694 |                             1 |
| hist_event_level            | historical_rule_label_time_matched_subset |  52 |            0.282609 |                              0.575521 |                                                0.693878 |                             1 |
| oracle_shape_label          | future_response_auxiliary_oracle          | 184 |            1        |                              0.561882 |                                                0.540237 |                             1 |
| meta_subject                | pre_anchor_metadata_proxy                 | 184 |            1        |                              0.5      |                                                0.5      |                             0 |
| meta_recording              | pre_anchor_metadata_proxy                 | 184 |            1        |                              0.5      |                                                0.5      |                             0 |
| meta_observation_bin        | pre_anchor_metadata_proxy                 | 184 |            1        |                              0.480383 |                                                0.451479 |                             1 |
| oracle_timing_label         | future_response_auxiliary_oracle          | 184 |            1        |                              0.477831 |                                                0.485996 |                             1 |
| hist_road_type_anchor       | historical_rule_label_time_matched_subset |  52 |            0.282609 |                              0.46875  |                                                0.469388 |                             1 |
| meta_event_order_bin        | pre_anchor_metadata_proxy                 | 184 |            1        |                              0.407974 |                                                0.424852 |                             1 |
| oracle_direction_label      | future_response_auxiliary_oracle          | 184 |            1        |                              0.3437   |                                                0.34931  |                             1 |
| hist_road_design_risk_class | historical_rule_label_time_matched_subset |  52 |            0.282609 |                              0.333333 |                                                0.564626 |                             1 |

## test v249_rmse eta by label
| label_family                | label_type                                |   eta_squared |   n |   label_key_n |
|:----------------------------|:------------------------------------------|--------------:|----:|--------------:|
| oracle_error_label          | future_error_leakage_diagnostic_only      |      0.332251 | 184 |             3 |
| oracle_strength_label       | future_response_auxiliary_oracle          |      0.247278 | 184 |             3 |
| hist_road_design_risk_class | historical_rule_label_time_matched_subset |      0.140764 |  52 |             4 |
| hist_event_level            | historical_rule_label_time_matched_subset |      0.138179 |  52 |             3 |
| meta_recording              | pre_anchor_metadata_proxy                 |      0.098486 | 184 |            14 |
| oracle_shape_label          | future_response_auxiliary_oracle          |      0.074011 | 184 |             4 |
| meta_subject                | pre_anchor_metadata_proxy                 |      0.030499 | 184 |             4 |
| hist_response_task_class    | historical_rule_label_time_matched_subset |      0.028417 |  52 |             4 |
| hist_response_task_track    | historical_rule_label_time_matched_subset |      0.028417 |  52 |             4 |
| meta_observation_bin        | pre_anchor_metadata_proxy                 |      0.017729 | 184 |             4 |
| meta_event_order_bin        | pre_anchor_metadata_proxy                 |      0.015192 | 184 |             4 |
| hist_road_type_anchor       | historical_rule_label_time_matched_subset |      0.001665 |  52 |             2 |
| oracle_direction_label      | future_response_auxiliary_oracle          |      0.000979 | 184 |             2 |
| oracle_timing_label         | future_response_auxiliary_oracle          |      0.000498 | 184 |             2 |

## label-known residual correction
| config_name                            | label_source                                      | group                       |   n |   coverage_in_split |   seen_key_rate |   baseline_rmse_mean |   corrected_rmse_mean |   delta_vs_v249_mean |
|:---------------------------------------|:--------------------------------------------------|:----------------------------|----:|--------------------:|----------------:|---------------------:|----------------------:|---------------------:|
| oracle_strength_timing_shape_direction | future_response_auxiliary_oracle                  | all_available               | 184 |            1        |        0.983696 |             0.396855 |              0.393506 |            -0.003349 |
| oracle_shape_direction                 | future_response_auxiliary_oracle                  | all_available               | 184 |            1        |        1        |             0.396855 |              0.393909 |            -0.002946 |
| oracle_strength_shape_direction        | future_response_auxiliary_oracle                  | all_available               | 184 |            1        |        1        |             0.396855 |              0.394954 |            -0.001901 |
| oracle_direction                       | future_response_auxiliary_oracle                  | all_available               | 184 |            1        |        1        |             0.396855 |              0.395137 |            -0.001718 |
| oracle_error_label_leaky               | future_error_leakage_diagnostic_only              | all_available               | 184 |            1        |        1        |             0.396855 |              0.395178 |            -0.001677 |
| oracle_shape                           | future_response_auxiliary_oracle                  | all_available               | 184 |            1        |        1        |             0.396855 |              0.395569 |            -0.001286 |
| history_task_track_tol1s               | historical_rule_label_time_matched_subset         | all_available               |  52 |            0.282609 |        1        |             0.427373 |              0.426169 |            -0.001204 |
| meta_subject                           | pre_anchor_metadata_proxy_subject_disjoint_unseen | all_available               | 184 |            1        |        0        |             0.396855 |              0.395799 |            -0.001055 |
| oracle_timing                          | future_response_auxiliary_oracle                  | all_available               | 184 |            1        |        1        |             0.396855 |              0.395904 |            -0.00095  |
| meta_observation_order                 | pre_anchor_metadata_proxy                         | all_available               | 184 |            1        |        1        |             0.396855 |              0.395937 |            -0.000918 |
| oracle_strength                        | future_response_auxiliary_oracle                  | all_available               | 184 |            1        |        1        |             0.396855 |              0.39603  |            -0.000825 |
| oracle_strength_shape                  | future_response_auxiliary_oracle                  | all_available               | 184 |            1        |        1        |             0.396855 |              0.396247 |            -0.000607 |
| history_task_track_risk_tol1s          | historical_rule_label_time_matched_subset         | all_available               |  52 |            0.282609 |        0.980769 |             0.427373 |              0.427264 |            -0.000108 |
| history_task_track_tol1s               | historical_rule_label_time_matched_subset         | bad_top10                   |   4 |            0.021739 |        1        |             1.12837  |              1.1188   |            -0.009572 |
| oracle_shape                           | future_response_auxiliary_oracle                  | bad_top10                   |  19 |            0.103261 |        1        |             0.89189  |              0.882628 |            -0.009262 |
| oracle_strength_shape                  | future_response_auxiliary_oracle                  | bad_top10                   |  19 |            0.103261 |        1        |             0.89189  |              0.883364 |            -0.008527 |
| oracle_shape_direction                 | future_response_auxiliary_oracle                  | bad_top10                   |  19 |            0.103261 |        1        |             0.89189  |              0.884611 |            -0.00728  |
| oracle_strength                        | future_response_auxiliary_oracle                  | bad_top10                   |  19 |            0.103261 |        1        |             0.89189  |              0.885158 |            -0.006732 |
| oracle_error_label_leaky               | future_error_leakage_diagnostic_only              | bad_top10                   |  19 |            0.103261 |        1        |             0.89189  |              0.885492 |            -0.006398 |
| history_task_track_risk_tol1s          | historical_rule_label_time_matched_subset         | bad_top10                   |   4 |            0.021739 |        1        |             1.12837  |              1.12211  |            -0.006263 |
| oracle_timing                          | future_response_auxiliary_oracle                  | bad_top10                   |  19 |            0.103261 |        1        |             0.89189  |              0.886493 |            -0.005398 |
| meta_subject                           | pre_anchor_metadata_proxy_subject_disjoint_unseen | bad_top10                   |  19 |            0.103261 |        0        |             0.89189  |              0.886801 |            -0.005089 |
| oracle_direction                       | future_response_auxiliary_oracle                  | bad_top10                   |  19 |            0.103261 |        1        |             0.89189  |              0.887975 |            -0.003916 |
| oracle_strength_shape_direction        | future_response_auxiliary_oracle                  | bad_top10                   |  19 |            0.103261 |        1        |             0.89189  |              0.888315 |            -0.003575 |
| oracle_strength_timing_shape_direction | future_response_auxiliary_oracle                  | bad_top10                   |  19 |            0.103261 |        1        |             0.89189  |              0.888338 |            -0.003552 |
| meta_observation_order                 | pre_anchor_metadata_proxy                         | bad_top10                   |  19 |            0.103261 |        1        |             0.89189  |              0.88841  |            -0.003481 |
| oracle_shape                           | future_response_auxiliary_oracle                  | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        1        |             0.908233 |              0.899979 |            -0.008255 |
| oracle_strength_shape                  | future_response_auxiliary_oracle                  | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        1        |             0.908233 |              0.900199 |            -0.008034 |
| oracle_strength                        | future_response_auxiliary_oracle                  | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        1        |             0.908233 |              0.902199 |            -0.006034 |
| oracle_shape_direction                 | future_response_auxiliary_oracle                  | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        1        |             0.908233 |              0.902568 |            -0.005666 |
| oracle_error_label_leaky               | future_error_leakage_diagnostic_only              | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        1        |             0.908233 |              0.902716 |            -0.005517 |
| oracle_timing                          | future_response_auxiliary_oracle                  | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        1        |             0.908233 |              0.903516 |            -0.004718 |
| meta_subject                           | pre_anchor_metadata_proxy_subject_disjoint_unseen | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        0        |             0.908233 |              0.903909 |            -0.004324 |
| history_task_track_tol1s               | historical_rule_label_time_matched_subset         | bad_top10_vehicle_ambiguous |   3 |            0.016304 |        1        |             1.24974  |              1.24595  |            -0.003797 |
| oracle_strength_timing_shape_direction | future_response_auxiliary_oracle                  | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        1        |             0.908233 |              0.905077 |            -0.003156 |
| oracle_direction                       | future_response_auxiliary_oracle                  | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        1        |             0.908233 |              0.905259 |            -0.002974 |
| oracle_strength_shape_direction        | future_response_auxiliary_oracle                  | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        1        |             0.908233 |              0.90532  |            -0.002913 |
| meta_observation_order                 | pre_anchor_metadata_proxy                         | bad_top10_vehicle_ambiguous |  15 |            0.081522 |        1        |             0.908233 |              0.905846 |            -0.002387 |
| history_task_track_risk_tol1s          | historical_rule_label_time_matched_subset         | bad_top10_vehicle_ambiguous |   3 |            0.016304 |        1        |             1.24974  |              1.25043  |             0.000685 |

## oracle label level examples
| label_family           | label            |   n |   bad_top10_rate |   bad_top10_enrichment_vs_split |   v249_rmse_mean |   true_peak_abs_mean |   true_line_length_mean |
|:-----------------------|:-----------------|----:|-----------------:|--------------------------------:|-----------------:|---------------------:|------------------------:|
| oracle_strength_label  | strong           |  53 |         0.283019 |                        2.74081  |         0.57576  |             2.2264   |                 4.2234  |
| oracle_direction_label | left             |  84 |         0.166667 |                        1.61404  |         0.405012 |             1.4021   |                 2.59973 |
| oracle_shape_label     | large_smooth     |  12 |         0.166667 |                        1.61404  |         0.555515 |             2.38323  |                 3.56476 |
| oracle_shape_label     | reverse          |  32 |         0.125    |                        1.21053  |         0.448362 |             1.44764  |                 2.77729 |
| oracle_shape_label     | multi_correction | 104 |         0.115385 |                        1.11741  |         0.398724 |             1.41223  |                 2.94364 |
| oracle_direction_label | right            | 100 |         0.05     |                        0.484211 |         0.390003 |             1.36051  |                 2.62759 |
| oracle_strength_label  | weak             |  56 |         0.035714 |                        0.345865 |         0.277676 |             0.635346 |                 1.28345 |
| oracle_shape_label     | single_or_smooth |  36 |         0.027778 |                        0.269006 |         0.292782 |             0.889798 |                 1.2041  |
| oracle_strength_label  | medium           |  75 |         0.026667 |                        0.258246 |         0.359415 |             1.33666  |                 2.47231 |

## history label match coverage
| split   |    n |   tol0p5_rate |   tol1_rate |
|:--------|-----:|--------------:|------------:|
| train   |  674 |      0.181009 |    0.209199 |
| val     |  309 |      0.187702 |    0.23301  |
| test    |  184 |      0.255435 |    0.282609 |
| all     | 1167 |      0.194516 |    0.227078 |

## guardrail
```json
{
  "pass": true,
  "event_n": 1167,
  "history_match_tol0p5_rate_all": 0.194515852613539,
  "history_match_tol1_rate_all": 0.22707797772065125,
  "history_match_tol1_rate_test": 0.2826086956521739,
  "best_oracle_response_config": "oracle_shape",
  "best_oracle_response_test_badtop10_delta": -0.009262216231539,
  "best_oracle_response_test_all_delta": -0.0012859252387298948,
  "oracle_error_label_leaky_test_badtop10_delta": -0.006398022343613483,
  "best_oracle_response_risk_label": "oracle_strength_label",
  "best_oracle_response_risk_test_auc": 0.773524720893142,
  "oracle_shape_label_test_v249_rmse_eta": 0.07401062365569566,
  "oracle_shape_label_test_badtop10_auc_from_train_rate": 0.5618819776714513,
  "best_pre_anchor_or_history_label": "hist_event_level",
  "best_pre_anchor_or_history_label_test_bad_auc": 0.5755208333333333,
  "best_pre_anchor_or_history_label_test_v249_rmse_eta": 0.13817888008266857,
  "oracle_response_label_upper_bound_useful": false,
  "oracle_response_label_all_no_big_harm": true,
  "deployable_event_label_available_now": false,
  "history_rule_label_coverage_sufficient_now": false,
  "manual_or_experimental_condition_label_priority": true,
  "coarse_response_labels_are_risk_markers_not_correction_solution": true,
  "future_derived_labels_used_as_inputs": false,
  "test_used_for_threshold_model_selection": false,
  "recommended_next_step": "build/current-event manual or experimental-condition labels, then train auxiliary response heads; do not use oracle labels as inputs",
  "goal_achieved_now": false
}
```