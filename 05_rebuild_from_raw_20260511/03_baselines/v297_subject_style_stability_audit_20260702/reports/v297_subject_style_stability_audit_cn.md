# v297 subject style stability audit

## 结论
- 当前审计只显示弱 subject 风格信号；驾驶风格可以作为辅助，但不应单独作为主线。
- train key subject eta mean=0.059843, same-subject distance ratio=0.710302, test rolling-history relative RMSE improvement=0.069876.

## 解释边界
- 这里不假设一场实验内前后 trial 有直接因果关系。
- rolling history 只用来检验同一被试是否存在稳定总体倾向，而不是事件序列记忆。
- oracle labels 来自未来轨迹，只能用于辅助监督/分层/上限分析，不能作为测试时直接输入。

## split / subject 概况
| split   |   n |
|:--------|----:|
| test    | 184 |
| train   | 674 |
| val     | 309 |

| subject   |   event_n |   recording_n |   train_n |   val_n |   test_n |
|:----------|----------:|--------------:|----------:|--------:|---------:|
| zx        |       153 |             9 |       153 |       0 |        0 |
| hzh       |       118 |             6 |       118 |       0 |        0 |
| gzj       |       105 |             6 |         0 |     105 |        0 |
| byx       |       102 |             5 |       102 |       0 |        0 |
| txj       |        91 |             4 |         0 |      91 |        0 |
| yyl       |        87 |             4 |        87 |       0 |        0 |
| rjy       |        82 |             5 |         0 |       0 |       82 |
| yzy       |        79 |             5 |        79 |       0 |        0 |
| lxy       |        65 |             3 |         0 |      65 |        0 |
| zdq       |        48 |             6 |         0 |      48 |        0 |
| cwh       |        46 |             4 |         0 |       0 |       46 |
| tyy       |        43 |             3 |         0 |       0 |       43 |
| jy        |        42 |             5 |        42 |       0 |        0 |
| gf        |        36 |             5 |        36 |       0 |        0 |
| zxy       |        36 |             5 |        36 |       0 |        0 |
| zt        |        15 |             1 |        15 |       0 |        0 |
| lx        |        13 |             2 |         0 |       0 |       13 |
| xst       |         6 |             1 |         6 |       0 |        0 |

## decision
| check                            | requirement                                                                |    value | pass   |
|:---------------------------------|:---------------------------------------------------------------------------|---------:|:-------|
| subject_eta_mean_train           | key train subject eta mean >= 0.05                                         | 0.059843 | True   |
| same_subject_distance_ratio      | same-subject response distance <= 0.95 * different-subject distance        | 0.710302 | True   |
| rolling_history_test_improvement | test rolling style mean relative RMSE improvement >= 0.02 for history_n>=3 | 0.069876 | True   |
| rolling_history_positive_targets | more than half key targets improve on test with history_n>=3               | 0.285714 | False  |

## subject eta top
| target              |   eta_squared |   n |   group_n |
|:--------------------|--------------:|----:|----------:|
| true_line_length    |      0.11971  | 674 |        10 |
| true_early_peak_abs |      0.089401 | 674 |        10 |
| true_range          |      0.084834 | 674 |        10 |
| true_peak_abs       |      0.084419 | 674 |        10 |
| true_late_peak_abs  |      0.076696 | 674 |        10 |
| v249_rmse           |      0.070471 | 674 |        10 |
| true_peak_time_s    |      0.051774 | 674 |        10 |
| true_tail_mean_abs  |      0.048768 | 674 |        10 |
| v249_tail_rmse      |      0.046683 | 674 |        10 |
| true_final_delta    |      0.03643  | 674 |        10 |
| v249_peak_abs_error |      0.034015 | 674 |        10 |
| v249_residual_final |      0.017778 | 674 |        10 |
| v249_residual_mean  |      0.009411 | 674 |        10 |

## pair distance
| pair_group          |   pair_n |   distance_mean |   distance_median |   same_subject_mean_distance_ratio |
|:--------------------|---------:|----------------:|------------------:|-----------------------------------:|
| same_subject        |    18472 |         2.32699 |           1.45119 |                           0.710302 |
| different_subject   |   231528 |         3.27606 |           2.02262 |                           0.710302 |
| same_recording      |     3874 |         2.50672 |           1.52292 |                           0.710302 |
| different_recording |   246126 |         3.21694 |           1.97014 |                           0.710302 |

## rolling history predictability
| target              |   n |   rmse_history |   rmse_global |   relative_rmse_improvement |   r2_history_vs_global |
|:--------------------|----:|---------------:|--------------:|----------------------------:|-----------------------:|
| v249_tail_rmse      | 172 |       0.326946 |      0.453985 |                    0.279832 |               0.481358 |
| v249_rmse           | 172 |       0.241324 |      0.332739 |                    0.274734 |               0.47399  |
| v249_peak_abs_error | 172 |       0.532962 |      0.542023 |                    0.016717 |               0.033155 |
| true_peak_time_s    | 172 |       0.405789 |      0.405145 |                   -0.001589 |              -0.00318  |
| true_line_length    | 172 |       1.42966  |      1.42707  |                   -0.001814 |              -0.003632 |
| true_peak_abs       | 172 |       0.700725 |      0.695094 |                   -0.008102 |              -0.016269 |
| v249_residual_mean  | 172 |       0.249379 |      0.247021 |                   -0.009549 |              -0.019188 |
| true_late_peak_abs  | 172 |       0.708428 |      0.701252 |                   -0.010232 |              -0.020569 |
| true_early_peak_abs | 172 |       0.587248 |      0.580422 |                   -0.011761 |              -0.02366  |
| true_tail_mean_abs  | 172 |       0.557367 |      0.549692 |                   -0.013962 |              -0.028119 |
| true_range          | 172 |       0.913417 |      0.896341 |                   -0.019051 |              -0.038464 |
| v249_residual_final | 172 |       0.611968 |      0.586131 |                   -0.044081 |              -0.090104 |
| true_final_delta    | 172 |       1.1539   |      1.10487  |                   -0.044379 |              -0.090728 |

## binary rolling history
| target                      |   n |   positive_rate |   history_auc |
|:----------------------------|----:|----------------:|--------------:|
| true_reverse_flag           | 172 |        0.436047 |      0.515189 |
| true_multi_correction_flag  | 172 |        0.575581 |      0.544278 |
| true_late_peak_flag         | 172 |        0.906977 |      0.433093 |
| bad_top10                   | 172 |        0.098837 |      0.562998 |
| bad_top10_vehicle_ambiguous | 172 |        0.081395 |      0.59991  |

## oracle label candidates
| label_family           | label            |   n |     rate |
|:-----------------------|:-----------------|----:|---------:|
| oracle_strength_label  | strong           | 431 | 0.369323 |
| oracle_strength_label  | medium           | 380 | 0.325621 |
| oracle_strength_label  | weak             | 356 | 0.305056 |
| oracle_timing_label    | late_peak        | 988 | 0.846615 |
| oracle_timing_label    | early_peak       | 179 | 0.153385 |
| oracle_shape_label     | multi_correction | 552 | 0.473008 |
| oracle_shape_label     | single_or_smooth | 293 | 0.251071 |
| oracle_shape_label     | reverse          | 224 | 0.191945 |
| oracle_shape_label     | large_smooth     |  98 | 0.083976 |
| oracle_direction_label | right            | 589 | 0.504713 |
| oracle_direction_label | left             | 578 | 0.495287 |
| oracle_error_label     | normal_error     | 607 | 0.520137 |
| oracle_error_label     | very_high_error  | 465 | 0.398458 |
| oracle_error_label     | high_error       |  95 | 0.081405 |

## guardrail
```json
{
  "pass": true,
  "event_n": 1167,
  "key_subject_eta_train_mean": 0.05984250666569984,
  "key_subject_eta_train_median": 0.05177379383776995,
  "same_subject_mean_distance_ratio": 0.7103016207957651,
  "rolling_history_test_relative_rmse_improvement_mean_history3": 0.06987629022762502,
  "rolling_history_test_positive_target_rate_history3": 0.2857142857142857,
  "binary_history_test_auc_mean_history3": 0.5310936078301502,
  "style_route_supported_now": false,
  "weak_style_signal_exists": true,
  "event_label_route_priority": true,
  "test_used_for_model_training_or_threshold": false,
  "future_derived_oracle_labels_are_not_deployable_inputs": true
}
```