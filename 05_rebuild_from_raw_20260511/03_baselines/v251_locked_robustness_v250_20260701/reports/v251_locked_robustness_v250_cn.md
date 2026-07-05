# v251 locked robustness audit for v250_minimal_lateral7

## 本轮边界

- 固定 v250 best validation model：`v250_minimal_lateral7`。
- 不重新训练，不调通道，不用 test 做选择。
- 只在 locked test 上做 bucket/delay、subject、recording、event-level bootstrap CI 和逐样本回退审计。
- 不做 anchor selector、gate/router、response-type hard routing，不删除样本。

## v250 选择来源

| model_name            |   n_hist_channels | channels                                                            |   best_epoch |   best_val_loss | accepted_as_channel_candidate   |
|:----------------------|------------------:|:--------------------------------------------------------------------|-------------:|----------------:|:--------------------------------|
| v250_minimal_lateral7 |                 7 | steering|speed_kmh|ay|yaw_rate|roll|lane_curvature|lateral_distance |           19 |        0.487806 | True                            |

## Locked Test Bucket/Delay 摘要

| bucket             |   delay_ms |   n |   event_n |   mean_tail_rmse_v241 |   mean_tail_rmse_v250 |   mean_delta_tail |   tail_improve_rate |   mean_delta_sample |   mean_delta_range_ratio |   mean_delta_slope_ratio |
|:-------------------|-----------:|----:|----------:|----------------------:|----------------------:|------------------:|--------------------:|--------------------:|-------------------------:|-------------------------:|
| all                |          0 | 184 |       184 |              0.475053 |              0.396888 |        -0.0781656 |            0.619565 |          -0.0573361 |              -0.0855774  |              -0.0145713  |
| all                |        600 | 184 |       184 |              0.38532  |              0.318    |        -0.0673206 |            0.63587  |          -0.0604629 |              -0.0577978  |              -0.0172045  |
| all                |       1000 | 184 |       184 |              0.304615 |              0.236594 |        -0.0680216 |            0.684783 |          -0.0680216 |              -0.344285   |              -0.0148508  |
| normal_predictable |          0 |  99 |        99 |              0.371164 |              0.292139 |        -0.0790246 |            0.616162 |          -0.0601449 |              -0.22357    |              -0.0857899  |
| normal_predictable |        600 |  99 |        99 |              0.297837 |              0.232909 |        -0.0649279 |            0.676768 |          -0.0562934 |              -0.164184   |              -0.033596   |
| normal_predictable |       1000 |  99 |        99 |              0.220595 |              0.177867 |        -0.0427276 |            0.646465 |          -0.0427276 |              -0.700936   |              -0.112426   |
| observe_later_like |          0 |  27 |        27 |              0.792468 |              0.639918 |        -0.15255   |            0.777778 |          -0.100892  |               0.0326462  |               0.0333531  |
| observe_later_like |        600 |  27 |        27 |              0.588433 |              0.511828 |        -0.0766045 |            0.666667 |          -0.0682823 |               0.0182036  |              -0.0286375  |
| observe_later_like |       1000 |  27 |        27 |              0.50421  |              0.375296 |        -0.128914  |            0.740741 |          -0.128914  |               0.013238   |               0.0822961  |
| strong_steer       |          0 |  80 |        80 |              0.590904 |              0.504266 |        -0.0866381 |            0.6375   |          -0.0596733 |               0.0778664  |               0.0813607  |
| strong_steer       |        600 |  80 |        80 |              0.494278 |              0.422398 |        -0.0718795 |            0.5875   |          -0.067391  |               0.06944    |               0.00114956 |
| strong_steer       |       1000 |  80 |        80 |              0.405783 |              0.307555 |        -0.0982275 |            0.725    |          -0.0982275 |               0.0933607  |               0.103132   |
| bad_top10_v241     |          0 |  24 |        24 |              1.09733  |              0.780268 |        -0.317058  |            0.833333 |          -0.208412  |              -0.00778932 |               0.0124293  |
| bad_top10_v241     |        600 |  18 |        18 |              1.0074   |              0.745516 |        -0.261883  |            0.833333 |          -0.229208  |               0.0477181  |              -0.0275456  |
| bad_top10_v241     |       1000 |  13 |        13 |              0.864711 |              0.509487 |        -0.355224  |            0.923077 |          -0.355224  |               0.198819   |               0.222572   |

## Subject-Level 摘要

| subject   | bucket             |   n |   event_n |   mean_delta_tail |   tail_improve_rate |   max_delta_tail |
|:----------|:-------------------|----:|----------:|------------------:|--------------------:|-----------------:|
| cwh       | all                | 276 |        46 |       -0.0555405  |            0.655797 |        0.601239  |
| lx        | all                |  78 |        13 |       -0.0936831  |            0.75641  |        0.212543  |
| rjy       | all                | 492 |        82 |       -0.0869452  |            0.648374 |        0.45795   |
| tyy       | all                | 258 |        43 |       -0.0441164  |            0.627907 |        1.49769   |
| cwh       | normal_predictable | 204 |        34 |       -0.0768123  |            0.696078 |        0.346098  |
| lx        | normal_predictable |  30 |         5 |       -0.0672365  |            0.733333 |        0.197123  |
| rjy       | normal_predictable | 246 |        41 |       -0.0883335  |            0.682927 |        0.45795   |
| tyy       | normal_predictable | 114 |        19 |       -0.00481242 |            0.578947 |        1.49769   |
| cwh       | observe_later_like |   6 |         1 |       -0.255843   |            1        |       -0.0182106 |
| lx        | observe_later_like |   6 |         1 |       -0.121036   |            1        |       -0.0795504 |
| rjy       | observe_later_like | 114 |        19 |       -0.104975   |            0.692982 |        0.320818  |
| tyy       | observe_later_like |  36 |         6 |       -0.0544411  |            0.583333 |        0.44153   |
| cwh       | strong_steer       |  72 |        12 |        0.00472938 |            0.541667 |        0.601239  |
| lx        | strong_steer       |  48 |         8 |       -0.110212   |            0.770833 |        0.212543  |
| rjy       | strong_steer       | 222 |        37 |       -0.0956042  |            0.617117 |        0.414782  |
| tyy       | strong_steer       | 138 |        23 |       -0.0779377  |            0.681159 |        0.68512   |

## Event-Level Bootstrap CI

| bucket             |    n |   event_n |   tail_delta_mean |   tail_delta_ci95_low |   tail_delta_ci95_high |   tail_prob_delta_lt0 | tail_ci_excludes_zero_negative   |
|:-------------------|-----:|----------:|------------------:|----------------------:|-----------------------:|----------------------:|:---------------------------------|
| all                | 1104 |       184 |        -0.0695612 |            -0.0925647 |             -0.046657  |                1      | True                             |
| normal_predictable |  594 |        99 |        -0.0672819 |            -0.0988602 |             -0.036085  |                1      | True                             |
| observe_later_like |  162 |        27 |        -0.0999278 |            -0.160822  |             -0.0386283 |                0.9985 | True                             |
| strong_steer       |  480 |        80 |        -0.0769359 |            -0.115148  |             -0.0386719 |                1      | True                             |
| bad_top10_v241     |  111 |        32 |        -0.303618  |            -0.381814  |             -0.22683   |                1      | True                             |

## 主要回退样本

| event_uid                                         | subject   |   delay_ms |   tail_rmse_v241 |   tail_rmse_v250 |   delta_tail_rmse_v250_minus_v241 |   sample_rmse_v241 |   sample_rmse_v250 |
|:--------------------------------------------------|:----------|-----------:|-----------------:|-----------------:|----------------------------------:|-------------------:|-------------------:|
| tyy_Entity_Recording_2025_09_28_14_40_01_v108_012 | tyy       |          0 |        0.345277  |         1.84297  |                          1.49769  |          0.293888  |           1.35857  |
| tyy_Entity_Recording_2025_09_28_14_40_01_v108_012 | tyy       |        200 |        0.322144  |         1.04773  |                          0.72559  |          0.297445  |           0.813992 |
| tyy_Entity_Recording_2025_09_28_14_40_01_v108_012 | tyy       |        800 |        0.465185  |         1.1795   |                          0.714319 |          0.43357   |           1.08617  |
| tyy_Entity_Recording_2025_09_28_14_57_17_v108_028 | tyy       |        200 |        0.445028  |         1.13015  |                          0.68512  |          0.375433  |           0.876174 |
| tyy_Entity_Recording_2025_09_28_14_57_17_v108_028 | tyy       |        400 |        0.381162  |         0.997535 |                          0.616373 |          0.412846  |           0.839359 |
| tyy_Entity_Recording_2025_09_28_14_57_17_v108_028 | tyy       |        600 |        0.346549  |         0.956243 |                          0.609694 |          0.300641  |           0.835865 |
| tyy_Entity_Recording_2025_09_28_14_57_17_v108_028 | tyy       |        800 |        0.336071  |         0.942334 |                          0.606263 |          0.325257  |           0.873992 |
| cwh_Entity_Recording_2025_09_26_20_06_19_v108_017 | cwh       |        400 |        0.21217   |         0.813409 |                          0.601239 |          0.255401  |           0.709705 |
| tyy_Entity_Recording_2025_09_28_14_57_17_v108_028 | tyy       |       1000 |        0.342162  |         0.927177 |                          0.585015 |          0.342162  |           0.927177 |
| tyy_Entity_Recording_2025_09_28_14_57_17_v108_028 | tyy       |          0 |        0.37245   |         0.874776 |                          0.502326 |          0.454472  |           0.721098 |
| rjy_Entity_Recording_2025_09_28_20_15_42_v108_027 | rjy       |        600 |        0.0769531 |         0.534903 |                          0.45795  |          0.0697392 |           0.458569 |
| tyy_Entity_Recording_2025_09_28_14_23_43_v108_004 | tyy       |        400 |        0.566774  |         1.0083   |                          0.44153  |          0.473808  |           0.826855 |

## 下一步决策

| decision_item                      | decision                                             | reason                                                                                                                              |
|:-----------------------------------|:-----------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| locked_robustness_pass             | True                                                 | All key test bucket/delay tail deltas are negative, all-delay event-level bootstrap CIs exclude zero, and subject win rate is high. |
| all_key_bucket_delay_tail_negative | True                                                 | Requires every test delay in all/normal/observe_later/strong to have mean tail delta < 0.                                           |
| all_delay_bootstrap_ci_pass        | True                                                 | Requires all-delay event-level bootstrap 95% CI upper bound < 0 for key buckets.                                                    |
| subject_win_rate                   | 0.9375                                               | Fraction of subject/bucket summaries with mean tail delta < 0 across key buckets.                                                   |
| formal_replacement_allowed         | False                                                | v251 is locked robustness evidence; formal replacement still needs mainline packaging and final consistency audit.                  |
| current_status                     | pass_locked_robustness                               | Robustness status for v250_minimal_lateral7.                                                                                        |
| recommended_next_task              | v252_mainline_candidate_pack_or_subject_level_retest | Next bounded step after locked robustness audit.                                                                                    |

## 关键图

- `figures\v251_test_bucket_delay_tail_delta.png`
- `figures\v251_subject_bucket_tail_delta.png`
- `figures\v251_bootstrap_ci_all_delay.png`
- `figures\v251_bad_top10_casebook.png`
- `figures\v251_worst_regression_casebook.png`

## 关键产物

- `tables/v251_sample_locked_delta.csv`
- `tables/v251_bucket_delay_locked_summary.csv`
- `tables/v251_subject_locked_summary.csv`
- `tables/v251_recording_locked_summary.csv`
- `tables/v251_event_bootstrap_ci.csv`
- `tables/v251_worst_regressions.csv`
- `tables/v251_bad_top10_casebook_index.csv`
