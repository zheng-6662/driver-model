# v299 within-subject split residual calibration

## 结论
- 在同被试内切分后，轻量 residual 校准已经显示明显收益，值得进入完整模型重训。
- 重要边界：本轮固定使用旧 v249 预测，新的 within-test 中有 0.582 原本属于旧 v249 train split，所以本轮是快速潜力审计，不是正式重训结论。
- val 选择方法 `base_curve_meta_subject__extra_trees_d5`：test all delta=-0.006707, test within_bad_top10 delta=-0.073816。
- test-best diagnostic `base_curve_meta_subject__extra_trees_d5`：test within_bad_top10 delta=-0.073816，该值只作诊断，不作为可部署选择。

## split guardrail
```json
{
  "event_n": 1167,
  "unique_event_n": 1167,
  "duplicate_event_uid_n": 0,
  "event_in_multiple_splits_n": 0,
  "subject_n": 18,
  "subject_with_all_three_splits_n": 18,
  "train_n": 702,
  "val_n": 233,
  "test_n": 232,
  "within_test_original_v249_train_n": 135,
  "within_test_original_v249_train_rate": 0.5818965517241379,
  "fixed_v249_predictions_have_original_split_exposure": true
}
```

## subject split counts
| subject   |   event_n |   train_n |   val_n |   test_n |   recording_n |
|:----------|----------:|----------:|--------:|---------:|--------------:|
| zx        |       153 |        92 |      31 |       30 |             9 |
| hzh       |       118 |        71 |      24 |       23 |             6 |
| gzj       |       105 |        63 |      21 |       21 |             6 |
| byx       |       102 |        61 |      20 |       21 |             5 |
| txj       |        91 |        55 |      18 |       18 |             4 |
| yyl       |        87 |        52 |      17 |       18 |             4 |
| rjy       |        82 |        49 |      16 |       17 |             5 |
| yzy       |        79 |        47 |      16 |       16 |             5 |
| lxy       |        65 |        39 |      13 |       13 |             3 |
| zdq       |        48 |        29 |      10 |        9 |             6 |
| cwh       |        46 |        28 |       9 |        9 |             4 |
| tyy       |        43 |        26 |       9 |        8 |             3 |
| jy        |        42 |        25 |       8 |        9 |             5 |
| gf        |        36 |        22 |       7 |        7 |             5 |
| zxy       |        36 |        22 |       7 |        7 |             5 |
| zt        |        15 |         9 |       3 |        3 |             1 |
| lx        |        13 |         8 |       3 |        2 |             2 |
| xst       |         6 |         4 |       1 |        1 |             1 |

## chosen by val
| choice_name               | method                                   | choice_rule                             |   val_all_delta |   val_within_bad_top10_delta |   test_all_delta |   test_within_bad_top10_delta |   test_within_bad_top10_rmse |
|:--------------------------|:-----------------------------------------|:----------------------------------------|----------------:|-----------------------------:|-----------------:|------------------------------:|-----------------------------:|
| chosen_by_val             | base_curve_meta_subject__extra_trees_d5  | val no-harm among non-recording methods |       -0.00316  |                    -0.065194 |        -0.006707 |                     -0.073816 |                     0.964519 |
| test_best_diagnostic      | base_curve_meta_subject__extra_trees_d5  | test diagnostic only, not selectable    |       -0.00316  |                    -0.065194 |        -0.006707 |                     -0.073816 |                     0.964519 |
| recording_best_diagnostic | recording_train_mean_residual_diagnostic | session/recording diagnostic only       |        0.006223 |                    -0.006463 |         0.007357 |                      0.026631 |                     1.06497  |

## test summary top methods
| method                                                   |   n |   baseline_rmse_mean |   method_rmse_mean |   delta_vs_v249_mean |   delta_vs_v249_median |   improved_rate |
|:---------------------------------------------------------|----:|---------------------:|-------------------:|---------------------:|-----------------------:|----------------:|
| base_curve_meta_subject__extra_trees_d5                  |  24 |              1.03834 |           0.964519 |            -0.073816 |              -0.020908 |        0.791667 |
| base_curve_plus_meta__ridge_a1                           |  24 |              1.03834 |           0.995654 |            -0.042682 |              -0.030129 |        0.75     |
| base_curve_only__ridge_a1                                |  24 |              1.03834 |           0.998067 |            -0.040268 |              -0.025111 |        0.75     |
| base_curve_plus_meta__extra_trees_d5                     |  24 |              1.03834 |           1.00135  |            -0.036984 |              -0.033633 |        0.875    |
| base_curve_meta_subject__ridge_a1                        |  24 |              1.03834 |           1.00159  |            -0.036744 |              -0.030612 |        0.583333 |
| base_curve_plus_meta__ridge_a10                          |  24 |              1.03834 |           1.00201  |            -0.036325 |              -0.022928 |        0.75     |
| base_curve_only__ridge_a10                               |  24 |              1.03834 |           1.00476  |            -0.033579 |              -0.023283 |        0.708333 |
| base_curve_meta_subject__ridge_a10                       |  24 |              1.03834 |           1.0063   |            -0.032034 |              -0.034909 |        0.625    |
| base_curve_plus_meta__ridge_a100                         |  24 |              1.03834 |           1.00709  |            -0.031242 |              -0.018933 |        0.833333 |
| base_curve_only__ridge_a100                              |  24 |              1.03834 |           1.00918  |            -0.029157 |              -0.0157   |        0.75     |
| base_curve_meta_subject__ridge_a100                      |  24 |              1.03834 |           1.00948  |            -0.028857 |              -0.031569 |        0.625    |
| subject_train_mean_residual                              |  24 |              1.03834 |           1.03407  |            -0.004266 |              -0.013482 |        0.541667 |
| subject_onehot_only__ridge_a100                          |  24 |              1.03834 |           1.03487  |            -0.003465 |              -0.013702 |        0.541667 |
| subject_onehot_only__ridge_a10                           |  24 |              1.03834 |           1.03506  |            -0.003281 |              -0.014253 |        0.541667 |
| subject_onehot_only__ridge_a1                            |  24 |              1.03834 |           1.03508  |            -0.003255 |              -0.014309 |        0.541667 |
| global_train_mean_residual                               |  24 |              1.03834 |           1.03607  |            -0.002268 |              -0.004015 |        0.708333 |
| v249_no_correction                                       |  24 |              1.03834 |           1.03834  |             0        |               0        |        0        |
| recording_train_mean_residual_diagnostic                 |  24 |              1.03834 |           1.06497  |             0.026631 |               0.019221 |        0.291667 |
| base_curve_meta_subject_recording_diagnostic__ridge_a100 |  24 |              1.03834 |           1.07481  |             0.036478 |               0.024328 |        0.333333 |
| base_curve_meta_subject_recording_diagnostic__ridge_a1   |  24 |              1.03834 |           1.07831  |             0.039974 |               0.051874 |        0.375    |
| base_curve_meta_subject_recording_diagnostic__ridge_a10  |  24 |              1.03834 |           1.08162  |             0.043282 |               0.05219  |        0.333333 |

## chosen method by original v249 split
| original_v249_split   | group            |   n |   delta_vs_v249_mean |   delta_vs_v249_median |   improved_rate |
|:----------------------|:-----------------|----:|---------------------:|-----------------------:|----------------:|
| test                  | all              |  36 |            -0.009609 |              -0.0109   |        0.666667 |
| test                  | within_bad_top10 |   6 |            -0.020667 |              -0.017587 |        1        |
| train                 | all              | 135 |            -0.004958 |              -0.003755 |        0.622222 |
| val                   | all              |  61 |            -0.008862 |               0.003502 |        0.491803 |
| val                   | within_bad_top10 |  18 |            -0.091533 |              -0.066107 |        0.722222 |
| nontrain_combined     | all              |  97 |            -0.00914  |              -0.009021 |        0.556701 |
| nontrain_combined     | within_bad_top10 |  24 |            -0.073816 |              -0.020908 |        0.791667 |

## guardrail
```json
{
  "pass": true,
  "split_method": "within_subject_random_event_split_60_20_20",
  "seed": 20260702,
  "same_event_never_repeated_across_splits": true,
  "uses_original_subject_disjoint_split_for_training": false,
  "full_v249_retrained": false,
  "experiment_scope": "fast residual calibration on fixed v249 predictions",
  "formal_claim_requires_full_retrain_on_within_subject_split": true,
  "event_n": 1167,
  "unique_event_n": 1167,
  "duplicate_event_uid_n": 0,
  "event_in_multiple_splits_n": 0,
  "subject_n": 18,
  "subject_with_all_three_splits_n": 18,
  "train_n": 702,
  "val_n": 233,
  "test_n": 232,
  "within_test_original_v249_train_n": 135,
  "within_test_original_v249_train_rate": 0.5818965517241379,
  "fixed_v249_predictions_have_original_split_exposure": true,
  "chosen_method": "base_curve_meta_subject__extra_trees_d5",
  "chosen_test_all_delta": -0.006706581865468402,
  "chosen_test_within_bad_top10_delta": -0.07381645453379433,
  "chosen_test_within_bad_top10_rmse": 0.9645188665735858,
  "chosen_test_within_bad_top10_baseline_rmse": 1.03833532110738,
  "test_best_method": "base_curve_meta_subject__extra_trees_d5",
  "test_best_within_bad_top10_delta": -0.07381645453379433,
  "test_best_all_delta": -0.006706581865468402,
  "chosen_test_original_nontrain_within_bad_top10_delta": -0.07381645453379433,
  "chosen_test_original_nontrain_within_bad_top10_n": 24,
  "within_subject_residual_route_promising": true,
  "complete_model_retrain_recommended_next": true,
  "goal_achieved_now": false
}
```