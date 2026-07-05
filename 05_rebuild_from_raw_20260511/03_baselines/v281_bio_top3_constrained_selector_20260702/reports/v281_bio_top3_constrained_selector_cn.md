# v281 bio-top3 constrained selector

## 目的

v272 的 bio top3 oracle 有少量上限，但 bio top1 和直接生理最近邻失败。v281 将问题缩窄为：在 vehicle top40 内只看 bio 最近 3 个候选，训练选择器判断哪个候选以及何时覆盖 latest。

## 核心结果

- fixed wait-latest test bad_top10: `0.695048`
- val 选择的最好可部署 test bad_top10: `0.695048`
- test diagnostic 最好 bad_top10: `0.684235`
- bio top3 oracle bad_top10: `0.673823`
- 可部署规则是否超过 fixed latest: `False`
- diagnostic 是否超过 fixed latest: `True`
- bio top3 oracle 是否超过 fixed latest: `True`

## 决策汇总

| source                     | label                                                                      |     rmse | deployable   |   override_rate |   val_bad_delta |   val_all_delta |   stable_pass |   delta_vs_fixed_latest | passes_fixed_latest   |
|:---------------------------|:---------------------------------------------------------------------------|---------:|:-------------|----------------:|----------------:|----------------:|--------------:|------------------------:|:----------------------|
| baseline                   | policy_wait_to_latest_anchor                                               | 0.695048 | True         |      nan        |     nan         |    nan          |           nan |             4.15347e-07 | False                 |
| oracle                     | oracle_best_anchor_upper_bound                                             | 0.612475 | False        |      nan        |     nan         |    nan          |           nan |            -0.0825726   | True                  |
| best_any                   | bio_top3_subject_seq_pca72_rankdist threshold=inf                          | 0.695048 | True         |        0        |       0         |      0          |             0 |             4.15347e-07 | False                 |
| best_active                | bio_top3_subject_seq_pca72_rankdist threshold=0.05000859323311404          | 0.695048 | True         |        0        |       0.0127131 |      0.00127543 |             0 |             4.15347e-07 | False                 |
| test_best_diagnostic       | bio_top3_subject_seq_pca72_rankdist_vehicle threshold=0.019248106103915027 | 0.684235 | False        |        0.105263 |       0.121841  |      0.0162302  |             0 |            -0.0108127   | True                  |
| bio_top3_oracle_val_chosen | subject_seq_pca72: oracle inside bio_top3                                  | 0.772747 | False        |      nan        |     nan         |    nan          |           nan |             0.0776989   | False                 |
| bio_top3_oracle_test_best  | subject_summary64: oracle inside bio_top3                                  | 0.673823 | False        |      nan        |     nan         |    nan          |           nan |            -0.0212249   | True                  |

## raw_set / top3 oracle 汇总

| raw_set           | split   |   all_n |   all_oracle_rmse |   bad_top10_n |   bad_top10_oracle_rmse |   bad_top10_latest_rmse |   bad_top10_delta_vs_latest |
|:------------------|:--------|--------:|------------------:|--------------:|------------------------:|------------------------:|----------------------------:|
| subject_seq_pca72 | val     |     309 |          0.474321 |            31 |                1.22004  |                1.07279  |                   0.147256  |
| subject_seq_pca72 | test    |     184 |          0.304556 |            19 |                0.772747 |                0.695048 |                   0.0776985 |
| subject_summary64 | val     |     309 |          0.485586 |            31 |                1.33951  |                1.07279  |                   0.266723  |
| subject_summary64 | test    |     184 |          0.288415 |            19 |                0.673823 |                0.695048 |                  -0.0212253 |

## 特征组

| feature_set                                     | raw_set           |   feature_n |   train_candidate_rows |   val_candidate_rows |
|:------------------------------------------------|:------------------|------------:|-----------------------:|---------------------:|
| bio_top3_subject_summary64_rankdist             | subject_summary64 |          10 |                   2022 |                  927 |
| bio_top3_subject_summary64_rankdist_vehicle     | subject_summary64 |          41 |                   2022 |                  927 |
| bio_top3_subject_summary64_rankdist_vehicle_bio | subject_summary64 |         177 |                   2022 |                  927 |
| bio_top3_subject_seq_pca72_rankdist             | subject_seq_pca72 |          10 |                   2022 |                  927 |
| bio_top3_subject_seq_pca72_rankdist_vehicle     | subject_seq_pca72 |          41 |                   2022 |                  927 |
| bio_top3_subject_seq_pca72_rankdist_vehicle_bio | subject_seq_pca72 |         177 |                   2022 |                  927 |

## val 口径排名前 18

| feature_set                                     |   threshold |   val_bad_top10_selected_rmse |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_override_rate |   selection_score |
|:------------------------------------------------|------------:|------------------------------:|--------------------------------:|--------------------------:|-------------------------------:|-------------------------------:|------------------:|
| bio_top3_subject_summary64_rankdist             | inf         |                       1.07279 |                      0          |               0           |                       0.695048 |                              0 |        0          |
| bio_top3_subject_summary64_rankdist_vehicle     | inf         |                       1.07279 |                      0          |               0           |                       0.695048 |                              0 |        0          |
| bio_top3_subject_summary64_rankdist_vehicle_bio |   0.0553856 |                       1.07279 |                      0          |              -0.000120068 |                       0.695048 |                              0 |        0          |
| bio_top3_subject_summary64_rankdist_vehicle_bio | inf         |                       1.07279 |                      0          |               0           |                       0.695048 |                              0 |        0          |
| bio_top3_subject_seq_pca72_rankdist             | inf         |                       1.07279 |                      0          |               0           |                       0.695048 |                              0 |        0          |
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.0656137 |                       1.07279 |                      0          |              -4.0785e-05  |                       0.695048 |                              0 |        0          |
| bio_top3_subject_seq_pca72_rankdist_vehicle     | inf         |                       1.07279 |                      0          |               0           |                       0.695048 |                              0 |        0          |
| bio_top3_subject_seq_pca72_rankdist_vehicle_bio | inf         |                       1.07279 |                      0          |               0           |                       0.695048 |                              0 |        0          |
| bio_top3_subject_summary64_rankdist_vehicle     |   0.0528102 |                       1.07279 |                      0          |               0.000735422 |                       0.695048 |                              0 |        0.00579435 |
| bio_top3_subject_summary64_rankdist_vehicle     |   0.0367707 |                       1.07279 |                      0          |               0.000836236 |                       0.695048 |                              0 |        0.00659899 |
| bio_top3_subject_summary64_rankdist_vehicle     |   0.0297261 |                       1.07279 |                      0          |               0.00077042  |                       0.695048 |                              0 |        0.00774954 |
| bio_top3_subject_summary64_rankdist             |   0.0532208 |                       1.07279 |                      0          |               0.00108593  |                       0.695048 |                              0 |        0.00855597 |
| bio_top3_subject_seq_pca72_rankdist_vehicle_bio |   0.0549604 |                       1.07279 |                      0          |               0.00154581  |                       0.695048 |                              0 |        0.0121794  |
| bio_top3_subject_summary64_rankdist_vehicle_bio |   0.0318805 |                       1.07279 |                      0          |               0.00160446  |                       0.695048 |                              0 |        0.0176671  |
| bio_top3_subject_seq_pca72_rankdist             |   0.0500086 |                       1.0855  |                      0.0127131  |               0.00127543  |                       0.695048 |                              0 |        0.0227621  |
| bio_top3_subject_summary64_rankdist_vehicle_bio |   0.028302  |                       1.07279 |                      0          |               0.00343476  |                       0.695048 |                              0 |        0.0366379  |
| bio_top3_subject_summary64_rankdist             |   0.0360252 |                       1.07629 |                      0.00349909 |               0.00395825  |                       0.695048 |                              0 |        0.0375815  |
| bio_top3_subject_summary64_rankdist_vehicle     |   0.0265617 |                       1.07279 |                      0          |               0.00448049  |                       0.695048 |                              0 |        0.043416   |

## test diagnostic 排名前 18

| feature_set                                     |   threshold |   val_bad_top10_selected_rmse |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate |
|:------------------------------------------------|------------:|------------------------------:|--------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.0192481 |                       1.19463 |                       0.121841  |                 0.0162302 |                       0.684235 |                      -0.0108131  |                      0.105263  |
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.0208493 |                       1.19463 |                       0.121841  |                 0.0175821 |                       0.684235 |                      -0.0108131  |                      0.105263  |
| bio_top3_subject_seq_pca72_rankdist_vehicle_bio |   0.0246963 |                       1.22708 |                       0.154292  |                 0.0241534 |                       0.686803 |                      -0.00824551 |                      0.105263  |
| bio_top3_subject_seq_pca72_rankdist_vehicle_bio |   0.0272694 |                       1.15147 |                       0.0786858 |                 0.0151726 |                       0.686803 |                      -0.00824551 |                      0.105263  |
| bio_top3_subject_seq_pca72_rankdist_vehicle_bio |   0.0198493 |                       1.22708 |                       0.154292  |                 0.0289169 |                       0.686803 |                      -0.00824551 |                      0.105263  |
| bio_top3_subject_seq_pca72_rankdist_vehicle_bio |   0.0221685 |                       1.22708 |                       0.154292  |                 0.0285699 |                       0.686803 |                      -0.00824551 |                      0.105263  |
| bio_top3_subject_summary64_rankdist_vehicle_bio |   0.0170793 |                       1.16118 |                       0.0883943 |                 0.0143841 |                       0.689395 |                      -0.00565375 |                      0.0526316 |
| bio_top3_subject_summary64_rankdist_vehicle_bio |   0.0154383 |                       1.16118 |                       0.0883943 |                 0.0163726 |                       0.689395 |                      -0.00565375 |                      0.0526316 |
| bio_top3_subject_summary64_rankdist_vehicle_bio |   0.0183245 |                       1.0913  |                       0.0185124 |                 0.0106217 |                       0.689395 |                      -0.00565375 |                      0.0526316 |
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.0231377 |                       1.19463 |                       0.121841  |                 0.0147336 |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.0223803 |                       1.19463 |                       0.121841  |                 0.0159001 |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.030423  |                       1.1813  |                       0.108513  |                 0.0124956 |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.0248124 |                       1.19463 |                       0.121841  |                 0.0140767 |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.0260061 |                       1.1813  |                       0.108513  |                 0.0110036 |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.0169588 |                       1.22061 |                       0.147827  |                 0.0199653 |                       0.693404 |                      -0.00164449 |                      0.157895  |
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.016122  |                       1.22061 |                       0.147827  |                 0.0202277 |                       0.693404 |                      -0.00164449 |                      0.157895  |
| bio_top3_subject_seq_pca72_rankdist_vehicle     |   0.017623  |                       1.21039 |                       0.137605  |                 0.0187222 |                       0.693404 |                      -0.00164449 |                      0.157895  |
| bio_top3_subject_summary64_rankdist             |   0.0144469 |                       1.1798  |                       0.107009  |                 0.0308953 |                       0.6935   |                      -0.00154885 |                      0.210526  |

## 产物

- `figures\v281_test_badtop10_bio_top3_selector.png`
- `tables/v281_bio_top3_candidates.csv`
- `tables/v281_feature_set_audit.csv`
- `tables/v281_predictions.csv`
- `tables/v281_threshold_search.csv`
- `tables/v281_selected_by_strategy.csv`
- `tables/v281_chosen_configs.csv`
- `tables/v281_decision_summary.csv`
- `logs/guardrail_check.json`
