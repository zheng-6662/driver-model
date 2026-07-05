# v271 calibrated raw physiology state

## 本轮目的

- v270 的 raw waveform latent 仍未突破 fixed wait-latest。
- v271 把 raw 生理从绝对值改为相对 subject / recording 无标签基线的状态变化。
- 这是 calibrated / transductive setting，不是纯 cold-start subject-disjoint；只用于判断个体基线是否能释放生理信息。

## 特征集

| raw_set                   |   feature_n |   subject_summary_n |   recording_summary_n |   subject_seq_pca_n |   subject_seq_diff_pca_n |   recording_seq_pca_n |   recording_seq_diff_pca_n |   behavior_eta_max_mean |   identity_eta_max_mean |   identity_to_behavior_ratio_median |
|:--------------------------|------------:|--------------------:|----------------------:|--------------------:|-------------------------:|----------------------:|---------------------------:|------------------------:|------------------------:|------------------------------------:|
| subject_summary64         |          64 |                  64 |                     0 |                   0 |                        0 |                     0 |                          0 |              0.00564221 |               0.18942   |                            37.6941  |
| recording_summary64       |          64 |                   0 |                    64 |                   0 |                        0 |                     0 |                          0 |              0.00546261 |               0.062182  |                            13.3725  |
| subject_seq_pca72         |          72 |                   0 |                     0 |                  48 |                       24 |                     0 |                          0 |              0.00365873 |               0.129357  |                            27.7506  |
| recording_seq_pca72       |          72 |                   0 |                     0 |                   0 |                        0 |                    48 |                         24 |              0.00375531 |               0.133097  |                            21.8042  |
| calibrated_screened64     |          64 |                  12 |                    25 |                   7 |                        1 |                    19 |                          0 |              0.00693037 |               0.0444822 |                             6.58732 |
| calibrated_low_identity48 |          48 |                   9 |                    10 |                   0 |                        0 |                    29 |                          0 |              0.00316252 |               0.0248705 |                             9.23949 |

## test bad_top10 决策收口

| source                    | label                                                 |     rmse |   delta_vs_fixed_latest | passes_fixed_latest   |
|:--------------------------|:------------------------------------------------------|---------:|------------------------:|:----------------------|
| baseline                  | policy_keep_0ms_anchor                                | 1.19771  |             0.502658    | False                 |
| baseline                  | policy_wait_to_latest_anchor                          | 0.695048 |             4.15347e-07 | False                 |
| baseline                  | oracle_best_anchor_upper_bound                        | 0.612475 |            -0.0825726   | True                  |
| wait_test_best            | wait_raw_recording_summary64_gain                     | 0.695048 |             4.15347e-07 | False                 |
| pair_candidate_oracle     | subject_summary64:pair_candidate_oracle_k40           | 0.616603 |            -0.0784452   | True                  |
| pair_test_best_deployable | subject_seq_pca72:pair_vehicle_bio_badweighted_hgb_k5 | 0.7853   |             0.0902517   | False                 |
| pair_val_best_vehicle_raw | calibrated_low_identity48:pair_vehicle_bio_hgb_k40    | 0.92321  |             0.228162    | False                 |

## wait gate test bad_top10 top

| strategy                                                    | strategy_family   |   selected_tail_rmse_mean |   delta_selected_minus_latest_mean |   selected_latest_rate |
|:------------------------------------------------------------|:------------------|--------------------------:|-----------------------------------:|-----------------------:|
| oracle_best_anchor_upper_bound                              | oracle            |                  0.612475 |                          -0.082573 |               0.368421 |
| policy_wait_to_latest_anchor                                | baseline          |                  0.695048 |                           0        |               1        |
| wait_raw_recording_summary64_gain                           | raw_bio           |                  0.695048 |                           0        |               1        |
| wait_raw_recording_seq_pca72_gain                           | raw_bio           |                  0.695048 |                           0        |               1        |
| wait_vehicle_raw_subject_seq_pca72_gain_badweighted         | vehicle_raw       |                  0.695048 |                           0        |               1        |
| wait_vehicle_raw_subject_seq_pca72_gain                     | vehicle_raw       |                  0.695048 |                           0        |               1        |
| wait_raw_subject_seq_pca72_gain                             | raw_bio           |                  0.695048 |                           0        |               1        |
| wait_vehicle_raw_recording_summary64_gain_badweighted       | vehicle_raw       |                  0.695048 |                           0        |               1        |
| wait_vehicle_raw_recording_seq_pca72_gain_badweighted       | vehicle_raw       |                  0.695048 |                           0        |               1        |
| wait_vehicle_raw_recording_seq_pca72_gain                   | vehicle_raw       |                  0.695048 |                           0        |               1        |
| wait_raw_calibrated_screened64_gain                         | raw_bio           |                  0.695048 |                           0        |               1        |
| wait_vehicle_raw_calibrated_low_identity48_gain_badweighted | vehicle_raw       |                  0.695048 |                           0        |               1        |

## pair reranker test bad_top10 top

| raw_set                   | strategy                       | strategy_family   |   selected_tail_rmse_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |   selected_latest_rate |
|:--------------------------|:-------------------------------|:------------------|--------------------------:|-----------------------------------:|-------------------------:|-----------------------:|
| subject_summary64         | oracle_best_anchor_upper_bound | oracle            |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| recording_summary64       | oracle_best_anchor_upper_bound | oracle            |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| recording_seq_pca72       | oracle_best_anchor_upper_bound | oracle            |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| subject_seq_pca72         | oracle_best_anchor_upper_bound | oracle            |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| calibrated_low_identity48 | oracle_best_anchor_upper_bound | oracle            |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| calibrated_screened64     | oracle_best_anchor_upper_bound | oracle            |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| subject_seq_pca72         | pair_candidate_oracle_k40      | candidate_oracle  |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| recording_seq_pca72       | pair_candidate_oracle_k40      | candidate_oracle  |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| calibrated_low_identity48 | pair_candidate_oracle_k40      | candidate_oracle  |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| calibrated_screened64     | pair_candidate_oracle_k40      | candidate_oracle  |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| recording_summary64       | pair_candidate_oracle_k40      | candidate_oracle  |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| subject_summary64         | pair_candidate_oracle_k40      | candidate_oracle  |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| subject_summary64         | pair_candidate_oracle_k20      | candidate_oracle  |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| recording_summary64       | pair_candidate_oracle_k20      | candidate_oracle  |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| recording_seq_pca72       | pair_candidate_oracle_k20      | candidate_oracle  |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| subject_seq_pca72         | pair_candidate_oracle_k20      | candidate_oracle  |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| calibrated_screened64     | pair_candidate_oracle_k20      | candidate_oracle  |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| calibrated_low_identity48 | pair_candidate_oracle_k20      | candidate_oracle  |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |

## val 选择 vehicle+raw 策略

| chosen_label              | chosen_strategy          | chosen_family   |   val_bad_top10_rmse |   val_bad_top10_delay_ms_mean | split   | event_group   |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |   selected_latest_rate |   improve_rate_vs_keep0 | raw_set                   |   raw_feature_n |
|:--------------------------|:-------------------------|:----------------|---------------------:|------------------------------:|:--------|:--------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|-------------------------:|-----------------------:|------------------------:|:--------------------------|----------------:|
| val_best_pair_vehicle_bio | pair_vehicle_bio_hgb_k40 | vehicle_bio     |               1.4318 |                        638.71 | val     | bad_top10     |  31 |                   1.4318  |                         -0.739771 |                           0.359009 |                  638.71  |               0.129032 |                0.935484 | calibrated_low_identity48 |              48 |
| val_best_pair_vehicle_bio | pair_vehicle_bio_hgb_k40 | vehicle_bio     |               1.4318 |                        638.71 | test    | bad_top10     |  19 |                   0.92321 |                         -0.274497 |                           0.228161 |                  484.211 |               0.210526 |                0.894737 | calibrated_low_identity48 |              48 |

## 判读

- 当前 calibrated raw 生理可部署策略仍未低于 fixed wait-latest，不能称为差样本本质改善。
- 最好可部署策略 `wait_raw_recording_summary64_gain` 的 test bad_top10 RMSE 为 `0.6950`。
- 若 calibrated setting 仍失败，说明当前生理即使有个体基线也难以支撑该预测任务；应转回车辆多未来/不确定性主线。

## 关键图

- `figures\v271_test_badtop10_decision_summary.png`