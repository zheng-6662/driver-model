# v291 multi-signal physiology supervised probe

## 本轮目的

- v288/v289/v290 已经排除 ECG、RESP、EDA 单路源信号 distance/rerank/gate 的可部署改善。
- v291 把三路源信号合并，改做严格监督探针：看生理是否能识别差样本、识别方法池有收益样本，并帮助选择现成预测方法。
- 仍然执行 train 训练、val 选阈值、test 只报告。

## route decision

| check                                                     | requirement                                                     | pass   |    evidence | deployable   | route_viable_now   |
|:----------------------------------------------------------|:----------------------------------------------------------------|:-------|------------:|:-------------|:-------------------|
| deployable_val_noharm_selector_beats_latest_bad_top10     | val no-harm active selector 在 test bad_top10 上低于 latest     | False  | nan         | True         | False              |
| deployable_val_noharm_selector_beats_latest_bad_ambiguous | 同一 selector 在 test bad_top10_vehicle_ambiguous 上低于 latest | False  | nan         | True         | False              |
| method_pool_oracle_has_enough_headroom                    | 现成方法池的事后 oracle 在 test bad_top10 至少改善 0.03 RMSE    | True   |  -0.0402458 | False        | False              |
| bio_classifier_badtop10_auc_gt_060                        | 源生理特征至少能在 test 上识别 bad_top10，AUC > 0.60            | False  |   0.539394  | False        | False              |

## method pool 上限

| event_group                 | method                     |   n |   rmse_mean |   delta_vs_latest_mean |   beats_latest_rate |
|:----------------------------|:---------------------------|----:|------------:|-----------------------:|--------------------:|
| all                         | latest                     | 184 |    0.304615 |              0         |          nan        |
| all                         | listrank_vehicle           | 184 |    0.361142 |              0.0565261 |            0.255435 |
| all                         | listrank_vehicle_bio       | 184 |    0.347658 |              0.043042  |            0.244565 |
| all                         | listrank_vehicle_style_bio | 184 |    0.357339 |              0.052724  |            0.222826 |
| all                         | oracle_best_of_methods     | 184 |    0.277629 |             -0.0269864 |            0.347826 |
| bad_top10                   | latest                     |  19 |    0.695048 |              0         |          nan        |
| bad_top10                   | listrank_vehicle           |  19 |    0.840349 |              0.1453    |            0.210526 |
| bad_top10                   | listrank_vehicle_bio       |  19 |    0.779034 |              0.0839859 |            0.263158 |
| bad_top10                   | listrank_vehicle_style_bio |  19 |    0.833619 |              0.138571  |            0.157895 |
| bad_top10                   | oracle_best_of_methods     |  19 |    0.654803 |             -0.0402458 |            0.368421 |
| bad_top10_vehicle_ambiguous | latest                     |  15 |    0.744423 |              0         |          nan        |
| bad_top10_vehicle_ambiguous | listrank_vehicle           |  15 |    0.896034 |              0.151611  |            0.2      |
| bad_top10_vehicle_ambiguous | listrank_vehicle_bio       |  15 |    0.818369 |              0.073946  |            0.266667 |
| bad_top10_vehicle_ambiguous | listrank_vehicle_style_bio |  15 |    0.877352 |              0.132929  |            0.133333 |
| bad_top10_vehicle_ambiguous | oracle_best_of_methods     |  15 |    0.701899 |             -0.0425238 |            0.4      |

## validation 选择出的 selector

| chosen_type                         | selector_tag                        |   threshold | feature_block       | model_name     |   feature_n |   val_bad_top10_delta_vs_latest_mean |   val_all_delta_vs_latest_mean |   test_bad_top10_delta_vs_latest_mean |   test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean |   test_bad_top10_override_rate |
|:------------------------------------|:------------------------------------|------------:|:--------------------|:---------------|------------:|-------------------------------------:|-------------------------------:|--------------------------------------:|--------------------------------------------------------:|-------------------------------:|
| fallback_no_override                | vehicle_scores_only__ridge_a1       | inf         | vehicle_scores_only | ridge_a1       |          13 |                           0          |                     0          |                            0          |                                               0         |                      0         |
| test_best_diagnostic_not_deployable | all_listrank_scores__extra_trees_d6 |   0.0575993 | all_listrank_scores | extra_trees_d6 |          39 |                          -0.00103724 |                     0.00476764 |                           -0.00925823 |                                              -0.0117271 |                      0.0526316 |

## test diagnostic top selector

| selector_tag                             |   threshold | feature_block                 | model_name     |   feature_n |   val_bad_top10_delta_vs_latest_mean |   val_all_delta_vs_latest_mean |   test_bad_top10_delta_vs_latest_mean |   test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean |   test_bad_top10_override_rate | noharm_val   |
|:-----------------------------------------|------------:|:------------------------------|:---------------|------------:|-------------------------------------:|-------------------------------:|--------------------------------------:|--------------------------------------------------------:|-------------------------------:|:-------------|
| all_listrank_scores__extra_trees_d6      |   0.0575993 | all_listrank_scores           | extra_trees_d6 |          39 |                          -0.00103724 |                     0.00476764 |                           -0.00925823 |                                              -0.0117271 |                      0.0526316 | False        |
| all_listrank_scores__ridge_a1            |   0.0454856 | all_listrank_scores           | ridge_a1       |          39 |                           0.265627   |                     0.0598758  |                            0.00393361 |                                               0.0134367 |                      0.157895  | False        |
| vehicle_scores_only__extra_trees_d6      |   0.0683864 | vehicle_scores_only           | extra_trees_d6 |          13 |                           0          |                     0.00186157 |                            0.0090851  |                                               0.0115078 |                      0.0526316 | True         |
| bio_source_all_top__ridge_a1             |   0.0645403 | bio_source_all_top            | ridge_a1       |         180 |                           0.104749   |                     0.0122328  |                            0.00971177 |                                               0.0123016 |                      0.210526  | False        |
| all_listrank_scores__ridge_a1            |   0.053863  | all_listrank_scores           | ridge_a1       |          39 |                           0.0546147  |                     0.0303195  |                            0.0106079  |                                               0.0134367 |                      0.105263  | False        |
| all_listrank_scores__ridge_a1            |   0.0497785 | all_listrank_scores           | ridge_a1       |          39 |                           0.194004   |                     0.0479161  |                            0.0106079  |                                               0.0134367 |                      0.105263  | False        |
| all_listrank_scores__ridge_a1            |   0.0468905 | all_listrank_scores           | ridge_a1       |          39 |                           0.265627   |                     0.0596422  |                            0.0106079  |                                               0.0134367 |                      0.105263  | False        |
| vehicle_scores_plus_bio_all__ridge_a10   |   0.0725645 | vehicle_scores_plus_bio_all   | ridge_a10      |         193 |                           0.030853   |                     0.00905402 |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| vehicle_scores_only__ridge_a1            |   0.0572539 | vehicle_scores_only           | ridge_a1       |          13 |                           0.0546147  |                     0.0204323  |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| vehicle_scores_only__ridge_a10           |   0.0574974 | vehicle_scores_only           | ridge_a10      |          13 |                           0.0546147  |                     0.0204323  |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| vehicle_scores_plus_bio_all__ridge_a1    |   0.0759131 | vehicle_scores_plus_bio_all   | ridge_a1       |         193 |                           0.055652   |                     0.0126166  |                            0.0198662  |                                               0.0251638 |                      0.105263  | False        |
| vehicle_scores_plus_bio_all__ridge_a10   |   0.0651961 | vehicle_scores_plus_bio_all   | ridge_a10      |         193 |                           0.055652   |                     0.0189289  |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| vehicle_scores_plus_bio_lowid__ridge_a1  |   0.0691533 | vehicle_scores_plus_bio_lowid | ridge_a1       |         173 |                           0.055652   |                     0.0185144  |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| vehicle_scores_plus_bio_lowid__ridge_a1  |   0.0762272 | vehicle_scores_plus_bio_lowid | ridge_a1       |         173 |                           0.055652   |                     0.0095932  |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| vehicle_scores_plus_bio_lowid__ridge_a10 |   0.0750474 | vehicle_scores_plus_bio_lowid | ridge_a10      |         173 |                           0.055652   |                     0.0112981  |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| vehicle_scores_plus_bio_lowid__ridge_a1  |   0.0633036 | vehicle_scores_plus_bio_lowid | ridge_a1       |         173 |                           0.0599575  |                     0.0237079  |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| vehicle_scores_plus_bio_lowid__ridge_a10 |   0.0592246 | vehicle_scores_plus_bio_lowid | ridge_a10      |         173 |                           0.0599575  |                     0.0275853  |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| vehicle_scores_plus_bio_lowid__ridge_a10 |   0.0618403 | vehicle_scores_plus_bio_lowid | ridge_a10      |         173 |                           0.0599575  |                     0.0252847  |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| vehicle_scores_plus_bio_lowid__ridge_a10 |   0.0670206 | vehicle_scores_plus_bio_lowid | ridge_a10      |         173 |                           0.0599575  |                     0.0224954  |                            0.0198662  |                                               0.0251638 |                      0.0526316 | False        |
| bio_source_lowid_top__ridge_a1           |   0.0679797 | bio_source_lowid_top          | ridge_a1       |         160 |                           0.0689845  |                     0.00986676 |                            0.0198662  |                                               0.0251638 |                      0.105263  | False        |

## 分类探针

| target                      | split   | feature_block                    | model_name      |   n |   positive_rate |      auc |   average_precision |   feature_n |
|:----------------------------|:--------|:---------------------------------|:----------------|----:|----------------:|---------:|--------------------:|------------:|
| bad_top10                   | test    | bio_source_lowid_top             | extra_trees_cls | 184 |       0.103261  | 0.539394 |           0.132182  |         160 |
| bad_top10                   | test    | bio_source_all_top               | extra_trees_cls | 184 |       0.103261  | 0.522488 |           0.11815   |         180 |
| bad_top10                   | test    | all_listrank_scores              | logreg_balanced | 184 |       0.103261  | 0.481021 |           0.154621  |          39 |
| bad_top10                   | test    | bio_source_lowid_top             | logreg_balanced | 184 |       0.103261  | 0.444338 |           0.0990017 |         160 |
| bad_top10                   | test    | all_listrank_scores              | extra_trees_cls | 184 |       0.103261  | 0.434131 |           0.12049   |          39 |
| bad_top10                   | val     | bio_source_lowid_top             | logreg_balanced | 309 |       0.100324  | 0.583546 |           0.13596   |         160 |
| bad_top10                   | val     | bio_source_lowid_top             | extra_trees_cls | 309 |       0.100324  | 0.528545 |           0.104851  |         160 |
| bad_top10                   | val     | vehicle_scores_plus_bio_lowid    | logreg_balanced | 309 |       0.100324  | 0.507774 |           0.101902  |         173 |
| bad_top10                   | val     | bio_source_all_top               | logreg_balanced | 309 |       0.100324  | 0.507194 |           0.130831  |         180 |
| bad_top10                   | val     | vehicle_scores_plus_bio_lowid    | extra_trees_cls | 309 |       0.100324  | 0.487468 |           0.0940704 |         173 |
| bad_top10_vehicle_ambiguous | test    | all_listrank_scores              | logreg_balanced | 184 |       0.0815217 | 0.424458 |           0.084373  |          39 |
| bad_top10_vehicle_ambiguous | test    | bio_source_all_top               | logreg_balanced | 184 |       0.0815217 | 0.424063 |           0.081054  |         180 |
| bad_top10_vehicle_ambiguous | test    | all_listrank_scores_plus_bio_all | logreg_balanced | 184 |       0.0815217 | 0.395266 |           0.0833483 |         219 |
| bad_top10_vehicle_ambiguous | test    | bio_source_lowid_top             | extra_trees_cls | 184 |       0.0815217 | 0.386193 |           0.0681431 |         160 |
| bad_top10_vehicle_ambiguous | test    | bio_source_lowid_top             | logreg_balanced | 184 |       0.0815217 | 0.381065 |           0.0714814 |         160 |
| bad_top10_vehicle_ambiguous | val     | bio_source_lowid_top             | logreg_balanced | 309 |       0.0873786 | 0.579722 |           0.115438  |         160 |
| bad_top10_vehicle_ambiguous | val     | vehicle_scores_plus_bio_lowid    | logreg_balanced | 309 |       0.0873786 | 0.526005 |           0.0947049 |         173 |
| bad_top10_vehicle_ambiguous | val     | bio_source_lowid_top             | extra_trees_cls | 309 |       0.0873786 | 0.464145 |           0.081915  |         160 |
| bad_top10_vehicle_ambiguous | val     | vehicle_scores_plus_bio_all      | logreg_balanced | 309 |       0.0873786 | 0.443656 |           0.124719  |         193 |
| bad_top10_vehicle_ambiguous | val     | all_listrank_scores              | extra_trees_cls | 309 |       0.0873786 | 0.410428 |           0.0753823 |          39 |
| method_oracle_gain_gt_002   | test    | all_listrank_scores              | extra_trees_cls | 184 |       0.288043  | 0.591531 |           0.327555  |          39 |
| method_oracle_gain_gt_002   | test    | all_listrank_scores_plus_bio_all | extra_trees_cls | 184 |       0.288043  | 0.589227 |           0.336021  |         219 |
| method_oracle_gain_gt_002   | test    | vehicle_scores_plus_bio_lowid    | logreg_balanced | 184 |       0.288043  | 0.565606 |           0.305471  |         173 |
| method_oracle_gain_gt_002   | test    | vehicle_scores_only              | extra_trees_cls | 184 |       0.288043  | 0.550194 |           0.329889  |          13 |
| method_oracle_gain_gt_002   | test    | vehicle_scores_plus_bio_all      | logreg_balanced | 184 |       0.288043  | 0.548754 |           0.315616  |         193 |
| method_oracle_gain_gt_002   | val     | all_listrank_scores_plus_bio_all | logreg_balanced | 309 |       0.275081  | 0.58771  |           0.320087  |         219 |
| method_oracle_gain_gt_002   | val     | vehicle_scores_plus_bio_all      | logreg_balanced | 309 |       0.275081  | 0.572374 |           0.330826  |         193 |
| method_oracle_gain_gt_002   | val     | bio_source_all_top               | extra_trees_cls | 309 |       0.275081  | 0.544748 |           0.305659  |         180 |
| method_oracle_gain_gt_002   | val     | vehicle_scores_plus_bio_all      | extra_trees_cls | 309 |       0.275081  | 0.536239 |           0.313169  |         193 |
| method_oracle_gain_gt_002   | val     | vehicle_scores_plus_bio_lowid    | extra_trees_cls | 309 |       0.275081  | 0.533666 |           0.305858  |         173 |
| vehicle_ambiguous           | test    | bio_source_lowid_top             | logreg_balanced | 184 |       0.706522  | 0.565385 |           0.75424   |         160 |
| vehicle_ambiguous           | test    | vehicle_scores_plus_bio_lowid    | logreg_balanced | 184 |       0.706522  | 0.564245 |           0.754753  |         173 |
| vehicle_ambiguous           | test    | vehicle_scores_plus_bio_lowid    | extra_trees_cls | 184 |       0.706522  | 0.526781 |           0.737816  |         173 |
| vehicle_ambiguous           | test    | bio_source_lowid_top             | extra_trees_cls | 184 |       0.706522  | 0.525926 |           0.745207  |         160 |
| vehicle_ambiguous           | test    | bio_source_all_top               | extra_trees_cls | 184 |       0.706522  | 0.504701 |           0.738987  |         180 |
| vehicle_ambiguous           | val     | vehicle_scores_plus_bio_lowid    | logreg_balanced | 309 |       0.708738  | 0.595535 |           0.786585  |         173 |
| vehicle_ambiguous           | val     | bio_source_lowid_top             | logreg_balanced | 309 |       0.708738  | 0.579655 |           0.777871  |         160 |
| vehicle_ambiguous           | val     | bio_source_lowid_top             | extra_trees_cls | 309 |       0.708738  | 0.575495 |           0.772445  |         160 |
| vehicle_ambiguous           | val     | vehicle_scores_plus_bio_all      | logreg_balanced | 309 |       0.708738  | 0.560781 |           0.764319  |         193 |
| vehicle_ambiguous           | val     | vehicle_scores_plus_bio_lowid    | extra_trees_cls | 309 |       0.708738  | 0.557585 |           0.756797  |         173 |

## 源生理特征筛选概况

| feature                                                | source   |   finite_rate_train |   behavior_corr_max |   identity_eta | low_identity_candidate   |
|:-------------------------------------------------------|:---------|--------------------:|--------------------:|---------------:|:-------------------------|
| bio288_w_dur3_endm1_ecg_rr_plausible_rate              | ecg      |            0.888724 |            0.170252 |     0.62755    | True                     |
| bio288_w_pre20_pre10_ecg_rr_plausible_rate             | ecg      |            0.888724 |            0.166411 |     0.755249   | False                    |
| bio288_w_dur3_endm0p5_ecg_peak_n                       | ecg      |            0.888724 |            0.159355 |     0.694468   | False                    |
| bio288_w_dur3_endm0p5_ecg_peak_rate_per_s              | ecg      |            0.888724 |            0.159347 |     0.694463   | False                    |
| bio288_w_dur3_endm1_ecg_peak_n                         | ecg      |            0.888724 |            0.156979 |     0.683127   | False                    |
| bio288_w_dur3_endm1_ecg_peak_rate_per_s                | ecg      |            0.888724 |            0.156977 |     0.683114   | False                    |
| bio288_w_dur2_endm0p5_ecg_peak_n                       | ecg      |            0.888724 |            0.152442 |     0.603279   | True                     |
| bio288_w_dur2_endm0p5_ecg_peak_rate_per_s              | ecg      |            0.888724 |            0.152428 |     0.60327    | True                     |
| bio288_ecg_polarity_code                               | ecg      |            0.888724 |            0.152006 |     0.724818   | False                    |
| bio290_w_pre30_pre20_eda_phasic_peak_n                 | eda      |            0.888724 |            0.149309 |     0.487103   | True                     |
| bio290_w_pre30_pre20_eda_phasic_peak_rate              | eda      |            0.888724 |            0.149213 |     0.487324   | True                     |
| bio289_w_dur2_end0_resp_z_min                          | resp     |            0.888724 |            0.148288 |     0.0450345  | True                     |
| bio288_w_dur3_endm0p5_ecg_rr_plausible_rate            | ecg      |            0.888724 |            0.147362 |     0.614157   | False                    |
| bio289_w_dur2_endm0p5_resp_z_abs_mean                  | resp     |            0.888724 |            0.147104 |     0.0249406  | True                     |
| bio289_w_dur3_endm0p5_resp_z_abs_mean                  | resp     |            0.888724 |            0.145703 |     0.037559   | True                     |
| bio288_w_dur1_end0_ecg_peak_score_p90                  | ecg      |            0.888724 |            0.144914 |     0.598722   | False                    |
| bio288_w_dur1_end0_ecg_rr_plausible_rate               | ecg      |            0.606825 |            0.144534 |     0.243098   | True                     |
| bio288_w_dur3_endm1_ecg_rr_rmssd                       | ecg      |            0.716617 |            0.144041 |     0.23364    | True                     |
| bio288_w_dur2_endm1_ecg_peak_n                         | ecg      |            0.888724 |            0.143623 |     0.591149   | False                    |
| bio288_w_dur2_endm1_ecg_peak_rate_per_s                | ecg      |            0.888724 |            0.143617 |     0.591126   | False                    |
| bio289_w_dur2_endm1_resp_z_abs_mean                    | resp     |            0.888724 |            0.142728 |     0.0328967  | True                     |
| bio288_w_dur5_end0_ecg_rr_plausible_rate               | ecg      |            0.888724 |            0.142617 |     0.701333   | False                    |
| bio288_w_dur1_endm0p5_ecg_peak_rate_per_s              | ecg      |            0.888724 |            0.141775 |     0.398786   | True                     |
| bio288_w_dur1_endm0p5_ecg_peak_n                       | ecg      |            0.888724 |            0.141775 |     0.398786   | True                     |
| bio288_w_pre10_pre5_ecg_rr_plausible_rate              | ecg      |            0.888724 |            0.141028 |     0.73096    | False                    |
| bio288_w_dur3_end0_ecg_rr_plausible_rate               | ecg      |            0.888724 |            0.140395 |     0.593965   | False                    |
| bio288_baseline_std                                    | ecg      |            0.888724 |            0.14024  |     0.881216   | False                    |
| bio288_w_dur2_endm1_ecg_rr_plausible_rate              | ecg      |            0.888724 |            0.138857 |     0.500909   | True                     |
| bio288_w_dur2_endm2_ecg_rr_plausible_rate              | ecg      |            0.885757 |            0.138828 |     0.502172   | True                     |
| bio288_w_dur1_endm1_ecg_peak_score_median              | ecg      |            0.888724 |            0.137662 |     0.608522   | False                    |
| bio289_delta_dur5_end0_minus_dur5_endm2_resp_z_std     | resp     |            0.888724 |            0.136959 |     0.0171658  | True                     |
| bio289_delta_dur10_end0_minus_pre10_pre5_resp_z_mean   | resp     |            0.888724 |            0.136555 |     0.00927093 | True                     |
| bio288_w_dur5_end0_ecg_peak_n                          | ecg      |            0.888724 |            0.136489 |     0.767424   | False                    |
| bio288_w_dur5_end0_ecg_peak_rate_per_s                 | ecg      |            0.888724 |            0.136485 |     0.767422   | False                    |
| bio288_w_pre20_pre10_ecg_peak_rate_per_s               | ecg      |            0.888724 |            0.13617  |     0.820095   | False                    |
| bio288_w_pre20_pre10_ecg_peak_n                        | ecg      |            0.888724 |            0.136154 |     0.820116   | False                    |
| bio288_w_dur3_end0_ecg_peak_n                          | ecg      |            0.888724 |            0.135637 |     0.676942   | False                    |
| bio288_w_dur3_end0_ecg_peak_rate_per_s                 | ecg      |            0.888724 |            0.135629 |     0.676932   | False                    |
| bio288_w_dur5_endm2_ecg_rr_plausible_rate              | ecg      |            0.888724 |            0.133833 |     0.721326   | False                    |
| bio288_delta_dur2_end0_minus_dur2_endm2_ecg_z_abs_mean | ecg      |            0.888724 |            0.133229 |     0.0127722  | True                     |

## guardrail

```json
{
  "pass": true,
  "event_n": 1167,
  "train_n": 674,
  "val_n": 309,
  "test_n": 184,
  "bio_source_feature_n": 1660,
  "screen_feature_n": 1404,
  "feature_block_n": 7,
  "selector_config_n": 28,
  "route_viable_now": false,
  "method_pool_test_badtop10_oracle_delta": -0.04024575650691988,
  "best_val_noharm_active_exists": false,
  "best_deployable_test_badtop10_delta": null,
  "best_test_diagnostic_badtop10_delta": -0.009258226344459946,
  "test_used_for_feature_screen_or_threshold": false
}
```

## 判断

- v291 没有找到可部署的多信号生理监督 selector。
- 如果 method-pool oracle 有上限但 selector 学不到，说明现有生理源信号不足以稳定判断何时覆盖 latest。
- 如果分类 AUC 有弱信号但 selector 不改善，说明生理更适合做可观测性/不确定性分层，而不是直接选择预测方法。