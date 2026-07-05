# v270 raw physiology state latent

## 本轮目的

- v269 的事件 biomarker 特征筛选只带来很小改善。
- v270 直接从 v256 raw 20s 生理序列构造 raw-state latent，检验底层波形是否有更强区分信息。

## 特征集

| raw_set            |   feature_n |   summary_n |   fft_n |   pca_n |   diff_pca_n |   behavior_eta_max_mean |   identity_eta_max_mean |   identity_to_behavior_ratio_median |
|:-------------------|------------:|------------:|--------:|--------:|-------------:|------------------------:|------------------------:|------------------------------------:|
| raw_summary_fft    |         160 |          85 |      75 |       0 |            0 |              0.00372648 |               0.304349  |                            112.936  |
| raw_pca96          |          96 |           0 |       0 |      64 |           32 |              0.0033795  |               0.0759459 |                             29.427  |
| raw_screened64     |          64 |          18 |       4 |      29 |           13 |              0.00663582 |               0.0909649 |                             13.8975 |
| raw_low_identity48 |          48 |           6 |       0 |      28 |           14 |              0.00308339 |               0.0571374 |                             24.2686 |

## test bad_top10 决策收口

| source                    | label                                               |     rmse |   delta_vs_fixed_latest | passes_fixed_latest   |
|:--------------------------|:----------------------------------------------------|---------:|------------------------:|:----------------------|
| baseline                  | policy_keep_0ms_anchor                              | 1.19771  |             0.502658    | False                 |
| baseline                  | policy_wait_to_latest_anchor                        | 0.695048 |             4.15347e-07 | False                 |
| baseline                  | oracle_best_anchor_upper_bound                      | 0.612475 |            -0.0825726   | True                  |
| wait_test_best            | wait_raw_raw_summary_fft_gain                       | 0.695048 |             4.15347e-07 | False                 |
| pair_candidate_oracle     | raw_summary_fft:pair_candidate_oracle_k40           | 0.616603 |            -0.0784452   | True                  |
| pair_test_best_deployable | raw_screened64:pair_vehicle_bio_badweighted_hgb_k20 | 0.786589 |             0.0915405   | False                 |
| pair_val_best_vehicle_raw | raw_low_identity48:pair_vehicle_bio_hgb_k5          | 0.814154 |             0.119106    | False                 |

## wait gate test bad_top10 top

| strategy                                             | strategy_family   |   selected_tail_rmse_mean |   delta_selected_minus_latest_mean |   selected_latest_rate |
|:-----------------------------------------------------|:------------------|--------------------------:|-----------------------------------:|-----------------------:|
| oracle_best_anchor_upper_bound                       | oracle            |                  0.612475 |                         -0.082573  |               0.368421 |
| policy_wait_to_latest_anchor                         | baseline          |                  0.695048 |                          0         |               1        |
| wait_raw_raw_summary_fft_gain                        | raw_bio           |                  0.695048 |                          0         |               1        |
| wait_vehicle_raw_raw_summary_fft_gain                | vehicle_raw       |                  0.695048 |                          0         |               1        |
| wait_vehicle_raw_raw_pca96_gain                      | vehicle_raw       |                  0.695048 |                          0         |               1        |
| wait_vehicle_raw_raw_screened64_gain_badweighted     | vehicle_raw       |                  0.695048 |                          0         |               1        |
| wait_vehicle_raw_raw_screened64_gain                 | vehicle_raw       |                  0.695048 |                          0         |               1        |
| wait_vehicle_raw_raw_pca96_gain_badweighted          | vehicle_raw       |                  0.695048 |                          0         |               1        |
| wait_vehicle_raw_raw_low_identity48_gain_badweighted | vehicle_raw       |                  0.695048 |                          0         |               1        |
| wait_raw_raw_pca96_gain                              | raw_bio           |                  0.714975 |                          0.0199262 |               0.947368 |
| wait_vehicle_raw_raw_low_identity48_gain             | vehicle_raw       |                  0.718501 |                          0.0234527 |               0.947368 |
| wait_raw_raw_screened64_gain                         | raw_bio           |                  0.728266 |                          0.0332172 |               0.947368 |

## pair reranker test bad_top10 top

| raw_set            | strategy                       | strategy_family   |   selected_tail_rmse_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |   selected_latest_rate |
|:-------------------|:-------------------------------|:------------------|--------------------------:|-----------------------------------:|-------------------------:|-----------------------:|
| raw_summary_fft    | oracle_best_anchor_upper_bound | oracle            |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| raw_pca96          | oracle_best_anchor_upper_bound | oracle            |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| raw_low_identity48 | oracle_best_anchor_upper_bound | oracle            |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| raw_screened64     | oracle_best_anchor_upper_bound | oracle            |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| raw_screened64     | pair_candidate_oracle_k40      | candidate_oracle  |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| raw_low_identity48 | pair_candidate_oracle_k40      | candidate_oracle  |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| raw_pca96          | pair_candidate_oracle_k40      | candidate_oracle  |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| raw_summary_fft    | pair_candidate_oracle_k40      | candidate_oracle  |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| raw_summary_fft    | pair_candidate_oracle_k20      | candidate_oracle  |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| raw_pca96          | pair_candidate_oracle_k20      | candidate_oracle  |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| raw_low_identity48 | pair_candidate_oracle_k20      | candidate_oracle  |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| raw_screened64     | pair_candidate_oracle_k20      | candidate_oracle  |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| raw_screened64     | pair_candidate_oracle_k10      | candidate_oracle  |                  0.642559 |                         -0.0524897 |                  818.421 |               0.263158 |
| raw_low_identity48 | pair_candidate_oracle_k10      | candidate_oracle  |                  0.642559 |                         -0.0524897 |                  818.421 |               0.263158 |
| raw_pca96          | pair_candidate_oracle_k10      | candidate_oracle  |                  0.642559 |                         -0.0524897 |                  818.421 |               0.263158 |
| raw_summary_fft    | pair_candidate_oracle_k10      | candidate_oracle  |                  0.642559 |                         -0.0524897 |                  818.421 |               0.263158 |
| raw_summary_fft    | policy_wait_to_latest_anchor   | baseline          |                  0.695048 |                          0         |                 1000     |               1        |
| raw_pca96          | policy_wait_to_latest_anchor   | baseline          |                  0.695048 |                          0         |                 1000     |               1        |

## val 选择 vehicle+raw 策略

| chosen_label              | chosen_strategy         | chosen_family   |   val_bad_top10_rmse |   val_bad_top10_delay_ms_mean | split   | event_group   |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |   selected_latest_rate |   improve_rate_vs_keep0 | raw_set            |   raw_feature_n |
|:--------------------------|:------------------------|:----------------|---------------------:|------------------------------:|:--------|:--------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|-------------------------:|-----------------------:|------------------------:|:-------------------|----------------:|
| val_best_pair_vehicle_bio | pair_vehicle_bio_hgb_k5 | vehicle_bio     |              1.39528 |                       627.419 | val     | bad_top10     |  31 |                  1.39528  |                         -0.77629  |                           0.322491 |                  627.419 |              0.0967742 |                0.903226 | raw_low_identity48 |              48 |
| val_best_pair_vehicle_bio | pair_vehicle_bio_hgb_k5 | vehicle_bio     |              1.39528 |                       627.419 | test    | bad_top10     |  19 |                  0.814154 |                         -0.383552 |                           0.119106 |                  613.158 |              0         |                0.894737 | raw_low_identity48 |              48 |

## 判读

- 当前 raw 生理可部署策略仍未低于 fixed wait-latest，不能称为差样本本质改善。
- 最好可部署策略 `wait_raw_raw_summary_fft_gain` 的 test bad_top10 RMSE 为 `0.6950`。
- 若 raw latent 仍失败，subject-disjoint 生理路线的可迁移增量已经非常有限；继续应转 subject-aware 个体校准或回到车辆多未来主线。

## 关键图

- `figures\v270_test_badtop10_decision_summary.png`