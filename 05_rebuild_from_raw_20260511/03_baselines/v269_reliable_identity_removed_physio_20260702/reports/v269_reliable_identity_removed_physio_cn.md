# v269 reliable / identity-removed physiology

## 本轮目的

- v268 显示当前派生生理表征存在不可用列和强身份混淆。
- v269 不继续堆模型，而是先把生理特征改成可靠、动态变化、低身份混淆的候选集合。
- 然后在 wait gate 与 pair reranker 两个可部署任务上验证是否真正改善 test bad_top10。

## 特征集审计

| bio_set                  |   feature_n |   delta_feature_n |   behavior_eta_max_mean |   identity_eta_max_mean |   identity_to_behavior_ratio_median |   finite_rate_train_mean |
|:-------------------------|------------:|------------------:|------------------------:|------------------------:|------------------------------------:|-------------------------:|
| reliable_top64           |          64 |                38 |              0.00775981 |               0.0905094 |                             12.7838 |                 0.886614 |
| dynamic_top48            |          48 |                48 |              0.0058569  |               0.08797   |                             16.6751 |                 0.888724 |
| low_identity_top32       |          32 |                23 |              0.00355112 |               0.0596864 |                             27.8206 |                 0.881167 |
| combo_identity_removed64 |          64 |                48 |              0.00513084 |               0.0830461 |                             17.4229 |                 0.884945 |

## test bad_top10 决策收口

| source                    | label                                                        |     rmse |   delta_vs_fixed_latest | passes_fixed_latest   |
|:--------------------------|:-------------------------------------------------------------|---------:|------------------------:|:----------------------|
| baseline                  | policy_keep_0ms_anchor                                       | 1.19771  |             0.502658    | False                 |
| baseline                  | policy_wait_to_latest_anchor                                 | 0.695048 |             4.15347e-07 | False                 |
| baseline                  | oracle_best_anchor_upper_bound                               | 0.612475 |            -0.0825726   | True                  |
| wait_gate_test_best       | wait_vehicle_gain_badweighted                                | 0.695048 |             4.15347e-07 | False                 |
| pair_candidate_oracle     | reliable_top64:pair_candidate_oracle_k40                     | 0.616603 |            -0.0784452   | True                  |
| pair_test_best_deployable | combo_identity_removed64:pair_base_hgb_k40                   | 0.778113 |             0.0830647   | False                 |
| pair_val_best_vehicle_bio | combo_identity_removed64:pair_vehicle_bio_badweighted_hgb_k5 | 0.836495 |             0.141447    | False                 |

## wait gate 关键结果

| strategy                                                   | strategy_family   |   n |   selected_tail_rmse_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |   selected_latest_rate |
|:-----------------------------------------------------------|:------------------|----:|--------------------------:|-----------------------------------:|-------------------------:|-----------------------:|
| oracle_best_anchor_upper_bound                             | oracle            |  19 |                  0.612475 |                       -0.082573    |                  818.421 |               0.368421 |
| policy_wait_to_latest_anchor                               | baseline          |  19 |                  0.695048 |                        0           |                 1000     |               1        |
| wait_bio_reliable_top64_gain                               | bio_only          |  19 |                  0.695048 |                        0           |                 1000     |               1        |
| wait_vehicle_gain_badweighted                              | vehicle_only      |  19 |                  0.695048 |                        0           |                 1000     |               1        |
| wait_vehicle_bio_reliable_top64_gain_badweighted           | vehicle_bio       |  19 |                  0.695048 |                        0           |                 1000     |               1        |
| wait_vehicle_bio_low_identity_top32_gain_badweighted       | vehicle_bio       |  19 |                  0.695048 |                        0           |                 1000     |               1        |
| wait_bio_combo_identity_removed64_gain                     | bio_only          |  19 |                  0.695048 |                        0           |                 1000     |               1        |
| wait_vehicle_bio_combo_identity_removed64_gain             | vehicle_bio       |  19 |                  0.695048 |                        0           |                 1000     |               1        |
| wait_vehicle_bio_low_identity_top32_gain                   | vehicle_bio       |  19 |                  0.695048 |                        0           |                 1000     |               1        |
| wait_vehicle_bio_combo_identity_removed64_gain_badweighted | vehicle_bio       |  19 |                  0.695048 |                        0           |                 1000     |               1        |
| wait_vehicle_gain                                          | vehicle_only      |  19 |                  0.695936 |                        0.000887764 |                  947.368 |               0.947368 |
| wait_vehicle_bio_dynamic_top48_gain_badweighted            | vehicle_bio       |  19 |                  0.695936 |                        0.000887764 |                  947.368 |               0.947368 |

## pair reranker 关键结果

| bio_set                  | strategy                       | strategy_family   |   n |   selected_tail_rmse_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |   selected_latest_rate |
|:-------------------------|:-------------------------------|:------------------|----:|--------------------------:|-----------------------------------:|-------------------------:|-----------------------:|
| reliable_top64           | oracle_best_anchor_upper_bound | oracle            |  19 |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| dynamic_top48            | oracle_best_anchor_upper_bound | oracle            |  19 |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| combo_identity_removed64 | oracle_best_anchor_upper_bound | oracle            |  19 |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| low_identity_top32       | oracle_best_anchor_upper_bound | oracle            |  19 |                  0.612475 |                         -0.082573  |                  818.421 |               0.368421 |
| low_identity_top32       | pair_candidate_oracle_k40      | candidate_oracle  |  19 |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| combo_identity_removed64 | pair_candidate_oracle_k40      | candidate_oracle  |  19 |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| dynamic_top48            | pair_candidate_oracle_k40      | candidate_oracle  |  19 |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| reliable_top64           | pair_candidate_oracle_k40      | candidate_oracle  |  19 |                  0.616603 |                         -0.0784456 |                  831.579 |               0.368421 |
| reliable_top64           | pair_candidate_oracle_k20      | candidate_oracle  |  19 |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| dynamic_top48            | pair_candidate_oracle_k20      | candidate_oracle  |  19 |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| combo_identity_removed64 | pair_candidate_oracle_k20      | candidate_oracle  |  19 |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| low_identity_top32       | pair_candidate_oracle_k20      | candidate_oracle  |  19 |                  0.625011 |                         -0.0700371 |                  813.158 |               0.315789 |
| low_identity_top32       | pair_candidate_oracle_k10      | candidate_oracle  |  19 |                  0.642559 |                         -0.0524897 |                  818.421 |               0.263158 |
| combo_identity_removed64 | pair_candidate_oracle_k10      | candidate_oracle  |  19 |                  0.642559 |                         -0.0524897 |                  818.421 |               0.263158 |
| dynamic_top48            | pair_candidate_oracle_k10      | candidate_oracle  |  19 |                  0.642559 |                         -0.0524897 |                  818.421 |               0.263158 |
| reliable_top64           | pair_candidate_oracle_k10      | candidate_oracle  |  19 |                  0.642559 |                         -0.0524897 |                  818.421 |               0.263158 |

## val 选择的 pair vehicle+bio 策略

| chosen_label              | chosen_strategy                     | chosen_family   |   val_bad_top10_rmse |   val_bad_top10_delay_ms_mean | split   | event_group   |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |   selected_latest_rate |   improve_rate_vs_keep0 | bio_set                  |   bio_feature_n |
|:--------------------------|:------------------------------------|:----------------|---------------------:|------------------------------:|:--------|:--------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|-------------------------:|-----------------------:|------------------------:|:-------------------------|----------------:|
| val_best_pair_vehicle_bio | pair_vehicle_bio_badweighted_hgb_k5 | vehicle_bio     |              1.30493 |                       662.903 | val     | bad_top10     |  31 |                  1.30493  |                         -0.866636 |                           0.232144 |                  662.903 |              0.129032  |                0.870968 | combo_identity_removed64 |              64 |
| val_best_pair_vehicle_bio | pair_vehicle_bio_badweighted_hgb_k5 | vehicle_bio     |              1.30493 |                       662.903 | test    | bad_top10     |  19 |                  0.836495 |                         -0.361211 |                           0.141447 |                  571.053 |              0.0526316 |                0.842105 | combo_identity_removed64 |              64 |

## 判读

- 当前可部署 v269 策略仍未低于 fixed wait-latest，因此还不能称为差样本本质改善。
- 最好可部署策略 `wait_vehicle_gain_badweighted` 的 test bad_top10 RMSE 为 `0.6950`。
- 这个最好策略实际退化为接近全 wait-latest，不是生理判断带来的新增收益。
- 若 v269 仍失败，说明问题不是简单特征筛选，而可能需要回到原始波形事件表示、更多驾驶员内校准，或承认当前 subject-disjoint 生理增量不足。

## 关键图

- `figures\v269_wait_gate_test_badtop10.png`
- `figures\v269_pair_reranker_test_badtop10.png`