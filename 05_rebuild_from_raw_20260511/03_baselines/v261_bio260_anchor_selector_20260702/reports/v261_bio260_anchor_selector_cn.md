# v261 bio260 事件级生理 anchor selector

## 本轮问题

- v260 说明事件级生理 biomarker 对 bad_top10 有少量风险识别能力，但直接预测未来行为不稳定。
- v261 不再把生理当成轨迹预测输入，而是只让它参与锚点/等待选择。
- 如果 bio260 selector 不能超过 vehicle selector，说明当前生理信号还不足以弥补锚点前车辆信息不足。

## 方法

- 候选锚点：复用 v247 的 50ms fine-grid 候选。
- 监督目标：复用 v247 的 `target_score_primary`，也就是候选锚点 replay 后的综合误差目标。
- 生理输入：使用 v260 从 200Hz 波形重构的 ECG/EDA/RESP/EMG 事件级 biomarker。
- 防泄漏：候选锚点为 50ms 粒度，生理按 floor coarse delay 合并，只使用不晚于候选锚点的生理窗口。
- 训练边界：只在 train split 拟合，val/test 只报告；oracle 只作上限。

## 合并与特征审计

|   candidate_rows |   event_n |   bio260_source_rows |   bio260_source_event_n |   bio260_feature_n |   bio260_merge_ok_rate |   bio260_feature_missing_rate_after_merge |   bio260_uses_post_observation_max |   candidate_delay_min |   candidate_delay_max |
|-----------------:|----------:|---------------------:|------------------------:|-------------------:|-----------------------:|------------------------------------------:|-----------------------------------:|----------------------:|----------------------:|
|            24507 |      1167 |                 7002 |                    1167 |                233 |               0.919452 |                                  0.256043 |                                  0 |                     0 |                  1000 |

| model                                   |   feature_n |   bio260_feature_n |
|:----------------------------------------|------------:|-------------------:|
| selector_vehicle_hgb                    |          35 |                  0 |
| selector_bio260_hgb                     |         234 |                234 |
| selector_vehicle_bio260_hgb             |         269 |                234 |
| selector_vehicle_bio260_badweighted_hgb |         269 |                234 |

## Test 关键结果

| event_group          | strategy                                |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   improve_rate_vs_keep0 |   selected_delay_ms_mean |
|:---------------------|:----------------------------------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|------------------------:|-------------------------:|
| all                  | policy_keep_0ms_anchor                  | 184 |                  0.475053 |                         0         |                          0.170438  |                0        |                    0     |
| all                  | policy_wait_to_latest_anchor            | 184 |                  0.304615 |                        -0.170438  |                          0         |                0.777174 |                 1000     |
| all                  | oracle_best_anchor_upper_bound          | 184 |                  0.23972  |                        -0.235333  |                         -0.0648958 |                0.951087 |                  711.957 |
| all                  | selector_vehicle_hgb                    | 184 |                  0.407466 |                        -0.0675872 |                          0.10285   |                0.570652 |                  423.913 |
| all                  | selector_bio260_hgb                     | 184 |                  0.40299  |                        -0.072063  |                          0.0983747 |                0.516304 |                  425     |
| all                  | selector_vehicle_bio260_hgb             | 184 |                  0.40624  |                        -0.068813  |                          0.101625  |                0.592391 |                  410.054 |
| all                  | selector_vehicle_bio260_badweighted_hgb | 184 |                  0.40561  |                        -0.0694434 |                          0.100994  |                0.603261 |                  432.065 |
| bad_top10            | policy_keep_0ms_anchor                  |  19 |                  1.19771  |                         0         |                          0.502658  |                0        |                    0     |
| bad_top10            | policy_wait_to_latest_anchor            |  19 |                  0.695048 |                        -0.502658  |                          0         |                1        |                 1000     |
| bad_top10            | oracle_best_anchor_upper_bound          |  19 |                  0.612475 |                        -0.585231  |                         -0.082573  |                1        |                  818.421 |
| bad_top10            | selector_vehicle_hgb                    |  19 |                  0.942475 |                        -0.255231  |                          0.247426  |                0.631579 |                  384.211 |
| bad_top10            | selector_bio260_hgb                     |  19 |                  1.01797  |                        -0.179732  |                          0.322926  |                0.421053 |                  284.211 |
| bad_top10            | selector_vehicle_bio260_hgb             |  19 |                  0.976487 |                        -0.221219  |                          0.281439  |                0.736842 |                  313.158 |
| bad_top10            | selector_vehicle_bio260_badweighted_hgb |  19 |                  0.98367  |                        -0.214036  |                          0.288622  |                0.684211 |                  281.579 |
| normal               | policy_keep_0ms_anchor                  | 104 |                  0.385937 |                         0         |                          0.159142  |                0        |                    0     |
| normal               | policy_wait_to_latest_anchor            | 104 |                  0.226795 |                        -0.159142  |                          0         |                0.740385 |                 1000     |
| normal               | oracle_best_anchor_upper_bound          | 104 |                  0.171435 |                        -0.214502  |                         -0.0553599 |                0.951923 |                  735.577 |
| normal               | selector_vehicle_hgb                    | 104 |                  0.319284 |                        -0.066653  |                          0.0924895 |                0.605769 |                  437.5   |
| normal               | selector_bio260_hgb                     | 104 |                  0.314461 |                        -0.0714758 |                          0.0876667 |                0.576923 |                  486.538 |
| normal               | selector_vehicle_bio260_hgb             | 104 |                  0.313005 |                        -0.0729317 |                          0.0862107 |                0.625    |                  433.173 |
| normal               | selector_vehicle_bio260_badweighted_hgb | 104 |                  0.31885  |                        -0.0670875 |                          0.092055  |                0.615385 |                  449.519 |
| observe_later_like   | policy_keep_0ms_anchor                  |  27 |                  0.792468 |                         0         |                          0.288258  |                0        |                    0     |
| observe_later_like   | policy_wait_to_latest_anchor            |  27 |                  0.50421  |                        -0.288258  |                          0         |                0.888889 |                 1000     |
| observe_later_like   | oracle_best_anchor_upper_bound          |  27 |                  0.415276 |                        -0.377192  |                         -0.0889338 |                1        |                  761.111 |
| observe_later_like   | selector_vehicle_hgb                    |  27 |                  0.667038 |                        -0.12543   |                          0.162828  |                0.481481 |                  314.815 |
| observe_later_like   | selector_bio260_hgb                     |  27 |                  0.665175 |                        -0.127293  |                          0.160965  |                0.481481 |                  303.704 |
| observe_later_like   | selector_vehicle_bio260_hgb             |  27 |                  0.651361 |                        -0.141107  |                          0.147151  |                0.62963  |                  327.778 |
| observe_later_like   | selector_vehicle_bio260_badweighted_hgb |  27 |                  0.659494 |                        -0.132974  |                          0.155284  |                0.62963  |                  329.63  |
| strong_steer         | policy_keep_0ms_anchor                  |  80 |                  0.590904 |                         0         |                          0.185121  |                0        |                    0     |
| strong_steer         | policy_wait_to_latest_anchor            |  80 |                  0.405783 |                        -0.185121  |                          0         |                0.825    |                 1000     |
| strong_steer         | oracle_best_anchor_upper_bound          |  80 |                  0.32849  |                        -0.262414  |                         -0.0772925 |                0.95     |                  681.25  |
| strong_steer         | selector_vehicle_hgb                    |  80 |                  0.522102 |                        -0.0688018 |                          0.11632   |                0.525    |                  406.25  |
| strong_steer         | selector_bio260_hgb                     |  80 |                  0.518078 |                        -0.0728264 |                          0.112295  |                0.4375   |                  345     |
| strong_steer         | selector_vehicle_bio260_hgb             |  80 |                  0.527446 |                        -0.0634586 |                          0.121663  |                0.55     |                  380     |
| strong_steer         | selector_vehicle_bio260_badweighted_hgb |  80 |                  0.518398 |                        -0.072506  |                          0.112615  |                0.5875   |                  409.375 |
| early_best_after_400 | policy_keep_0ms_anchor                  | 150 |                  0.507357 |                         0         |                          0.218617  |                0        |                    0     |
| early_best_after_400 | policy_wait_to_latest_anchor            | 150 |                  0.28874  |                        -0.218617  |                          0         |                0.866667 |                 1000     |
| early_best_after_400 | oracle_best_anchor_upper_bound          | 150 |                  0.236262 |                        -0.271095  |                         -0.0524782 |                1        |                  846.667 |
| early_best_after_400 | selector_vehicle_hgb                    | 150 |                  0.413384 |                        -0.0939737 |                          0.124644  |                0.653333 |                  432.333 |
| early_best_after_400 | selector_bio260_hgb                     | 150 |                  0.415387 |                        -0.0919701 |                          0.126647  |                0.566667 |                  429.333 |
| early_best_after_400 | selector_vehicle_bio260_hgb             | 150 |                  0.416601 |                        -0.0907562 |                          0.127861  |                0.666667 |                  416     |
| early_best_after_400 | selector_vehicle_bio260_badweighted_hgb | 150 |                  0.414954 |                        -0.0924028 |                          0.126214  |                0.673333 |                  441.667 |

## v258 physio200 参考

| source             | strategy                                |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   selected_delay_ms_mean |
|:-------------------|:----------------------------------------|----:|--------------------------:|----------------------------------:|-------------------------:|
| v258_physio200_ref | policy_keep_0ms_anchor                  |  19 |                  1.19771  |                          0        |                    0     |
| v258_physio200_ref | policy_wait_to_latest_anchor            |  19 |                  0.695048 |                         -0.502658 |                 1000     |
| v258_physio200_ref | oracle_best_anchor_upper_bound          |  19 |                  0.612475 |                         -0.585231 |                  818.421 |
| v258_physio200_ref | selector_vehicle_hgb                    |  19 |                  0.930009 |                         -0.267697 |                  397.368 |
| v258_physio200_ref | selector_vehicle_physio_hgb             |  19 |                  0.934206 |                         -0.2635   |                  413.158 |
| v258_physio200_ref | selector_vehicle_physio_badweighted_hgb |  19 |                  0.959266 |                         -0.238441 |                  347.368 |

## 判读

- bad_top10 / policy_keep_0ms_anchor: tail=1.1977, delta_keep0=+0.0000, delay=0.0ms.
- bad_top10 / selector_vehicle_hgb: tail=0.9425, delta_keep0=-0.2552, delay=384.2ms.
- bad_top10 / selector_bio260_hgb: tail=1.0180, delta_keep0=-0.1797, delay=284.2ms.
- bad_top10 / selector_vehicle_bio260_hgb: tail=0.9765, delta_keep0=-0.2212, delay=313.2ms.
- bad_top10 / selector_vehicle_bio260_badweighted_hgb: tail=0.9837, delta_keep0=-0.2140, delay=281.6ms.
- bad_top10 / policy_wait_to_latest_anchor: tail=0.6950, delta_keep0=-0.5027, delay=1000.0ms.
- bad_top10 / oracle_best_anchor_upper_bound: tail=0.6125, delta_keep0=-0.5852, delay=818.4ms.

- 结论：bad-weighted vehicle+bio260 比 vehicle selector 高 0.0412，说明 bio260 尚不能稳定改善差样本锚点选择。
- 与固定 latest 比：bad-weighted vehicle+bio260 tail=0.9837，latest tail=0.6950；如果仍高于 latest，说明最简单的“多看一点”仍比生理驱动选择更稳。
- 与 0ms 原锚点比：bad-weighted vehicle+bio260 改变量为 -0.2140；这是判断是否弥补锚点前信息不足的核心数字。

## 关键图

- `figures\v261_anchor_selector_test_badtop10.png`