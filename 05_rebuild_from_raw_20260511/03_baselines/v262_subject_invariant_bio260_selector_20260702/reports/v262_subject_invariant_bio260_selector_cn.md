# v262 subject-invariant bio260 anchor selector

## 本轮问题

- v261 全量 bio260 selector 在 bad_top10 上弱于 vehicle selector。
- v260 eta2 显示部分生理特征有强 subject / recording 成分，这会干扰 subject-disjoint 泛化。
- v262 检查：剔除高个体差异特征、保留状态变化特征后，生理是否能重新产生锚点选择增益。

## 合并审计

|   candidate_rows |   event_n |   bio260_source_rows |   bio260_source_event_n |   bio260_feature_n |   bio260_merge_ok_rate |   bio260_feature_missing_rate_after_merge |   bio260_uses_post_observation_max |   candidate_delay_min |   candidate_delay_max |
|-----------------:|----------:|---------------------:|------------------------:|-------------------:|-----------------------:|------------------------------------------:|-----------------------------------:|----------------------:|----------------------:|
|            24507 |      1167 |                 7002 |                    1167 |                233 |               0.919452 |                                  0.256043 |                                  0 |                     0 |                  1000 |

## 特征集合

| feature_set         |   feature_n |   eta_known_feature_n |   bad_eta_mean |   future_eta_mean |   subject_recording_eta_max_mean |   state_change_rate | row_type   |   bad_top10_v250_diagnostic |   future_cluster4 |   high_future_abs_q75 |   recording |   subject |   future_eta_max |   subject_recording_eta_max |   invariant_score |   column |   feature |   is_state_change |   in_sp32 |   in_sp64 |   in_state_change |
|:--------------------|------------:|----------------------:|---------------:|------------------:|---------------------------------:|--------------------:|:-----------|----------------------------:|------------------:|----------------------:|------------:|----------:|-----------------:|----------------------------:|------------------:|---------:|----------:|------------------:|----------:|----------:|------------------:|
| bio260_sp32         |          33 |                    32 |    0.000932998 |        0.00320971 |                        0.0723383 |            0.5625   | summary    |                         nan |               nan |                   nan |         nan |       nan |              nan |                         nan |               nan |      nan |       nan |               nan |       nan |       nan |               nan |
| bio260_sp64         |          65 |                    64 |    0.00172684  |        0.00348339 |                        0.138431  |            0.484375 | summary    |                         nan |               nan |                   nan |         nan |       nan |              nan |                         nan |               nan |      nan |       nan |               nan |       nan |       nan |               nan |
| bio260_state_change |          69 |                    68 |    0.000771762 |        0.00222318 |                        0.0816951 |            1        | summary    |                         nan |               nan |                   nan |         nan |       nan |              nan |                         nan |               nan |      nan |       nan |               nan |       nan |       nan |               nan |

| model                                        |   feature_n |   bio260_feature_n |
|:---------------------------------------------|------------:|-------------------:|
| selector_vehicle_hgb                         |          35 |                  0 |
| selector_vehicle_bio260_sp32_hgb             |          68 |                 33 |
| selector_vehicle_bio260_sp64_hgb             |         100 |                 65 |
| selector_vehicle_bio260_sp64_badweighted_hgb |         100 |                 65 |
| selector_vehicle_bio260_state_change_hgb     |         104 |                 69 |

## Test 关键结果

| event_group          | strategy                                     |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   improve_rate_vs_keep0 |   selected_delay_ms_mean |
|:---------------------|:---------------------------------------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|------------------------:|-------------------------:|
| all                  | policy_keep_0ms_anchor                       | 184 |                  0.475053 |                         0         |                          0.170438  |                0        |                    0     |
| all                  | policy_wait_to_latest_anchor                 | 184 |                  0.304615 |                        -0.170438  |                          0         |                0.777174 |                 1000     |
| all                  | oracle_best_anchor_upper_bound               | 184 |                  0.23972  |                        -0.235333  |                         -0.0648958 |                0.951087 |                  711.957 |
| all                  | selector_vehicle_hgb                         | 184 |                  0.403998 |                        -0.071055  |                          0.0993827 |                0.559783 |                  422.826 |
| all                  | selector_vehicle_bio260_sp32_hgb             | 184 |                  0.400987 |                        -0.0740659 |                          0.0963718 |                0.581522 |                  431.25  |
| all                  | selector_vehicle_bio260_sp64_hgb             | 184 |                  0.393478 |                        -0.0815748 |                          0.0888629 |                0.592391 |                  438.315 |
| all                  | selector_vehicle_bio260_sp64_badweighted_hgb | 184 |                  0.410113 |                        -0.0649405 |                          0.105497  |                0.603261 |                  433.696 |
| all                  | selector_vehicle_bio260_state_change_hgb     | 184 |                  0.40098  |                        -0.0740729 |                          0.0963648 |                0.570652 |                  439.674 |
| bad_top10            | policy_keep_0ms_anchor                       |  19 |                  1.19771  |                         0         |                          0.502658  |                0        |                    0     |
| bad_top10            | policy_wait_to_latest_anchor                 |  19 |                  0.695048 |                        -0.502658  |                          0         |                1        |                 1000     |
| bad_top10            | oracle_best_anchor_upper_bound               |  19 |                  0.612475 |                        -0.585231  |                         -0.082573  |                1        |                  818.421 |
| bad_top10            | selector_vehicle_hgb                         |  19 |                  0.941915 |                        -0.255791  |                          0.246867  |                0.684211 |                  413.158 |
| bad_top10            | selector_vehicle_bio260_sp32_hgb             |  19 |                  0.981934 |                        -0.215772  |                          0.286886  |                0.736842 |                  352.632 |
| bad_top10            | selector_vehicle_bio260_sp64_hgb             |  19 |                  0.905944 |                        -0.291762  |                          0.210896  |                0.789474 |                  507.895 |
| bad_top10            | selector_vehicle_bio260_sp64_badweighted_hgb |  19 |                  1.06315  |                        -0.134557  |                          0.368101  |                0.631579 |                  268.421 |
| bad_top10            | selector_vehicle_bio260_state_change_hgb     |  19 |                  0.954688 |                        -0.243018  |                          0.25964   |                0.684211 |                  394.737 |
| normal               | policy_keep_0ms_anchor                       | 104 |                  0.385937 |                         0         |                          0.159142  |                0        |                    0     |
| normal               | policy_wait_to_latest_anchor                 | 104 |                  0.226795 |                        -0.159142  |                          0         |                0.740385 |                 1000     |
| normal               | oracle_best_anchor_upper_bound               | 104 |                  0.171435 |                        -0.214502  |                         -0.0553599 |                0.951923 |                  735.577 |
| normal               | selector_vehicle_hgb                         | 104 |                  0.320284 |                        -0.0656532 |                          0.0934892 |                0.567308 |                  426.923 |
| normal               | selector_vehicle_bio260_sp32_hgb             | 104 |                  0.311265 |                        -0.074672  |                          0.0844704 |                0.586538 |                  459.615 |
| normal               | selector_vehicle_bio260_sp64_hgb             | 104 |                  0.310592 |                        -0.0753449 |                          0.0837975 |                0.586538 |                  453.846 |
| normal               | selector_vehicle_bio260_sp64_badweighted_hgb | 104 |                  0.329499 |                        -0.0564385 |                          0.102704  |                0.615385 |                  453.365 |
| normal               | selector_vehicle_bio260_state_change_hgb     | 104 |                  0.30533  |                        -0.0806074 |                          0.078535  |                0.605769 |                  483.173 |
| observe_later_like   | policy_keep_0ms_anchor                       |  27 |                  0.792468 |                         0         |                          0.288258  |                0        |                    0     |
| observe_later_like   | policy_wait_to_latest_anchor                 |  27 |                  0.50421  |                        -0.288258  |                          0         |                0.888889 |                 1000     |
| observe_later_like   | oracle_best_anchor_upper_bound               |  27 |                  0.415276 |                        -0.377192  |                         -0.0889338 |                1        |                  761.111 |
| observe_later_like   | selector_vehicle_hgb                         |  27 |                  0.658355 |                        -0.134113  |                          0.154145  |                0.555556 |                  312.963 |
| observe_later_like   | selector_vehicle_bio260_sp32_hgb             |  27 |                  0.673151 |                        -0.119317  |                          0.168941  |                0.555556 |                  312.963 |
| observe_later_like   | selector_vehicle_bio260_sp64_hgb             |  27 |                  0.632447 |                        -0.160022  |                          0.128236  |                0.592593 |                  372.222 |
| observe_later_like   | selector_vehicle_bio260_sp64_badweighted_hgb |  27 |                  0.684674 |                        -0.107794  |                          0.180464  |                0.555556 |                  329.63  |
| observe_later_like   | selector_vehicle_bio260_state_change_hgb     |  27 |                  0.673485 |                        -0.118983  |                          0.169275  |                0.481481 |                  309.259 |
| strong_steer         | policy_keep_0ms_anchor                       |  80 |                  0.590904 |                         0         |                          0.185121  |                0        |                    0     |
| strong_steer         | policy_wait_to_latest_anchor                 |  80 |                  0.405783 |                        -0.185121  |                          0         |                0.825    |                 1000     |
| strong_steer         | oracle_best_anchor_upper_bound               |  80 |                  0.32849  |                        -0.262414  |                         -0.0772925 |                0.95     |                  681.25  |
| strong_steer         | selector_vehicle_hgb                         |  80 |                  0.512827 |                        -0.0780772 |                          0.107044  |                0.55     |                  417.5   |
| strong_steer         | selector_vehicle_bio260_sp32_hgb             |  80 |                  0.517626 |                        -0.0732779 |                          0.111844  |                0.575    |                  394.375 |
| strong_steer         | selector_vehicle_bio260_sp64_hgb             |  80 |                  0.50123  |                        -0.0896737 |                          0.0954477 |                0.6      |                  418.125 |
| strong_steer         | selector_vehicle_bio260_sp64_badweighted_hgb |  80 |                  0.514911 |                        -0.0759932 |                          0.109128  |                0.5875   |                  408.125 |
| strong_steer         | selector_vehicle_bio260_state_change_hgb     |  80 |                  0.525326 |                        -0.0655779 |                          0.119543  |                0.525    |                  383.125 |
| early_best_after_400 | policy_keep_0ms_anchor                       | 150 |                  0.507357 |                         0         |                          0.218617  |                0        |                    0     |
| early_best_after_400 | policy_wait_to_latest_anchor                 | 150 |                  0.28874  |                        -0.218617  |                          0         |                0.866667 |                 1000     |
| early_best_after_400 | oracle_best_anchor_upper_bound               | 150 |                  0.236262 |                        -0.271095  |                         -0.0524782 |                1        |                  846.667 |
| early_best_after_400 | selector_vehicle_hgb                         | 150 |                  0.411768 |                        -0.0955893 |                          0.123028  |                0.626667 |                  430.333 |
| early_best_after_400 | selector_vehicle_bio260_sp32_hgb             | 150 |                  0.408051 |                        -0.0993061 |                          0.119311  |                0.653333 |                  444     |
| early_best_after_400 | selector_vehicle_bio260_sp64_hgb             | 150 |                  0.399221 |                        -0.108136  |                          0.110481  |                0.66     |                  448.333 |
| early_best_after_400 | selector_vehicle_bio260_sp64_badweighted_hgb | 150 |                  0.421887 |                        -0.0854703 |                          0.133147  |                0.66     |                  436     |
| early_best_after_400 | selector_vehicle_bio260_state_change_hgb     | 150 |                  0.40993  |                        -0.097427  |                          0.12119   |                0.633333 |                  448.667 |

## v261 全量 bio260 参考

| source               | strategy                                |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   selected_delay_ms_mean |
|:---------------------|:----------------------------------------|----:|--------------------------:|----------------------------------:|-------------------------:|
| v261_full_bio260_ref | policy_keep_0ms_anchor                  |  19 |                  1.19771  |                          0        |                    0     |
| v261_full_bio260_ref | policy_wait_to_latest_anchor            |  19 |                  0.695048 |                         -0.502658 |                 1000     |
| v261_full_bio260_ref | oracle_best_anchor_upper_bound          |  19 |                  0.612475 |                         -0.585231 |                  818.421 |
| v261_full_bio260_ref | selector_vehicle_hgb                    |  19 |                  0.942475 |                         -0.255231 |                  384.211 |
| v261_full_bio260_ref | selector_bio260_hgb                     |  19 |                  1.01797  |                         -0.179732 |                  284.211 |
| v261_full_bio260_ref | selector_vehicle_bio260_hgb             |  19 |                  0.976487 |                         -0.221219 |                  313.158 |
| v261_full_bio260_ref | selector_vehicle_bio260_badweighted_hgb |  19 |                  0.98367  |                         -0.214036 |                  281.579 |

## 判读

- bad_top10 / policy_keep_0ms_anchor: tail=1.1977, delta_keep0=+0.0000, delay=0.0ms.
- bad_top10 / selector_vehicle_hgb: tail=0.9419, delta_keep0=-0.2558, delay=413.2ms.
- bad_top10 / selector_vehicle_bio260_sp32_hgb: tail=0.9819, delta_keep0=-0.2158, delay=352.6ms.
- bad_top10 / selector_vehicle_bio260_sp64_hgb: tail=0.9059, delta_keep0=-0.2918, delay=507.9ms.
- bad_top10 / selector_vehicle_bio260_sp64_badweighted_hgb: tail=1.0631, delta_keep0=-0.1346, delay=268.4ms.
- bad_top10 / selector_vehicle_bio260_state_change_hgb: tail=0.9547, delta_keep0=-0.2430, delay=394.7ms.
- bad_top10 / policy_wait_to_latest_anchor: tail=0.6950, delta_keep0=-0.5027, delay=1000.0ms.
- bad_top10 / oracle_best_anchor_upper_bound: tail=0.6125, delta_keep0=-0.5852, delay=818.4ms.

- 结论：最佳 subject-invariant bio260 策略 `selector_vehicle_bio260_sp64_hgb` 比 vehicle selector 低 0.0360，说明去个体差异后生理存在可用增益。
- 与固定 latest 比：最佳 bio260 tail=0.9059，latest tail=0.6950；若仍高很多，则当前生理还不能替代简单多观察。

## 关键图

- `figures\v262_subject_invariant_bio260_test_badtop10.png`