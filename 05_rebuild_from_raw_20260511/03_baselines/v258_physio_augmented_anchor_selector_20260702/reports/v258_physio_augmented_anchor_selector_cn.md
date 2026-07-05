# v258 生理增强 anchor selector

## 本轮问题

- v245-v247 证明等待/重锚定对差样本有明确上限。
- v254b-v257 证明生理直接预测轨迹、候选未来或个体化记忆都没有本质改善。
- v258 检查最后一个较合理的生理用途：让生理参与判断什么时候等待/重锚定。

## 方法

- 复用 v247 的 50ms fine-grid 候选锚点和 v241 replay 误差。
- 候选特征 = v247 车辆/road/phase 特征 + v254b 生理特征。
- 生理特征用 floor coarse delay 合并，确保不使用候选锚点之后的生理。
- 训练目标是候选 `target_score_primary`，训练 split 训练，val/test 只报告。

## 合并与特征

|   candidate_rows |   event_n |   physio_feature_n |   physio_merge_ok_rate |   candidate_delay_min |   candidate_delay_max |
|-----------------:|----------:|-------------------:|-----------------------:|----------------------:|----------------------:|
|            24507 |      1167 |                196 |               0.919452 |                     0 |                  1000 |

| model                                   |   feature_n |   physio_feature_n |
|:----------------------------------------|------------:|-------------------:|
| selector_vehicle_hgb                    |          35 |                  0 |
| selector_vehicle_physio_hgb             |         232 |                197 |
| selector_vehicle_physio_badweighted_hgb |         232 |                197 |

## Test 关键结果

| event_group          | strategy                                |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   improve_rate_vs_keep0 |   selected_delay_ms_mean |
|:---------------------|:----------------------------------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|------------------------:|-------------------------:|
| all                  | policy_keep_0ms_anchor                  | 184 |                  0.475053 |                         0         |                          0.170438  |                0        |                    0     |
| all                  | policy_wait_to_latest_anchor            | 184 |                  0.304615 |                        -0.170438  |                          0         |                0.777174 |                 1000     |
| all                  | oracle_best_anchor_upper_bound          | 184 |                  0.23972  |                        -0.235333  |                         -0.0648958 |                0.951087 |                  711.957 |
| all                  | selector_vehicle_hgb                    | 184 |                  0.401096 |                        -0.0739569 |                          0.0964808 |                0.554348 |                  422.554 |
| all                  | selector_vehicle_physio_hgb             | 184 |                  0.412364 |                        -0.0626897 |                          0.107748  |                0.576087 |                  421.467 |
| all                  | selector_vehicle_physio_badweighted_hgb | 184 |                  0.403531 |                        -0.0715226 |                          0.0989151 |                0.570652 |                  414.946 |
| bad_top10            | policy_keep_0ms_anchor                  |  19 |                  1.19771  |                         0         |                          0.502658  |                0        |                    0     |
| bad_top10            | policy_wait_to_latest_anchor            |  19 |                  0.695048 |                        -0.502658  |                          0         |                1        |                 1000     |
| bad_top10            | oracle_best_anchor_upper_bound          |  19 |                  0.612475 |                        -0.585231  |                         -0.082573  |                1        |                  818.421 |
| bad_top10            | selector_vehicle_hgb                    |  19 |                  0.930009 |                        -0.267697  |                          0.23496   |                0.684211 |                  397.368 |
| bad_top10            | selector_vehicle_physio_hgb             |  19 |                  0.934206 |                        -0.2635    |                          0.239157  |                0.789474 |                  413.158 |
| bad_top10            | selector_vehicle_physio_badweighted_hgb |  19 |                  0.959266 |                        -0.238441  |                          0.264217  |                0.684211 |                  347.368 |
| normal               | policy_keep_0ms_anchor                  | 104 |                  0.385937 |                         0         |                          0.159142  |                0        |                    0     |
| normal               | policy_wait_to_latest_anchor            | 104 |                  0.226795 |                        -0.159142  |                          0         |                0.740385 |                 1000     |
| normal               | oracle_best_anchor_upper_bound          | 104 |                  0.171435 |                        -0.214502  |                         -0.0553599 |                0.951923 |                  735.577 |
| normal               | selector_vehicle_hgb                    | 104 |                  0.314256 |                        -0.0716813 |                          0.0874611 |                0.576923 |                  439.423 |
| normal               | selector_vehicle_physio_hgb             | 104 |                  0.31907  |                        -0.066867  |                          0.0922754 |                0.596154 |                  452.885 |
| normal               | selector_vehicle_physio_badweighted_hgb | 104 |                  0.318028 |                        -0.067909  |                          0.0912334 |                0.567308 |                  443.269 |
| observe_later_like   | policy_keep_0ms_anchor                  |  27 |                  0.792468 |                         0         |                          0.288258  |                0        |                    0     |
| observe_later_like   | policy_wait_to_latest_anchor            |  27 |                  0.50421  |                        -0.288258  |                          0         |                0.888889 |                 1000     |
| observe_later_like   | oracle_best_anchor_upper_bound          |  27 |                  0.415276 |                        -0.377192  |                         -0.0889338 |                1        |                  761.111 |
| observe_later_like   | selector_vehicle_hgb                    |  27 |                  0.647971 |                        -0.144497  |                          0.143761  |                0.555556 |                  314.815 |
| observe_later_like   | selector_vehicle_physio_hgb             |  27 |                  0.665929 |                        -0.126539  |                          0.161719  |                0.703704 |                  362.963 |
| observe_later_like   | selector_vehicle_physio_badweighted_hgb |  27 |                  0.631895 |                        -0.160573  |                          0.127685  |                0.555556 |                  333.333 |
| strong_steer         | policy_keep_0ms_anchor                  |  80 |                  0.590904 |                         0         |                          0.185121  |                0        |                    0     |
| strong_steer         | policy_wait_to_latest_anchor            |  80 |                  0.405783 |                        -0.185121  |                          0         |                0.825    |                 1000     |
| strong_steer         | oracle_best_anchor_upper_bound          |  80 |                  0.32849  |                        -0.262414  |                         -0.0772925 |                0.95     |                  681.25  |
| strong_steer         | selector_vehicle_hgb                    |  80 |                  0.513989 |                        -0.0769152 |                          0.108206  |                0.525    |                  400.625 |
| strong_steer         | selector_vehicle_physio_hgb             |  80 |                  0.533645 |                        -0.0572591 |                          0.127862  |                0.55     |                  380.625 |
| strong_steer         | selector_vehicle_physio_badweighted_hgb |  80 |                  0.514684 |                        -0.0762202 |                          0.108901  |                0.575    |                  378.125 |
| early_best_after_400 | policy_keep_0ms_anchor                  | 150 |                  0.507357 |                         0         |                          0.218617  |                0        |                    0     |
| early_best_after_400 | policy_wait_to_latest_anchor            | 150 |                  0.28874  |                        -0.218617  |                          0         |                0.866667 |                 1000     |
| early_best_after_400 | oracle_best_anchor_upper_bound          | 150 |                  0.236262 |                        -0.271095  |                         -0.0524782 |                1        |                  846.667 |
| early_best_after_400 | selector_vehicle_hgb                    | 150 |                  0.407034 |                        -0.100323  |                          0.118294  |                0.62     |                  436     |
| early_best_after_400 | selector_vehicle_physio_hgb             | 150 |                  0.423656 |                        -0.0837016 |                          0.134916  |                0.64     |                  428     |
| early_best_after_400 | selector_vehicle_physio_badweighted_hgb | 150 |                  0.413581 |                        -0.0937761 |                          0.124841  |                0.64     |                  422.333 |

## 判读

- bad_top10 / policy_keep_0ms_anchor: tail=1.1977, delta_keep0=+0.0000.
- bad_top10 / selector_vehicle_hgb: tail=0.9300, delta_keep0=-0.2677.
- bad_top10 / selector_vehicle_physio_hgb: tail=0.9342, delta_keep0=-0.2635.
- bad_top10 / selector_vehicle_physio_badweighted_hgb: tail=0.9593, delta_keep0=-0.2384.
- bad_top10 / policy_wait_to_latest_anchor: tail=0.6950, delta_keep0=-0.5027.
- bad_top10 / oracle_best_anchor_upper_bound: tail=0.6125, delta_keep0=-0.5852.
- 若 vehicle+physio selector 不明显优于 vehicle selector 和固定 wait-latest，则当前生理不能承担等待决策。
- 若 selector 仍弱于 wait-latest，说明等待上限主要来自多观察车辆状态，而不是生理状态判断。

## 关键图

- `figures\v258_anchor_selector_test_badtop10.png`