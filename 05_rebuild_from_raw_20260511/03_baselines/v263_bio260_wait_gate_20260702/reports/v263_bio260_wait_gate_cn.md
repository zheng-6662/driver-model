# v263 bio260 wait gate

## 本轮问题

- v261/v262 的 fine-grid selector 只得到小幅生理增益，可能是任务目标太细。
- v263 把任务简化成 0ms 决策：直接保留原锚点，还是等到 1000ms。
- 如果 vehicle+bio260 wait gate 仍不能明显接近 fixed latest，说明当前生理状态对差样本决策仍不够强。

## 特征与阈值

| model                         |   feature_n |   bio260_feature_n |
|:------------------------------|------------:|-------------------:|
| gate_vehicle_gain             |          35 |                  0 |
| gate_vehicle_bio260_sp64_gain |         100 |                 65 |

| pred_col                      |   threshold |   bad_weight |   val_weighted_tail_rmse |   val_tail_rmse |   val_latest_rate | selected_by_script   |
|:------------------------------|------------:|-------------:|-------------------------:|----------------:|------------------:|:---------------------|
| pred_gain_vehicle             |  -0.06856   |            0 |                 0.477979 |        0.477979 |          0.996764 | True                 |
| pred_gain_vehicle_bio260_sp64 |  -0.0588379 |            0 |                 0.478215 |        0.478215 |          0.996764 | True                 |
| pred_gain_vehicle_bio260_sp64 |  -0.0588379 |            4 |                 0.648485 |        0.478215 |          0.996764 | True                 |

## Test 关键结果

| event_group          | strategy                                      |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   improve_rate_vs_keep0 |   selected_delay_ms_mean |   selected_latest_rate |
|:---------------------|:----------------------------------------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|------------------------:|-------------------------:|-----------------------:|
| all                  | policy_keep_0ms_anchor                        | 184 |                  0.475053 |                          0        |                          0.170438  |                0        |                    0     |               0        |
| all                  | policy_wait_to_latest_anchor                  | 184 |                  0.304615 |                         -0.170438 |                          0         |                0.777174 |                 1000     |               1        |
| all                  | oracle_best_anchor_upper_bound                | 184 |                  0.23972  |                         -0.235333 |                         -0.0648958 |                0.951087 |                  711.957 |               0.217391 |
| all                  | gate_vehicle_gain_t0                          | 184 |                  0.343707 |                         -0.131346 |                          0.0390912 |                0.597826 |                  788.043 |               0.788043 |
| all                  | gate_vehicle_bio260_sp64_gain_t0              | 184 |                  0.353017 |                         -0.122036 |                          0.0484016 |                0.597826 |                  782.609 |               0.782609 |
| all                  | gate_vehicle_bio260_sp64_gain_val_all         | 184 |                  0.304615 |                         -0.170438 |                          0         |                0.777174 |                 1000     |               1        |
| all                  | gate_vehicle_bio260_sp64_gain_val_badweighted | 184 |                  0.304615 |                         -0.170438 |                          0         |                0.777174 |                 1000     |               1        |
| bad_top10            | policy_keep_0ms_anchor                        |  19 |                  1.19771  |                          0        |                          0.502658  |                0        |                    0     |               0        |
| bad_top10            | policy_wait_to_latest_anchor                  |  19 |                  0.695048 |                         -0.502658 |                          0         |                1        |                 1000     |               1        |
| bad_top10            | oracle_best_anchor_upper_bound                |  19 |                  0.612475 |                         -0.585231 |                         -0.082573  |                1        |                  818.421 |               0.368421 |
| bad_top10            | gate_vehicle_gain_t0                          |  19 |                  0.752834 |                         -0.444873 |                          0.0577852 |                0.789474 |                  789.474 |               0.789474 |
| bad_top10            | gate_vehicle_bio260_sp64_gain_t0              |  19 |                  0.874785 |                         -0.322921 |                          0.179737  |                0.736842 |                  736.842 |               0.736842 |
| bad_top10            | gate_vehicle_bio260_sp64_gain_val_all         |  19 |                  0.695048 |                         -0.502658 |                          0         |                1        |                 1000     |               1        |
| bad_top10            | gate_vehicle_bio260_sp64_gain_val_badweighted |  19 |                  0.695048 |                         -0.502658 |                          0         |                1        |                 1000     |               1        |
| normal               | policy_keep_0ms_anchor                        | 104 |                  0.385937 |                          0        |                          0.159142  |                0        |                    0     |               0        |
| normal               | policy_wait_to_latest_anchor                  | 104 |                  0.226795 |                         -0.159142 |                          0         |                0.740385 |                 1000     |               1        |
| normal               | oracle_best_anchor_upper_bound                | 104 |                  0.171435 |                         -0.214502 |                         -0.0553599 |                0.951923 |                  735.577 |               0.230769 |
| normal               | gate_vehicle_gain_t0                          | 104 |                  0.25531  |                         -0.130627 |                          0.0285157 |                0.615385 |                  846.154 |               0.846154 |
| normal               | gate_vehicle_bio260_sp64_gain_t0              | 104 |                  0.272802 |                         -0.113135 |                          0.0460073 |                0.567308 |                  788.462 |               0.788462 |
| normal               | gate_vehicle_bio260_sp64_gain_val_all         | 104 |                  0.226795 |                         -0.159142 |                          0         |                0.740385 |                 1000     |               1        |
| normal               | gate_vehicle_bio260_sp64_gain_val_badweighted | 104 |                  0.226795 |                         -0.159142 |                          0         |                0.740385 |                 1000     |               1        |
| observe_later_like   | policy_keep_0ms_anchor                        |  27 |                  0.792468 |                          0        |                          0.288258  |                0        |                    0     |               0        |
| observe_later_like   | policy_wait_to_latest_anchor                  |  27 |                  0.50421  |                         -0.288258 |                          0         |                0.888889 |                 1000     |               1        |
| observe_later_like   | oracle_best_anchor_upper_bound                |  27 |                  0.415276 |                         -0.377192 |                         -0.0889338 |                1        |                  761.111 |               0.296296 |
| observe_later_like   | gate_vehicle_gain_t0                          |  27 |                  0.569472 |                         -0.222996 |                          0.0652618 |                0.666667 |                  777.778 |               0.777778 |
| observe_later_like   | gate_vehicle_bio260_sp64_gain_t0              |  27 |                  0.622804 |                         -0.169664 |                          0.118594  |                0.666667 |                  777.778 |               0.777778 |
| observe_later_like   | gate_vehicle_bio260_sp64_gain_val_all         |  27 |                  0.50421  |                         -0.288258 |                          0         |                0.888889 |                 1000     |               1        |
| observe_later_like   | gate_vehicle_bio260_sp64_gain_val_badweighted |  27 |                  0.50421  |                         -0.288258 |                          0         |                0.888889 |                 1000     |               1        |
| strong_steer         | policy_keep_0ms_anchor                        |  80 |                  0.590904 |                          0        |                          0.185121  |                0        |                    0     |               0        |
| strong_steer         | policy_wait_to_latest_anchor                  |  80 |                  0.405783 |                         -0.185121 |                          0         |                0.825    |                 1000     |               1        |
| strong_steer         | oracle_best_anchor_upper_bound                |  80 |                  0.32849  |                         -0.262414 |                         -0.0772925 |                0.95     |                  681.25  |               0.2      |
| strong_steer         | gate_vehicle_gain_t0                          |  80 |                  0.458622 |                         -0.132282 |                          0.0528393 |                0.575    |                  712.5   |               0.7125   |
| strong_steer         | gate_vehicle_bio260_sp64_gain_t0              |  80 |                  0.457297 |                         -0.133607 |                          0.0515142 |                0.6375   |                  775     |               0.775    |
| strong_steer         | gate_vehicle_bio260_sp64_gain_val_all         |  80 |                  0.405783 |                         -0.185121 |                          0         |                0.825    |                 1000     |               1        |
| strong_steer         | gate_vehicle_bio260_sp64_gain_val_badweighted |  80 |                  0.405783 |                         -0.185121 |                          0         |                0.825    |                 1000     |               1        |
| early_best_after_400 | policy_keep_0ms_anchor                        | 150 |                  0.507357 |                          0        |                          0.218617  |                0        |                    0     |               0        |
| early_best_after_400 | policy_wait_to_latest_anchor                  | 150 |                  0.28874  |                         -0.218617 |                          0         |                0.866667 |                 1000     |               1        |
| early_best_after_400 | oracle_best_anchor_upper_bound                | 150 |                  0.236262 |                         -0.271095 |                         -0.0524782 |                1        |                  846.667 |               0.266667 |
| early_best_after_400 | gate_vehicle_gain_t0                          | 150 |                  0.337539 |                         -0.169818 |                          0.0487991 |                0.68     |                  793.333 |               0.793333 |
| early_best_after_400 | gate_vehicle_bio260_sp64_gain_t0              | 150 |                  0.349933 |                         -0.157424 |                          0.0611931 |                0.666667 |                  793.333 |               0.793333 |
| early_best_after_400 | gate_vehicle_bio260_sp64_gain_val_all         | 150 |                  0.28874  |                         -0.218617 |                          0         |                0.866667 |                 1000     |               1        |
| early_best_after_400 | gate_vehicle_bio260_sp64_gain_val_badweighted | 150 |                  0.28874  |                         -0.218617 |                          0         |                0.866667 |                 1000     |               1        |

## 判读

- bad_top10 / policy_keep_0ms_anchor: tail=1.1977, latest_rate=0.000.
- bad_top10 / gate_vehicle_gain_t0: tail=0.7528, latest_rate=0.789.
- bad_top10 / gate_vehicle_bio260_sp64_gain_t0: tail=0.8748, latest_rate=0.737.
- bad_top10 / gate_vehicle_bio260_sp64_gain_val_all: tail=0.6950, latest_rate=1.000.
- bad_top10 / gate_vehicle_bio260_sp64_gain_val_badweighted: tail=0.6950, latest_rate=1.000.
- bad_top10 / policy_wait_to_latest_anchor: tail=0.6950, latest_rate=1.000.
- bad_top10 / oracle_best_anchor_upper_bound: tail=0.6125, latest_rate=0.368.
- 结论：bio260 wait gate 比 vehicle gate 高 0.1220。
- fixed latest 是不需要生理判断的强基线，tail=0.6950；任何生理 gate 若接近它才有实际意义。

## 关键图

- `figures\v263_bio260_wait_gate_test_badtop10.png`