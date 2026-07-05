# v264 online subject-aware physiology calibration

## 本轮边界

- 这不是正式 subject-disjoint 替代结果。
- 本轮允许同一 subject 的更早事件结果作为在线历史反馈，只用于判断生理是否在 subject-aware / online adaptation 设定下更有价值。
- 当前事件本身、当前事件之后的结果、observation_s 之后的生理都不进入输入。

## 特征与在线历史

| model                           |   feature_n |   bio260_feature_n |
|:--------------------------------|------------:|-------------------:|
| global_vehicle_gain             |          35 |                  0 |
| global_vehicle_bio260_sp64_gain |         100 |                 65 |

| split   | subject   |   event_n |   history_ge_min_rate |   physio_knn_used_rate |   bad_top10_n |
|:--------|:----------|----------:|----------------------:|-----------------------:|--------------:|
| test    | cwh       |        46 |              0.934783 |               0.934783 |             1 |
| test    | lx        |        13 |              0.769231 |               0.769231 |             1 |
| test    | rjy       |        82 |              0.963415 |               0.963415 |            12 |
| test    | tyy       |        43 |              0.930233 |               0.930233 |             5 |
| train   | byx       |       102 |              0.970588 |               0.970588 |            15 |
| train   | gf        |        36 |              0.916667 |               0.916667 |             4 |
| train   | hzh       |       118 |              0.974576 |               0.974576 |             7 |
| train   | jy        |        42 |              0.928571 |               0.928571 |             4 |
| train   | xst       |         6 |              0.5      |               0.5      |             0 |
| train   | yyl       |        87 |              0.965517 |               0.965517 |             5 |
| train   | yzy       |        79 |              0.962025 |               0.962025 |             8 |
| train   | zt        |        15 |              0.8      |               0.8      |             0 |
| train   | zx        |       153 |              0.980392 |               0.980392 |            17 |
| train   | zxy       |        36 |              0.916667 |               0.916667 |             8 |
| val     | gzj       |       105 |              0.971429 |               0.971429 |            10 |
| val     | lxy       |        65 |              0.953846 |               0.953846 |             2 |
| val     | txj       |        91 |              0.967033 |               0.967033 |            13 |
| val     | zdq       |        48 |              0.9375   |               0.9375   |             6 |

## Test 关键结果

| event_group        | strategy                         |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   improve_rate_vs_keep0 |   selected_delay_ms_mean |   selected_latest_rate |
|:-------------------|:---------------------------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|------------------------:|-------------------------:|-----------------------:|
| all                | policy_keep_0ms_anchor           | 184 |                  0.475053 |                          0        |                         0.170438   |                0        |                    0     |               0        |
| all                | policy_wait_to_latest_anchor     | 184 |                  0.304615 |                         -0.170438 |                         0          |                0.777174 |                 1000     |               1        |
| all                | oracle_best_anchor_upper_bound   | 184 |                  0.23972  |                         -0.235333 |                        -0.0648958  |                0.951087 |                  711.957 |               0.217391 |
| all                | gate_vehicle_gain_t0             | 184 |                  0.343707 |                         -0.131346 |                         0.0390912  |                0.597826 |                  788.043 |               0.788043 |
| all                | gate_vehicle_bio260_sp64_gain_t0 | 184 |                  0.353017 |                         -0.122036 |                         0.0484016  |                0.597826 |                  782.609 |               0.782609 |
| all                | online_subject_mean_vehicle      | 184 |                  0.307669 |                         -0.167384 |                         0.00305354 |                0.76087  |                  978.261 |               0.978261 |
| all                | online_subject_recent_vehicle    | 184 |                  0.312353 |                         -0.162701 |                         0.00773713 |                0.73913  |                  956.522 |               0.956522 |
| all                | online_physio_knn_vehicle        | 184 |                  0.308603 |                         -0.16645  |                         0.00398774 |                0.755435 |                  972.826 |               0.972826 |
| all                | online_subject_mean_vehicle_bio  | 184 |                  0.306197 |                         -0.168856 |                         0.00158161 |                0.771739 |                  994.565 |               0.994565 |
| all                | online_physio_knn_vehicle_bio    | 184 |                  0.315313 |                         -0.15974  |                         0.0106979  |                0.76087  |                  983.696 |               0.983696 |
| bad_top10          | policy_keep_0ms_anchor           |  19 |                  1.19771  |                          0        |                         0.502658   |                0        |                    0     |               0        |
| bad_top10          | policy_wait_to_latest_anchor     |  19 |                  0.695048 |                         -0.502658 |                         0          |                1        |                 1000     |               1        |
| bad_top10          | oracle_best_anchor_upper_bound   |  19 |                  0.612475 |                         -0.585231 |                        -0.082573   |                1        |                  818.421 |               0.368421 |
| bad_top10          | gate_vehicle_gain_t0             |  19 |                  0.752834 |                         -0.444873 |                         0.0577852  |                0.789474 |                  789.474 |               0.789474 |
| bad_top10          | gate_vehicle_bio260_sp64_gain_t0 |  19 |                  0.874785 |                         -0.322921 |                         0.179737   |                0.736842 |                  736.842 |               0.736842 |
| bad_top10          | online_subject_mean_vehicle      |  19 |                  0.711167 |                         -0.48654  |                         0.0161183  |                0.947368 |                  947.368 |               0.947368 |
| bad_top10          | online_subject_recent_vehicle    |  19 |                  0.711167 |                         -0.48654  |                         0.0161183  |                0.947368 |                  947.368 |               0.947368 |
| bad_top10          | online_physio_knn_vehicle        |  19 |                  0.711167 |                         -0.48654  |                         0.0161183  |                0.947368 |                  947.368 |               0.947368 |
| bad_top10          | online_subject_mean_vehicle_bio  |  19 |                  0.695048 |                         -0.502658 |                         0          |                1        |                 1000     |               1        |
| bad_top10          | online_physio_knn_vehicle_bio    |  19 |                  0.769754 |                         -0.427952 |                         0.0747056  |                0.947368 |                  947.368 |               0.947368 |
| normal             | policy_keep_0ms_anchor           | 104 |                  0.385937 |                          0        |                         0.159142   |                0        |                    0     |               0        |
| normal             | policy_wait_to_latest_anchor     | 104 |                  0.226795 |                         -0.159142 |                         0          |                0.740385 |                 1000     |               1        |
| normal             | oracle_best_anchor_upper_bound   | 104 |                  0.171435 |                         -0.214502 |                        -0.0553599  |                0.951923 |                  735.577 |               0.230769 |
| normal             | gate_vehicle_gain_t0             | 104 |                  0.25531  |                         -0.130627 |                         0.0285157  |                0.615385 |                  846.154 |               0.846154 |
| normal             | gate_vehicle_bio260_sp64_gain_t0 | 104 |                  0.272802 |                         -0.113135 |                         0.0460073  |                0.567308 |                  788.462 |               0.788462 |
| normal             | online_subject_mean_vehicle      | 104 |                  0.226795 |                         -0.159142 |                         0          |                0.740385 |                 1000     |               1        |
| normal             | online_subject_recent_vehicle    | 104 |                  0.233028 |                         -0.152909 |                         0.00623354 |                0.721154 |                  980.769 |               0.980769 |
| normal             | online_physio_knn_vehicle        | 104 |                  0.226795 |                         -0.159142 |                         0          |                0.740385 |                 1000     |               1        |
| normal             | online_subject_mean_vehicle_bio  | 104 |                  0.226795 |                         -0.159142 |                         0          |                0.740385 |                 1000     |               1        |
| normal             | online_physio_knn_vehicle_bio    | 104 |                  0.242924 |                         -0.143014 |                         0.0161289  |                0.721154 |                  980.769 |               0.980769 |
| observe_later_like | policy_keep_0ms_anchor           |  27 |                  0.792468 |                          0        |                         0.288258   |                0        |                    0     |               0        |
| observe_later_like | policy_wait_to_latest_anchor     |  27 |                  0.50421  |                         -0.288258 |                         0          |                0.888889 |                 1000     |               1        |
| observe_later_like | oracle_best_anchor_upper_bound   |  27 |                  0.415276 |                         -0.377192 |                        -0.0889338  |                1        |                  761.111 |               0.296296 |
| observe_later_like | gate_vehicle_gain_t0             |  27 |                  0.569472 |                         -0.222996 |                         0.0652618  |                0.666667 |                  777.778 |               0.777778 |
| observe_later_like | gate_vehicle_bio260_sp64_gain_t0 |  27 |                  0.622804 |                         -0.169664 |                         0.118594   |                0.666667 |                  777.778 |               0.777778 |
| observe_later_like | online_subject_mean_vehicle      |  27 |                  0.515553 |                         -0.276916 |                         0.0113425  |                0.851852 |                  962.963 |               0.962963 |
| observe_later_like | online_subject_recent_vehicle    |  27 |                  0.515553 |                         -0.276916 |                         0.0113425  |                0.851852 |                  962.963 |               0.962963 |
| observe_later_like | online_physio_knn_vehicle        |  27 |                  0.515553 |                         -0.276916 |                         0.0113425  |                0.851852 |                  962.963 |               0.962963 |
| observe_later_like | online_subject_mean_vehicle_bio  |  27 |                  0.50421  |                         -0.288258 |                         0          |                0.888889 |                 1000     |               1        |
| observe_later_like | online_physio_knn_vehicle_bio    |  27 |                  0.556781 |                         -0.235687 |                         0.0525706  |                0.851852 |                  962.963 |               0.962963 |
| strong_steer       | policy_keep_0ms_anchor           |  80 |                  0.590904 |                          0        |                         0.185121   |                0        |                    0     |               0        |
| strong_steer       | policy_wait_to_latest_anchor     |  80 |                  0.405783 |                         -0.185121 |                         0          |                0.825    |                 1000     |               1        |
| strong_steer       | oracle_best_anchor_upper_bound   |  80 |                  0.32849  |                         -0.262414 |                        -0.0772925  |                0.95     |                  681.25  |               0.2      |
| strong_steer       | gate_vehicle_gain_t0             |  80 |                  0.458622 |                         -0.132282 |                         0.0528393  |                0.575    |                  712.5   |               0.7125   |
| strong_steer       | gate_vehicle_bio260_sp64_gain_t0 |  80 |                  0.457297 |                         -0.133607 |                         0.0515142  |                0.6375   |                  775     |               0.775    |
| strong_steer       | online_subject_mean_vehicle      |  80 |                  0.412806 |                         -0.178098 |                         0.00702315 |                0.7875   |                  950     |               0.95     |
| strong_steer       | online_subject_recent_vehicle    |  80 |                  0.415474 |                         -0.17543  |                         0.00969179 |                0.7625   |                  925     |               0.925    |
| strong_steer       | online_physio_knn_vehicle        |  80 |                  0.414954 |                         -0.17595  |                         0.0091718  |                0.775    |                  937.5   |               0.9375   |
| strong_steer       | online_subject_mean_vehicle_bio  |  80 |                  0.40942  |                         -0.181484 |                         0.0036377  |                0.8125   |                  987.5   |               0.9875   |
| strong_steer       | online_physio_knn_vehicle_bio    |  80 |                  0.40942  |                         -0.181484 |                         0.0036377  |                0.8125   |                  987.5   |               0.9875   |

## 判读

- bad_top10 / policy_keep_0ms_anchor: tail=1.1977, latest_rate=0.000.
- bad_top10 / policy_wait_to_latest_anchor: tail=0.6950, latest_rate=1.000.
- bad_top10 / gate_vehicle_gain_t0: tail=0.7528, latest_rate=0.789.
- bad_top10 / gate_vehicle_bio260_sp64_gain_t0: tail=0.8748, latest_rate=0.737.
- bad_top10 / online_subject_mean_vehicle: tail=0.7112, latest_rate=0.947.
- bad_top10 / online_subject_recent_vehicle: tail=0.7112, latest_rate=0.947.
- bad_top10 / online_physio_knn_vehicle: tail=0.7112, latest_rate=0.947.
- bad_top10 / online_subject_mean_vehicle_bio: tail=0.6950, latest_rate=1.000.
- bad_top10 / online_physio_knn_vehicle_bio: tail=0.7698, latest_rate=0.947.
- bad_top10 / oracle_best_anchor_upper_bound: tail=0.6125, latest_rate=0.368.

- 最佳 online 策略 `online_subject_mean_vehicle_bio` 相对 global vehicle gate 改变量为 -0.0578。
- physiology KNN online 相对纯 subject mean online 改变量为 +0.0586；这是判断生理额外价值的核心数。
- 如果 online subject calibration 有效但 physiology KNN 无额外收益，说明需要的是同驾驶员反馈，而不是当前生理特征本身。

## 关键图

- `figures\v264_online_subject_physio_badtop10.png`