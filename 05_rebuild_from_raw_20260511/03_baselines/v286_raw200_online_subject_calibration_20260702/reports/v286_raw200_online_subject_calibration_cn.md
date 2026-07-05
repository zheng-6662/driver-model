# v286 raw-200Hz online subject-aware calibration

## 本轮边界

- 这不是 subject-disjoint 正式结果，而是 subject-aware / online adaptation 边界实验。
- global gate 只用 train split 训练；val/test 校准只用同 split 同 subject 的更早事件。
- 生理表示来自 v285 raw 200Hz shape-state train-only feature set。

## 特征与在线历史

| model                      |   feature_n |   raw285_feature_n |
|:---------------------------|------------:|-------------------:|
| global_vehicle_gain        |          35 |                  0 |
| global_vehicle_raw285_gain |         165 |                130 |
| online_raw285_knn          |         130 |                130 |

| split   | subject   |   event_n |   history_ge_min_rate |   raw285_knn_used_rate |   bad_top10_n |
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

| event_group        | strategy                           |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   improve_rate_vs_keep0 |   selected_delay_ms_mean |   selected_latest_rate |
|:-------------------|:-----------------------------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|------------------------:|-------------------------:|-----------------------:|
| all                | policy_keep_0ms_anchor             | 184 |                  0.475053 |                          0        |                         0.170438   |                0        |                    0     |               0        |
| all                | policy_wait_to_latest_anchor       | 184 |                  0.304615 |                         -0.170438 |                         0          |                0.777174 |                 1000     |               1        |
| all                | oracle_best_anchor_upper_bound     | 184 |                  0.23972  |                         -0.235333 |                        -0.0648958  |                0.951087 |                  711.957 |               0.217391 |
| all                | gate_vehicle_gain_t0               | 184 |                  0.343707 |                         -0.131346 |                         0.0390912  |                0.597826 |                  788.043 |               0.788043 |
| all                | gate_vehicle_raw285_gain_t0        | 184 |                  0.324307 |                         -0.150746 |                         0.0196912  |                0.690217 |                  880.435 |               0.880435 |
| all                | online_subject_mean_vehicle        | 184 |                  0.307669 |                         -0.167384 |                         0.00305354 |                0.76087  |                  978.261 |               0.978261 |
| all                | online_subject_recent_vehicle      | 184 |                  0.312353 |                         -0.162701 |                         0.00773713 |                0.73913  |                  956.522 |               0.956522 |
| all                | online_raw285_knn_vehicle          | 184 |                  0.314774 |                         -0.160279 |                         0.0101587  |                0.728261 |                  945.652 |               0.945652 |
| all                | online_subject_mean_vehicle_raw285 | 184 |                  0.306013 |                         -0.16904  |                         0.00139795 |                0.766304 |                  983.696 |               0.983696 |
| all                | online_raw285_knn_vehicle_raw285   | 184 |                  0.30987  |                         -0.165183 |                         0.00525476 |                0.755435 |                  978.261 |               0.978261 |
| bad_top10          | policy_keep_0ms_anchor             |  19 |                  1.19771  |                          0        |                         0.502658   |                0        |                    0     |               0        |
| bad_top10          | policy_wait_to_latest_anchor       |  19 |                  0.695048 |                         -0.502658 |                         0          |                1        |                 1000     |               1        |
| bad_top10          | oracle_best_anchor_upper_bound     |  19 |                  0.612475 |                         -0.585231 |                        -0.082573   |                1        |                  818.421 |               0.368421 |
| bad_top10          | gate_vehicle_gain_t0               |  19 |                  0.752834 |                         -0.444873 |                         0.0577852  |                0.789474 |                  789.474 |               0.789474 |
| bad_top10          | gate_vehicle_raw285_gain_t0        |  19 |                  0.801685 |                         -0.396021 |                         0.106636   |                0.736842 |                  736.842 |               0.736842 |
| bad_top10          | online_subject_mean_vehicle        |  19 |                  0.711167 |                         -0.48654  |                         0.0161183  |                0.947368 |                  947.368 |               0.947368 |
| bad_top10          | online_subject_recent_vehicle      |  19 |                  0.711167 |                         -0.48654  |                         0.0161183  |                0.947368 |                  947.368 |               0.947368 |
| bad_top10          | online_raw285_knn_vehicle          |  19 |                  0.735776 |                         -0.461931 |                         0.0407271  |                0.894737 |                  894.737 |               0.894737 |
| bad_top10          | online_subject_mean_vehicle_raw285 |  19 |                  0.695048 |                         -0.502658 |                         0          |                1        |                 1000     |               1        |
| bad_top10          | online_raw285_knn_vehicle_raw285   |  19 |                  0.719657 |                         -0.478049 |                         0.0246088  |                0.947368 |                  947.368 |               0.947368 |
| normal             | policy_keep_0ms_anchor             | 104 |                  0.385937 |                          0        |                         0.159142   |                0        |                    0     |               0        |
| normal             | policy_wait_to_latest_anchor       | 104 |                  0.226795 |                         -0.159142 |                         0          |                0.740385 |                 1000     |               1        |
| normal             | oracle_best_anchor_upper_bound     | 104 |                  0.171435 |                         -0.214502 |                        -0.0553599  |                0.951923 |                  735.577 |               0.230769 |
| normal             | gate_vehicle_gain_t0               | 104 |                  0.25531  |                         -0.130627 |                         0.0285157  |                0.615385 |                  846.154 |               0.846154 |
| normal             | gate_vehicle_raw285_gain_t0        | 104 |                  0.242148 |                         -0.143789 |                         0.0153531  |                0.663462 |                  884.615 |               0.884615 |
| normal             | online_subject_mean_vehicle        | 104 |                  0.226795 |                         -0.159142 |                         0          |                0.740385 |                 1000     |               1        |
| normal             | online_subject_recent_vehicle      | 104 |                  0.233028 |                         -0.152909 |                         0.00623354 |                0.721154 |                  980.769 |               0.980769 |
| normal             | online_raw285_knn_vehicle          | 104 |                  0.23007  |                         -0.155867 |                         0.00327504 |                0.711538 |                  971.154 |               0.971154 |
| normal             | online_subject_mean_vehicle_raw285 | 104 |                  0.229268 |                         -0.156669 |                         0.00247329 |                0.721154 |                  971.154 |               0.971154 |
| normal             | online_raw285_knn_vehicle_raw285   | 104 |                  0.228449 |                         -0.157488 |                         0.00165408 |                0.721154 |                  980.769 |               0.980769 |
| observe_later_like | policy_keep_0ms_anchor             |  27 |                  0.792468 |                          0        |                         0.288258   |                0        |                    0     |               0        |
| observe_later_like | policy_wait_to_latest_anchor       |  27 |                  0.50421  |                         -0.288258 |                         0          |                0.888889 |                 1000     |               1        |
| observe_later_like | oracle_best_anchor_upper_bound     |  27 |                  0.415276 |                         -0.377192 |                        -0.0889338  |                1        |                  761.111 |               0.296296 |
| observe_later_like | gate_vehicle_gain_t0               |  27 |                  0.569472 |                         -0.222996 |                         0.0652618  |                0.666667 |                  777.778 |               0.777778 |
| observe_later_like | gate_vehicle_raw285_gain_t0        |  27 |                  0.553533 |                         -0.238935 |                         0.049323   |                0.814815 |                  925.926 |               0.925926 |
| observe_later_like | online_subject_mean_vehicle        |  27 |                  0.515553 |                         -0.276916 |                         0.0113425  |                0.851852 |                  962.963 |               0.962963 |
| observe_later_like | online_subject_recent_vehicle      |  27 |                  0.515553 |                         -0.276916 |                         0.0113425  |                0.851852 |                  962.963 |               0.962963 |
| observe_later_like | online_raw285_knn_vehicle          |  27 |                  0.53731  |                         -0.255159 |                         0.0330996  |                0.777778 |                  888.889 |               0.888889 |
| observe_later_like | online_subject_mean_vehicle_raw285 |  27 |                  0.50421  |                         -0.288258 |                         0          |                0.888889 |                 1000     |               1        |
| observe_later_like | online_raw285_knn_vehicle_raw285   |  27 |                  0.521527 |                         -0.270941 |                         0.0173173  |                0.851852 |                  962.963 |               0.962963 |
| strong_steer       | policy_keep_0ms_anchor             |  80 |                  0.590904 |                          0        |                         0.185121   |                0        |                    0     |               0        |
| strong_steer       | policy_wait_to_latest_anchor       |  80 |                  0.405783 |                         -0.185121 |                         0          |                0.825    |                 1000     |               1        |
| strong_steer       | oracle_best_anchor_upper_bound     |  80 |                  0.32849  |                         -0.262414 |                        -0.0772925  |                0.95     |                  681.25  |               0.2      |
| strong_steer       | gate_vehicle_gain_t0               |  80 |                  0.458622 |                         -0.132282 |                         0.0528393  |                0.575    |                  712.5   |               0.7125   |
| strong_steer       | gate_vehicle_raw285_gain_t0        |  80 |                  0.431113 |                         -0.159791 |                         0.0253308  |                0.725    |                  875     |               0.875    |
| strong_steer       | online_subject_mean_vehicle        |  80 |                  0.412806 |                         -0.178098 |                         0.00702315 |                0.7875   |                  950     |               0.95     |
| strong_steer       | online_subject_recent_vehicle      |  80 |                  0.415474 |                         -0.17543  |                         0.00969179 |                0.7625   |                  925     |               0.925    |
| strong_steer       | online_raw285_knn_vehicle          |  80 |                  0.42489  |                         -0.166014 |                         0.0191074  |                0.75     |                  912.5   |               0.9125   |
| strong_steer       | online_subject_mean_vehicle_raw285 |  80 |                  0.405783 |                         -0.185121 |                         0          |                0.825    |                 1000     |               1        |
| strong_steer       | online_raw285_knn_vehicle_raw285   |  80 |                  0.415718 |                         -0.175186 |                         0.00993565 |                0.8      |                  975     |               0.975    |

## 判读

- bad_top10 / policy_keep_0ms_anchor: tail=1.1977.
- bad_top10 / policy_wait_to_latest_anchor: tail=0.6950.
- bad_top10 / gate_vehicle_gain_t0: tail=0.7528.
- bad_top10 / gate_vehicle_raw285_gain_t0: tail=0.8017.
- bad_top10 / online_subject_mean_vehicle: tail=0.7112.
- bad_top10 / online_subject_recent_vehicle: tail=0.7112.
- bad_top10 / online_raw285_knn_vehicle: tail=0.7358.
- bad_top10 / online_subject_mean_vehicle_raw285: tail=0.6950.
- bad_top10 / online_raw285_knn_vehicle_raw285: tail=0.7197.
- bad_top10 / oracle_best_anchor_upper_bound: tail=0.6125.
- raw285 KNN online 相对纯 subject mean online 改变量为 +0.0246。
- vehicle+raw285 global 后再 raw285 KNN online，相对纯 subject mean online 改变量为 +0.0085。
- fixed wait-latest bad_top10 为 0.6950，这是当前线上策略必须击败的强基线。
- 若 subject-aware online 仍无法稳定低于 wait-latest，说明当前生理数据即使在个体化边界下也没有形成差样本本质改善。

## 关键图

- `figures\v286_raw285_online_badtop10.png`

## guardrail

```json
{
  "pass": true,
  "zip_testzip": true,
  "task_boundary": "subject_aware_online_adaptation_diagnostic_not_formal_subject_disjoint",
  "global_model_train_only": true,
  "online_history_only_previous_same_split_subject_events": true,
  "min_history_for_calibration": 3,
  "raw285_knn_k": 10,
  "event_n": 1167,
  "vehicle_feature_n": 35,
  "raw285_feature_n": 130,
  "v285_source_guardrail_pass": true
}
```