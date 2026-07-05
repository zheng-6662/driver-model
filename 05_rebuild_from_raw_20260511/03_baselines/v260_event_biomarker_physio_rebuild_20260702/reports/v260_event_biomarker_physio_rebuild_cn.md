# v260 事件型生理 biomarker 重构与诊断

## 本轮问题

- v254b-v259 显示：现有生理统计/序列直接拼接或 attention 融合没有形成稳定预测增量。
- v260 因此改查数据层：从 ECG/EDA/RESP/EMG 连续波形重新派生事件型 biomarker，再评估其可辨识性。
- 本轮仍不删除样本、不使用 observation_s 之后数据、不使用 test 后验误差做部署输入。

## 生理特征重构

- ECG：按 baseline robust z 后检测局部峰，计算 peak rate、IBI、SDNN、RMSSD、derived BPM。
- EDA/SCR：使用 EDA_Phasic，若缺失则回退 EDA_filt200，计算正向面积、峰、burst episode。
- RESP：从 RESP_filt200 重新计算零交叉、周期、BPM 和窗口末端相位。
- EMG：从 EMG_RMS 计算 burst rate、burst episode、绝对/正向面积和近期变化。

## 对齐覆盖

| split   |    n |   ok_rate |   uses_post_observation_rate |   baseline_rows_mean |   baseline_duration_s_mean |
|:--------|-----:|----------:|-----------------------------:|---------------------:|---------------------------:|
| test    | 1104 |  0.896739 |                            0 |              7971.56 |                    39.8528 |
| train   | 4044 |  0.888724 |                            0 |              7959.22 |                    39.7911 |
| val     | 1854 |  1        |                            0 |              7966.88 |                    39.8294 |

## 特征块

| feature_block                      |   raw_dim |   kept_dim |   kept_bio260_columns |   kept_physio200_columns |
|:-----------------------------------|----------:|-----------:|----------------------:|-------------------------:|
| vehicle_only                       |       268 |        268 |                     0 |                        0 |
| bio260_curated                     |       234 |        203 |                   203 |                        0 |
| vehicle_plus_bio260_curated        |       502 |        471 |                   203 |                        0 |
| physio200_curated_ref              |       168 |        168 |                     0 |                      168 |
| vehicle_plus_physio200_curated_ref |       436 |        436 |                     0 |                      168 |

## Subject-disjoint test 分类结果

| target                    | feature_block                      |   n_eval |   accuracy |   macro_f1 |        auc |   vehicle_metric |   delta_macro_f1_minus_vehicle |
|:--------------------------|:-----------------------------------|---------:|-----------:|-----------:|-----------:|-----------------:|-------------------------------:|
| future_cluster4           | vehicle_only                       |     1104 |   0.754529 |   0.739618 | nan        |         0.739618 |                     0          |
| future_cluster4           | bio260_curated                     |     1104 |   0.276268 |   0.261018 | nan        |         0.739618 |                    -0.4786     |
| future_cluster4           | vehicle_plus_bio260_curated        |     1104 |   0.628623 |   0.624737 | nan        |         0.739618 |                    -0.114881   |
| future_cluster4           | physio200_curated_ref              |     1104 |   0.269022 |   0.255883 | nan        |         0.739618 |                    -0.483736   |
| future_cluster4           | vehicle_plus_physio200_curated_ref |     1104 |   0.682065 |   0.670094 | nan        |         0.739618 |                    -0.0695238  |
| high_future_abs_q75       | vehicle_only                       |     1104 |   0.770833 |   0.686338 |   0.756538 |         0.686338 |                     0          |
| high_future_abs_q75       | bio260_curated                     |     1104 |   0.588768 |   0.522977 |   0.568938 |         0.686338 |                    -0.163361   |
| high_future_abs_q75       | vehicle_plus_bio260_curated        |     1104 |   0.726449 |   0.647684 |   0.726739 |         0.686338 |                    -0.038654   |
| high_future_abs_q75       | physio200_curated_ref              |     1104 |   0.463768 |   0.437169 |   0.484998 |         0.686338 |                    -0.249169   |
| high_future_abs_q75       | vehicle_plus_physio200_curated_ref |     1104 |   0.669384 |   0.598081 |   0.67531  |         0.686338 |                    -0.0882566  |
| high_future_range_q75     | vehicle_only                       |     1104 |   0.742754 |   0.660264 |   0.722726 |         0.660264 |                     0          |
| high_future_range_q75     | bio260_curated                     |     1104 |   0.658514 |   0.558724 |   0.597416 |         0.660264 |                    -0.10154    |
| high_future_range_q75     | vehicle_plus_bio260_curated        |     1104 |   0.720109 |   0.604437 |   0.666649 |         0.660264 |                    -0.0558262  |
| high_future_range_q75     | physio200_curated_ref              |     1104 |   0.441123 |   0.428035 |   0.460083 |         0.660264 |                    -0.232228   |
| high_future_range_q75     | vehicle_plus_physio200_curated_ref |     1104 |   0.651268 |   0.567869 |   0.619817 |         0.660264 |                    -0.0923944  |
| strong_steer_existing     | vehicle_only                       |     1104 |   0.647645 |   0.639394 |   0.687583 |         0.639394 |                     0          |
| strong_steer_existing     | bio260_curated                     |     1104 |   0.5      |   0.494986 |   0.522272 |         0.639394 |                    -0.144407   |
| strong_steer_existing     | vehicle_plus_bio260_curated        |     1104 |   0.615036 |   0.612631 |   0.645663 |         0.639394 |                    -0.0267629  |
| strong_steer_existing     | physio200_curated_ref              |     1104 |   0.499094 |   0.497458 |   0.514059 |         0.639394 |                    -0.141936   |
| strong_steer_existing     | vehicle_plus_physio200_curated_ref |     1104 |   0.618659 |   0.613545 |   0.680582 |         0.639394 |                    -0.0258482  |
| bad_top10_v250_diagnostic | vehicle_only                       |     1104 |   0.761775 |   0.502487 |   0.542269 |         0.502487 |                     0          |
| bad_top10_v250_diagnostic | bio260_curated                     |     1104 |   0.834239 |   0.494656 |   0.543802 |         0.502487 |                    -0.00783116 |
| bad_top10_v250_diagnostic | vehicle_plus_bio260_curated        |     1104 |   0.865036 |   0.512019 |   0.593969 |         0.502487 |                     0.00953175 |
| bad_top10_v250_diagnostic | physio200_curated_ref              |     1104 |   0.639493 |   0.448219 |   0.51619  |         0.502487 |                    -0.0542681  |
| bad_top10_v250_diagnostic | vehicle_plus_physio200_curated_ref |     1104 |   0.871377 |   0.515999 |   0.576581 |         0.502487 |                     0.0135116  |

## Subject-disjoint test 回归结果

| target          | feature_block                      |   n_eval |         r2 |      mae |   vehicle_metric |   delta_r2_minus_vehicle |
|:----------------|:-----------------------------------|---------:|-----------:|---------:|-----------------:|-------------------------:|
| future_peak_abs | vehicle_only                       |     1104 |  0.15494   | 0.546967 |        0.15494   |                0         |
| future_peak_abs | bio260_curated                     |     1104 | -0.256635  | 0.738749 |        0.15494   |               -0.411576  |
| future_peak_abs | vehicle_plus_bio260_curated        |     1104 |  0.0384408 | 0.591637 |        0.15494   |               -0.1165    |
| future_peak_abs | physio200_curated_ref              |     1104 | -0.426795  | 0.800954 |        0.15494   |               -0.581735  |
| future_peak_abs | vehicle_plus_physio200_curated_ref |     1104 |  0.0051643 | 0.610826 |        0.15494   |               -0.149776  |
| future_range    | vehicle_only                       |     1104 |  0.0288895 | 0.640475 |        0.0288895 |                0         |
| future_range    | bio260_curated                     |     1104 | -0.179592  | 0.796632 |        0.0288895 |               -0.208482  |
| future_range    | vehicle_plus_bio260_curated        |     1104 | -0.0469712 | 0.672378 |        0.0288895 |               -0.0758607 |
| future_range    | physio200_curated_ref              |     1104 | -0.393545  | 0.890391 |        0.0288895 |               -0.422435  |
| future_range    | vehicle_plus_physio200_curated_ref |     1104 | -0.12248   | 0.711887 |        0.0288895 |               -0.15137   |
| future_mean_abs | vehicle_only                       |     1104 | -0.0492649 | 0.326029 |       -0.0492649 |                0         |
| future_mean_abs | bio260_curated                     |     1104 | -0.217442  | 0.438962 |       -0.0492649 |               -0.168177  |
| future_mean_abs | vehicle_plus_bio260_curated        |     1104 | -0.2188    | 0.356081 |       -0.0492649 |               -0.169535  |
| future_mean_abs | physio200_curated_ref              |     1104 | -0.3253    | 0.457418 |       -0.0492649 |               -0.276035  |
| future_mean_abs | vehicle_plus_physio200_curated_ref |     1104 | -0.15771   | 0.353892 |       -0.0492649 |               -0.108445  |
| future_final    | vehicle_only                       |     1104 | -1.18991   | 0.664334 |       -1.18991   |                0         |
| future_final    | bio260_curated                     |     1104 | -0.279336  | 1.11253  |       -1.18991   |                0.910575  |
| future_final    | vehicle_plus_bio260_curated        |     1104 | -1.34977   | 0.756379 |       -1.18991   |               -0.159864  |
| future_final    | physio200_curated_ref              |     1104 | -0.133499  | 0.994585 |       -1.18991   |                1.05641   |
| future_final    | vehicle_plus_physio200_curated_ref |     1104 | -1.11156   | 0.713414 |       -1.18991   |                0.0783538 |
| future_slope    | vehicle_only                       |     1104 | -1.18991   | 0.332167 |       -1.18991   |                0         |
| future_slope    | bio260_curated                     |     1104 | -0.279336  | 0.556263 |       -1.18991   |                0.910575  |
| future_slope    | vehicle_plus_bio260_curated        |     1104 | -1.34977   | 0.37819  |       -1.18991   |               -0.159864  |
| future_slope    | physio200_curated_ref              |     1104 | -0.133499  | 0.497292 |       -1.18991   |                1.05641   |
| future_slope    | vehicle_plus_physio200_curated_ref |     1104 | -1.11156   | 0.356707 |       -1.18991   |                0.0783538 |

## eta² top

| target                    | feature                                                  | signal   |       eta2 |
|:--------------------------|:---------------------------------------------------------|:---------|-----------:|
| bad_top10_v250_diagnostic | bio260_pre5_0_ecg_peak_count                             | ecg      | 0.0537405  |
| bad_top10_v250_diagnostic | bio260_pre5_0_ecg_peak_rate_per_min                      | ecg      | 0.0537175  |
| bad_top10_v250_diagnostic | bio260_pre5_0_ecg_ibi_mean_s                             | ecg      | 0.0520667  |
| bad_top10_v250_diagnostic | bio260_pre5_0_ecg_bpm_from_peaks                         | ecg      | 0.0518643  |
| bad_top10_v250_diagnostic | bio260_pre10_pre5_ecg_ibi_mean_s                         | ecg      | 0.0456986  |
| bad_top10_v250_diagnostic | bio260_pre5_pre2_ecg_ibi_mean_s                          | ecg      | 0.0453141  |
| bad_top10_v250_diagnostic | bio260_pre5_pre2_ecg_bpm_from_peaks                      | ecg      | 0.0447631  |
| bad_top10_v250_diagnostic | bio260_pre5_pre2_ecg_peak_count                          | ecg      | 0.0418102  |
| bad_top10_v250_diagnostic | bio260_pre5_pre2_ecg_peak_rate_per_min                   | ecg      | 0.0417802  |
| bad_top10_v250_diagnostic | bio260_pre10_pre5_ecg_bpm_from_peaks                     | ecg      | 0.041246   |
| bad_top10_v250_diagnostic | bio260_pre10_pre5_ecg_peak_count                         | ecg      | 0.0409009  |
| bad_top10_v250_diagnostic | bio260_pre10_pre5_ecg_peak_rate_per_min                  | ecg      | 0.0408764  |
| future_cluster4           | bio260_delta_pre2_0_minus_pre20_pre10_ecg_bpm_from_peaks | ecg      | 0.0079461  |
| future_cluster4           | bio260_pre2_0_ecg_bpm_from_peaks                         | ecg      | 0.00785221 |
| future_cluster4           | bio260_delta_pre2_0_minus_pre5_pre2_ecg_bpm_from_peaks   | ecg      | 0.00762911 |
| future_cluster4           | bio260_pre2_0_ecg_ibi_mean_s                             | ecg      | 0.00750484 |
| future_cluster4           | bio260_delta_pre2_0_minus_pre10_pre5_ecg_bpm_from_peaks  | ecg      | 0.00691053 |
| future_cluster4           | bio260_pre2_0_resp_period_std_s                          | resp     | 0.00587681 |
| future_cluster4           | bio260_pre2_0_ecg_peak_rate_per_min                      | ecg      | 0.00550201 |
| future_cluster4           | bio260_pre2_0_ecg_peak_count                             | ecg      | 0.00548641 |
| future_cluster4           | bio260_pre2_0_ecg_ibi_sdnn_s                             | ecg      | 0.00533167 |
| future_cluster4           | bio260_pre5_0_ecg_bpm_from_peaks                         | ecg      | 0.00500007 |
| future_cluster4           | bio260_pre2_0_emg_z_slope                                | emg      | 0.0049365  |
| future_cluster4           | bio260_pre5_0_scr_burst_rate                             | scr      | 0.00455693 |
| high_future_abs_q75       | bio260_delta_pre2_0_minus_pre10_pre5_ecg_bpm_from_peaks  | ecg      | 0.0090871  |
| high_future_abs_q75       | bio260_pre2_0_hr_z_range                                 | hr       | 0.00813707 |
| high_future_abs_q75       | bio260_pre2_0_ecg_ibi_sdnn_s                             | ecg      | 0.00783054 |
| high_future_abs_q75       | bio260_pre2_0_ecg_peak_rate_per_min                      | ecg      | 0.00758623 |
| high_future_abs_q75       | bio260_pre2_0_ecg_peak_count                             | ecg      | 0.0075844  |
| high_future_abs_q75       | bio260_pre5_0_hr_z_range                                 | hr       | 0.00707991 |
| high_future_abs_q75       | bio260_pre20_pre10_ecg_peak_amp_mean                     | ecg      | 0.00703595 |
| high_future_abs_q75       | bio260_pre5_pre2_ecg_peak_amp_mean                       | ecg      | 0.00694806 |
| high_future_abs_q75       | bio260_pre10_pre5_ecg_peak_amp_mean                      | ecg      | 0.00666372 |
| high_future_abs_q75       | bio260_pre2_0_hr_z_std                                   | hr       | 0.00654422 |
| high_future_abs_q75       | bio260_pre5_pre2_resp_period_mean_s                      | resp     | 0.00615487 |
| high_future_abs_q75       | bio260_pre5_0_ecg_peak_amp_mean                          | ecg      | 0.0060667  |
| subject                   | bio260_pre20_pre10_ecg_peak_amp_mean                     | ecg      | 0.969836   |
| subject                   | bio260_pre10_pre5_ecg_peak_amp_mean                      | ecg      | 0.963737   |
| subject                   | bio260_pre5_0_ecg_peak_amp_mean                          | ecg      | 0.954322   |
| subject                   | bio260_pre20_pre10_ecg_peak_amp_p90                      | ecg      | 0.949641   |
| subject                   | bio260_pre5_pre2_ecg_peak_amp_mean                       | ecg      | 0.945039   |
| subject                   | bio260_pre10_pre5_ecg_peak_amp_p90                       | ecg      | 0.944699   |
| subject                   | bio260_pre5_pre2_ecg_peak_amp_p90                        | ecg      | 0.923535   |
| subject                   | bio260_pre5_0_ecg_peak_amp_p90                           | ecg      | 0.920788   |
| subject                   | bio260_pre2_0_ecg_peak_amp_mean                          | ecg      | 0.909628   |
| subject                   | bio260_pre2_0_ecg_peak_amp_p90                           | ecg      | 0.903329   |
| subject                   | bio260_pre20_pre10_ecg_ibi_mean_s                        | ecg      | 0.692633   |
| subject                   | bio260_pre5_0_ecg_ibi_mean_s                             | ecg      | 0.684801   |

## 判读

- bad_top10 subject-disjoint：vehicle macro-F1=0.5025；bio260=0.4947；vehicle+bio260=0.5120。
- 与 v254b 参考相比：physio200_curated_ref bad_top10 macro-F1=0.4482；bio260_curated=0.4947。
- 若 bio260 明显超过 physio200_curated_ref，说明数据层重构比旧统计更有价值，可进入 v261 selector/预测实验。
- 若 vehicle+bio260 仍不超过 vehicle_only，说明即使事件型 biomarker 也没有形成正式跨驾驶员预测增量。

## 关键图

- `figures\v260_subject_disjoint_test_macro_f1.png`
- `figures\v260_eta2_top_features.png`