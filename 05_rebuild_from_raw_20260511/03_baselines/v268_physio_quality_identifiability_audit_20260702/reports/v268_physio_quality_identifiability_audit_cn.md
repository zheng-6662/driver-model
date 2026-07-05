# v268 physiology quality / alignment / identifiability audit

## 本轮问题

- v267 已经验证更强监督式候选重排仍未达标。
- v268 不再训练新预测模型，而是审计生理链路：源质量、事件窗口覆盖、身份混淆、候选排序可识别性。

## 总体判定

| check                            | status   | evidence                                                      | interpretation                                                                                       |
|:---------------------------------|:---------|:--------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|
| source_timing_integrity          | pass     | median_hz=200.000, gaps=0, duplicates=0                       | 200Hz 连续层时序本身稳定；失败不能简单归咎于采样断裂。                                               |
| derived_signal_availability      | warn     | HRV_RMSSD:0/82; RESP_Amplitude:0/82; RESP_BPM:0/82            | 部分派生生理列不可用或近常数，尤其 HRV_RMSSD、RESP_BPM/Amplitude、部分 EDA；这会削弱高层 biomarker。 |
| event_window_coverage            | pass     | min split-delay ok_rate=0.889                                 | 事件窗口覆盖整体尚可，问题更可能是特征有效性和泛化，而不是大面积缺失。                               |
| identity_vs_behavior_signal      | warn     | median family identity/behavior eta ratio=68.74               | bio260 更容易区分 subject/recording，而不是行为目标；这解释了 subject-disjoint 下增量不稳定。        |
| candidate_rerank_identifiability | warn     | pair_vehicle_bio chosen_minus_latest=+0.1509, top3_rate=0.211 | 即使候选库有 oracle headroom，bio pair 分数也不能稳定把最佳候选排到前面。                            |

## 源 recording 质量

|   recording_n |   subject_n |   duration_s_sum |   duration_s_median |   median_hz_median |   gap_gt_20ms_total |   duplicate_t_total |   negative_or_zero_dt_total |   core_columns_all_present_rate |
|--------------:|------------:|-----------------:|--------------------:|-------------------:|--------------------:|--------------------:|----------------------------:|--------------------------------:|
|            82 |          18 |          42210.4 |              572.32 |                200 |                   0 |                   0 |                           0 |                               1 |

## 信号可用性

| signal         |   recording_count |   usable_basic_count |   usable_basic_rate |   near_constant_count |   all_nan_count |   high_missing_count |
|:---------------|------------------:|---------------------:|--------------------:|----------------------:|----------------:|---------------------:|
| ECG_filt200    |                82 |                   82 |            1        |                     0 |               0 |                    0 |
| ECG_raw200     |                82 |                   82 |            1        |                     0 |               0 |                    0 |
| EDA_Phasic     |                82 |                   73 |            0.890244 |                     9 |               9 |                    9 |
| EDA_Tonic      |                82 |                   73 |            0.890244 |                     9 |               9 |                    9 |
| EDA_filt200    |                82 |                   73 |            0.890244 |                     9 |               0 |                    0 |
| EDA_raw200     |                82 |                   73 |            0.890244 |                     9 |               0 |                    0 |
| EMG_RMS        |                82 |                   82 |            1        |                     0 |               0 |                    0 |
| EMG_filt200    |                82 |                   82 |            1        |                     0 |               0 |                    0 |
| EMG_raw200     |                82 |                   82 |            1        |                     0 |               0 |                    0 |
| HRV_RMSSD      |                82 |                    0 |            0        |                    82 |              82 |                   82 |
| HR_bpm         |                82 |                   82 |            1        |                     0 |               0 |                    0 |
| RESP_Amplitude |                82 |                    0 |            0        |                    82 |               0 |                    0 |
| RESP_BPM       |                82 |                    0 |            0        |                    82 |              17 |                   17 |
| RESP_filt200   |                82 |                   82 |            1        |                     0 |               0 |                    0 |
| RESP_raw200    |                82 |                   82 |            1        |                     0 |               0 |                    0 |

### 按信号族汇总

| family   |   signal_n |   usable_basic_rate_mean |   near_constant_count_sum |   all_nan_count_sum |   high_missing_count_sum |
|:---------|-----------:|-------------------------:|--------------------------:|--------------------:|-------------------------:|
| ecg      |          2 |                 1        |                         0 |                   0 |                        0 |
| eda      |          4 |                 0.890244 |                        36 |                  18 |                       18 |
| emg      |          3 |                 1        |                         0 |                   0 |                        0 |
| hr       |          1 |                 1        |                         0 |                   0 |                        0 |
| hrv      |          1 |                 0        |                        82 |                  82 |                       82 |
| resp     |          4 |                 0.5      |                       164 |                  17 |                       17 |

## 事件窗口覆盖

| split   |   delay_ms |   row_n |   event_n |   ok_rate |   post_observation_rate |   baseline_rows_mean |   baseline_duration_s_mean |   recording_duration_s_median |
|:--------|-----------:|--------:|----------:|----------:|------------------------:|---------------------:|---------------------------:|------------------------------:|
| test    |          0 |     184 |       184 |  0.896739 |                       0 |              7968.73 |                    39.8386 |                       583.815 |
| test    |       1000 |     184 |       184 |  0.896739 |                       0 |              7974.32 |                    39.8666 |                       583.815 |
| train   |          0 |     674 |       674 |  0.888724 |                       0 |              7957.36 |                    39.7818 |                       585.54  |
| train   |       1000 |     674 |       674 |  0.888724 |                       0 |              7961.1  |                    39.8005 |                       585.54  |
| val     |          0 |     309 |       309 |  1        |                       0 |              7964.67 |                    39.8183 |                       614.765 |
| val     |       1000 |     309 |       309 |  1        |                       0 |              7969.2  |                    39.841  |                       614.765 |

## 事件特征缺失按信号族

| family   |   feature_n |   missing_rate_all_mean |   missing_rate_ok_rows_mean |   zero_variance_feature_n |
|:---------|------------:|------------------------:|----------------------------:|--------------------------:|
| ecg      |          43 |               0.103751  |                   0.0252353 |                         0 |
| emg      |          38 |               0.0805484 |                   0         |                         0 |
| hr       |          33 |               0.0805484 |                   0         |                         0 |
| hrv      |          30 |               1         |                   1         |                        30 |
| other    |           1 |               0         |                   0         |                         0 |
| resp     |          51 |               0.214325  |                   0.145496  |                         0 |
| scr      |          38 |               0.22365   |                   0.155638  |                         0 |

## bio260 身份信号 vs 行为信号

| family   |   feature_n |   identity_eta_max_mean |   behavior_eta_max_mean |   identity_to_behavior_ratio_median |   features_identity_gt_behavior_5x |   features_behavior_eta_ge_0p02 |
|:---------|------------:|------------------------:|------------------------:|------------------------------------:|-----------------------------------:|--------------------------------:|
| ecg      |           5 |                0.211186 |              0.00185659 |                             97.62   |                                  5 |                               0 |
| emg      |          12 |                0.125879 |              0.00528782 |                             28.0056 |                                 12 |                               0 |
| hr       |          10 |                0.126504 |              0.00205019 |                             68.7352 |                                 10 |                               0 |
| resp     |          19 |                0.203305 |              0.00192796 |                            187.659  |                                 19 |                               0 |
| scr      |          18 |                0.108992 |              0.00194713 |                             49.5886 |                                 18 |                               0 |

## v267 候选排序可识别性

| split   | bad_top10   | score                                 |   event_n |   chosen_rmse_mean |   best_candidate_rmse_mean |   chosen_minus_best_mean |   chosen_minus_latest_mean |   true_best_rank_mean |   true_best_top3_rate |   spearman_score_vs_target_rmse_mean |
|:--------|:------------|:--------------------------------------|----------:|-------------------:|---------------------------:|-------------------------:|---------------------------:|----------------------:|----------------------:|-------------------------------------:|
| test    | True        | bio_distance                          |        19 |           0.798872 |                   0.616603 |                 0.18227  |                   0.103824 |               8.10526 |             0.263158  |                           0.0131948  |
| test    | True        | pred_pair_vehicle_bio_hgb             |        19 |           0.845988 |                   0.616603 |                 0.229385 |                   0.150939 |              13.5263  |             0.210526  |                           0.133085   |
| test    | True        | pred_pair_bio_hgb                     |        19 |           0.85961  |                   0.616603 |                 0.243007 |                   0.164562 |              11       |             0.210526  |                           0.201955   |
| test    | True        | pred_pair_vehicle_bio_badweighted_hgb |        19 |           0.862965 |                   0.616603 |                 0.246362 |                   0.167917 |              13.7368  |             0.0526316 |                           0.178767   |
| test    | True        | pred_pair_vehicle_hgb                 |        19 |           0.874617 |                   0.616603 |                 0.258014 |                   0.179568 |              12.6842  |             0.368421  |                           0.169898   |
| test    | True        | vehicle_distance                      |        19 |           0.878536 |                   0.616603 |                 0.261933 |                   0.183488 |              10.6842  |             0.210526  |                          -0.00780694 |

## 结论

- 200Hz 源时序质量基本稳定，不能把失败简单归因于采样断裂。
- 但派生生理列存在结构性弱点：HRV_RMSSD 全不可用，RESP_BPM/RESP_Amplitude 基本不可用，EDA 有一部分 recording 近常数/全缺。
- 事件级 bio260 覆盖率尚可，post-observation guardrail 通过；核心问题更偏向特征有效性和 subject-disjoint 可迁移性。
- 身份/recording 可分性高于行为/等待收益可分性，说明 bio260 在跨驾驶员泛化时更容易携带个体/设备/记录差异。
- v267 候选库虽然有 oracle headroom，但 bio/pair 分数不能稳定把最佳候选排到前面；这解释了为什么更强 reranker 仍不能超过 fixed wait-latest。

## 关键图

- `figures\v268_signal_availability.png`
- `figures\v268_identity_vs_behavior_eta.png`
- `figures\v268_test_badtop10_candidate_rank_quality.png`