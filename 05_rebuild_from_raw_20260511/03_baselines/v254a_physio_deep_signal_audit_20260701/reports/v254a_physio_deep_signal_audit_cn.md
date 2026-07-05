# v254a 生理信号深层挖掘审计

## 本轮问题

本轮不把生理数据简单拼接进轨迹模型，而是先检查生理数据是否真的含有可学习的驾驶员状态结构，以及这种结构是否与未来驾驶行为有关。

## 数据与边界

- 生理主输入：`physio_features_10hz.csv`，来自清洗后 200Hz 连续生理层的 10Hz 聚合特征。
- 对照输入：v253a 已提取的 1Hz 锚点前窗口生理特征。
- 事件样本：复用 v252/v253 固定 rolling sample 和 split。
- 诊断模型：只训练轻量 logistic/ridge 分类或回归头，不训练轨迹预测模型。
- 训练口径：所有诊断模型只在 train 拟合，val/test 只报告，不用于调参。

## 对齐覆盖

| split   |    n |   recording_inventory_rate |   recording_has_10hz_rate |   uses_post_observation_rate |   pre20_pre10_rows_mean |   pre20_pre10_rows_p10 |   pre20_pre10_rows_p50 |   pre20_pre10_rows_p90 |   pre10_pre5_rows_mean |   pre10_pre5_rows_p10 |   pre10_pre5_rows_p50 |   pre10_pre5_rows_p90 |   pre5_pre2_rows_mean |   pre5_pre2_rows_p10 |   pre5_pre2_rows_p50 |   pre5_pre2_rows_p90 |   pre2_0_rows_mean |   pre2_0_rows_p10 |   pre2_0_rows_p50 |   pre2_0_rows_p90 |
|:--------|-----:|---------------------------:|--------------------------:|-----------------------------:|------------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|---------------------:|---------------------:|---------------------:|-------------------:|------------------:|------------------:|------------------:|
| test    | 1104 |                   0.896739 |                  0.896739 |                            0 |                 100.849 |                    100 |                    101 |                    101 |                50.8586 |                    50 |                    51 |                    51 |               30.8646 |                   30 |                   31 |                   31 |            20.8697 |                20 |                21 |                21 |
| train   | 4044 |                   0.888724 |                  0.888724 |                            0 |                 100.868 |                    100 |                    101 |                    101 |                50.8848 |                    50 |                    51 |                    51 |               30.8926 |                   30 |                   31 |                   31 |            20.8979 |                20 |                21 |                21 |
| val     | 1854 |                   1        |                  1        |                            0 |                 100.866 |                    100 |                    101 |                    101 |                50.8846 |                    50 |                    51 |                    51 |               30.8932 |                   30 |                   31 |                   31 |            20.8991 |                20 |                21 |                21 |

## 特征块维度

| feature_block           |   raw_dim |   kept_dim |   kept_physio_columns |
|:------------------------|----------:|-----------:|----------------------:|
| vehicle_only            |       268 |        268 |                     0 |
| physio10hz_deep         |       840 |        700 |                   700 |
| physio1hz_v253a         |       209 |        184 |                   184 |
| vehicle_plus_physio10hz |      1108 |        968 |                   700 |

## 生理信号质量摘要

| signal         |   n_features |   finite_rate_train_mean |   train_std_median |   near_constant_feature_rate |
|:---------------|-------------:|-------------------------:|-------------------:|-----------------------------:|
| ECG_filt200    |           56 |                0.888724  |         0.0326417  |                    0.0714286 |
| ECG_raw200     |           56 |                0.888724  |         0.0355176  |                    0.0714286 |
| EDA_filt200    |           56 |                0.888724  |         1.2737     |                    0.0714286 |
| EMG_RMS        |           56 |                0.888724  |         0.0186289  |                    0.0714286 |
| EDA_raw200     |           56 |                0.888724  |         6.77743    |                    0.0714286 |
| EMG_raw200     |           56 |                0.888724  |         0.00290322 |                    0.0714286 |
| EMG_filt200    |           56 |                0.888724  |         0.00264271 |                    0.0714286 |
| RESP_raw200    |           56 |                0.888724  |         0.159149   |                    0.0714286 |
| RESP_filt200   |           56 |                0.888724  |         0.158476   |                    0.0714286 |
| HR_bpm         |           56 |                0.888724  |         8.18246    |                    0.0714286 |
| RESP_Amplitude |           56 |                0.888724  |         1.26483    |                    0.428571  |
| RESP_BPM       |           56 |                0.840504  |         2.99415    |                    0.357143  |
| EDA_Phasic     |           56 |                0.748198  |         0.434794   |                    0         |
| EDA_Tonic      |           56 |                0.748198  |         6.31696    |                    0         |
| HRV_RMSSD      |           56 |                0.0634803 |         0          |                    1         |

## 未来轨迹聚类摘要

| cluster_target   |   cluster_id |   n_all |   n_train |   n_val |   n_test |   train_future_peak_abs_mean |   train_future_range_mean |   train_future_final_mean |
|:-----------------|-------------:|--------:|----------:|--------:|---------:|-----------------------------:|--------------------------:|--------------------------:|
| future_cluster4  |            0 |    1418 |       833 |     396 |      189 |                     2.25866  |                  2.36363  |                 -1.37784  |
| future_cluster4  |            1 |    1006 |       512 |     329 |      165 |                     2.12535  |                  2.22601  |                  0.965655 |
| future_cluster4  |            2 |    2673 |      1607 |     616 |      450 |                     1.00624  |                  1.20908  |                 -0.688915 |
| future_cluster4  |            3 |    1905 |      1092 |     513 |      300 |                     1.64449  |                  1.86877  |                  1.28072  |
| future_cluster6  |            0 |    1019 |       559 |     312 |      148 |                     1.55461  |                  1.75179  |                 -0.419373 |
| future_cluster6  |            1 |    2081 |      1224 |     503 |      354 |                     0.727648 |                  0.924952 |                 -0.107312 |
| future_cluster6  |            2 |    1526 |       870 |     406 |      250 |                     1.88872  |                  2.09063  |                  1.46382  |
| future_cluster6  |            3 |    1316 |       830 |     288 |      198 |                     1.86266  |                  2.01813  |                 -1.52147  |
| future_cluster6  |            4 |     211 |       129 |      60 |       22 |                     3.90672  |                  3.95705  |                 -3.10292  |
| future_cluster6  |            5 |     849 |       432 |     285 |      132 |                     2.15191  |                  2.26057  |                  0.910868 |

## 行为分类诊断

| task_type      | target                | feature_block           | eval_split   |   n_eval |   accuracy |   macro_f1 |        auc |   vehicle_metric |   delta_macro_f1_minus_vehicle |
|:---------------|:----------------------|:------------------------|:-------------|---------:|-----------:|-----------:|-----------:|-----------------:|-------------------------------:|
| classification | future_cluster4       | vehicle_only            | val          |     1854 |   0.700108 |   0.703472 | nan        |         0.703472 |                      0         |
| classification | future_cluster4       | vehicle_only            | test         |     1104 |   0.743659 |   0.731745 | nan        |         0.731745 |                      0         |
| classification | future_cluster4       | physio10hz_deep         | val          |     1854 |   0.272923 |   0.269765 | nan        |         0.703472 |                     -0.433707  |
| classification | future_cluster4       | physio10hz_deep         | test         |     1104 |   0.317029 |   0.294405 | nan        |         0.731745 |                     -0.437339  |
| classification | future_cluster4       | physio1hz_v253a         | val          |     1854 |   0.271305 |   0.255234 | nan        |         0.703472 |                     -0.448238  |
| classification | future_cluster4       | physio1hz_v253a         | test         |     1104 |   0.310688 |   0.295411 | nan        |         0.731745 |                     -0.436334  |
| classification | future_cluster4       | vehicle_plus_physio10hz | val          |     1854 |   0.563107 |   0.563087 | nan        |         0.703472 |                     -0.140385  |
| classification | future_cluster4       | vehicle_plus_physio10hz | test         |     1104 |   0.519022 |   0.502035 | nan        |         0.731745 |                     -0.22971   |
| classification | high_future_abs_q75   | vehicle_only            | val          |     1854 |   0.740561 |   0.720299 |   0.786269 |         0.720299 |                      0         |
| classification | high_future_abs_q75   | vehicle_only            | test         |     1104 |   0.796196 |   0.711168 |   0.772184 |         0.711168 |                      0         |
| classification | high_future_abs_q75   | physio10hz_deep         | val          |     1854 |   0.491909 |   0.470356 |   0.489615 |         0.720299 |                     -0.249943  |
| classification | high_future_abs_q75   | physio10hz_deep         | test         |     1104 |   0.549819 |   0.489676 |   0.507186 |         0.711168 |                     -0.221492  |
| classification | high_future_abs_q75   | physio1hz_v253a         | val          |     1854 |   0.569579 |   0.554048 |   0.582552 |         0.720299 |                     -0.166252  |
| classification | high_future_abs_q75   | physio1hz_v253a         | test         |     1104 |   0.400362 |   0.392449 |   0.524543 |         0.711168 |                     -0.318719  |
| classification | high_future_abs_q75   | vehicle_plus_physio10hz | val          |     1854 |   0.62945  |   0.594682 |   0.634545 |         0.720299 |                     -0.125617  |
| classification | high_future_abs_q75   | vehicle_plus_physio10hz | test         |     1104 |   0.734601 |   0.623876 |   0.60986  |         0.711168 |                     -0.0872915 |
| classification | high_future_range_q75 | vehicle_only            | val          |     1854 |   0.701726 |   0.690898 |   0.764224 |         0.690898 |                      0         |
| classification | high_future_range_q75 | vehicle_only            | test         |     1104 |   0.742754 |   0.658648 |   0.701055 |         0.658648 |                      0         |
| classification | high_future_range_q75 | physio10hz_deep         | val          |     1854 |   0.535059 |   0.501656 |   0.511024 |         0.690898 |                     -0.189242  |
| classification | high_future_range_q75 | physio10hz_deep         | test         |     1104 |   0.514493 |   0.46161  |   0.494686 |         0.658648 |                     -0.197037  |
| classification | high_future_range_q75 | physio1hz_v253a         | val          |     1854 |   0.561489 |   0.547937 |   0.580306 |         0.690898 |                     -0.142961  |
| classification | high_future_range_q75 | physio1hz_v253a         | test         |     1104 |   0.423007 |   0.415992 |   0.543192 |         0.658648 |                     -0.242656  |
| classification | high_future_range_q75 | vehicle_plus_physio10hz | val          |     1854 |   0.639698 |   0.598173 |   0.631361 |         0.690898 |                     -0.0927255 |
| classification | high_future_range_q75 | vehicle_plus_physio10hz | test         |     1104 |   0.702899 |   0.601935 |   0.579146 |         0.658648 |                     -0.0567132 |
| classification | strong_steer_existing | vehicle_only            | val          |     1854 |   0.683387 |   0.669882 |   0.719877 |         0.669882 |                      0         |
| classification | strong_steer_existing | vehicle_only            | test         |     1104 |   0.67029  |   0.663034 |   0.712557 |         0.663034 |                      0         |
| classification | strong_steer_existing | physio10hz_deep         | val          |     1854 |   0.470334 |   0.466844 |   0.483048 |         0.669882 |                     -0.203037  |
| classification | strong_steer_existing | physio10hz_deep         | test         |     1104 |   0.569746 |   0.548217 |   0.53548  |         0.663034 |                     -0.114817  |
| classification | strong_steer_existing | physio1hz_v253a         | val          |     1854 |   0.569579 |   0.561608 |   0.582697 |         0.669882 |                     -0.108274  |
| classification | strong_steer_existing | physio1hz_v253a         | test         |     1104 |   0.526268 |   0.521605 |   0.581527 |         0.663034 |                     -0.141429  |
| classification | strong_steer_existing | vehicle_plus_physio10hz | val          |     1854 |   0.564725 |   0.562335 |   0.594038 |         0.669882 |                     -0.107547  |
| classification | strong_steer_existing | vehicle_plus_physio10hz | test         |     1104 |   0.621377 |   0.588052 |   0.613889 |         0.663034 |                     -0.0749818 |

## 未来摘要回归诊断

| task_type   | target          | feature_block           | eval_split   |   n_eval |         r2 |      mae |   vehicle_metric |   delta_r2_minus_vehicle |
|:------------|:----------------|:------------------------|:-------------|---------:|-----------:|---------:|-----------------:|-------------------------:|
| regression  | future_peak_abs | vehicle_only            | val          |     1854 | -0.0439201 | 0.742314 |       -0.0439201 |                 0        |
| regression  | future_peak_abs | vehicle_only            | test         |     1104 | -0.064     | 0.563296 |       -0.064     |                 0        |
| regression  | future_peak_abs | physio10hz_deep         | val          |     1854 | -4.64226   | 1.54149  |       -0.0439201 |                -4.59834  |
| regression  | future_peak_abs | physio10hz_deep         | test         |     1104 | -2.68149   | 1.18778  |       -0.064     |                -2.61749  |
| regression  | future_peak_abs | physio1hz_v253a         | val          |     1854 | -1.4102    | 1.1282   |       -0.0439201 |                -1.36628  |
| regression  | future_peak_abs | physio1hz_v253a         | test         |     1104 | -0.669083  | 0.866647 |       -0.064     |                -0.605083 |
| regression  | future_peak_abs | vehicle_plus_physio10hz | val          |     1854 | -1.46579   | 1.13855  |       -0.0439201 |                -1.42187  |
| regression  | future_peak_abs | vehicle_plus_physio10hz | test         |     1104 | -1.08441   | 0.846181 |       -0.064     |                -1.02041  |
| regression  | future_range    | vehicle_only            | val          |     1854 |  0.0939517 | 0.836336 |        0.0939517 |                 0        |
| regression  | future_range    | vehicle_only            | test         |     1104 | -0.228292  | 0.657307 |       -0.228292  |                 0        |
| regression  | future_range    | physio10hz_deep         | val          |     1854 | -3.63459   | 1.68366  |        0.0939517 |                -3.72855  |
| regression  | future_range    | physio10hz_deep         | test         |     1104 | -3.02768   | 1.36158  |       -0.228292  |                -2.79939  |
| regression  | future_range    | physio1hz_v253a         | val          |     1854 | -1.25067   | 1.26774  |        0.0939517 |                -1.34462  |
| regression  | future_range    | physio1hz_v253a         | test         |     1104 | -0.618307  | 0.949695 |       -0.228292  |                -0.390015 |
| regression  | future_range    | vehicle_plus_physio10hz | val          |     1854 | -1.47822   | 1.31367  |        0.0939517 |                -1.57217  |
| regression  | future_range    | vehicle_plus_physio10hz | test         |     1104 | -1.58331   | 1.0434   |       -0.228292  |                -1.35501  |
| regression  | future_mean_abs | vehicle_only            | val          |     1854 | -0.165788  | 0.42148  |       -0.165788  |                 0        |
| regression  | future_mean_abs | vehicle_only            | test         |     1104 | -0.413613  | 0.333427 |       -0.413613  |                 0        |
| regression  | future_mean_abs | physio10hz_deep         | val          |     1854 | -5.46637   | 0.861501 |       -0.165788  |                -5.30058  |
| regression  | future_mean_abs | physio10hz_deep         | test         |     1104 | -1.82723   | 0.62561  |       -0.413613  |                -1.41362  |
| regression  | future_mean_abs | physio1hz_v253a         | val          |     1854 | -1.74311   | 0.636745 |       -0.165788  |                -1.57732  |
| regression  | future_mean_abs | physio1hz_v253a         | test         |     1104 | -0.307678  | 0.464575 |       -0.413613  |                 0.105935 |
| regression  | future_mean_abs | vehicle_plus_physio10hz | val          |     1854 | -1.48213   | 0.624829 |       -0.165788  |                -1.31635  |
| regression  | future_mean_abs | vehicle_plus_physio10hz | test         |     1104 | -0.928852  | 0.464709 |       -0.413613  |                -0.515239 |

## 可分性 eta^2 Top 特征

| target                | feature                                      | signal         |       eta2 |
|:----------------------|:---------------------------------------------|:---------------|-----------:|
| future_cluster4       | physio10_pre20_pre10_EDA_Tonic_valid_ratio   | EDA_Tonic      | 0.0152023  |
| future_cluster4       | physio10_pre20_pre10_EDA_Phasic_valid_ratio  | EDA_Phasic     | 0.0152023  |
| future_cluster4       | physio10_pre10_pre5_EDA_Tonic_valid_ratio    | EDA_Tonic      | 0.0152023  |
| future_cluster4       | physio10_pre10_pre5_EDA_Phasic_valid_ratio   | EDA_Phasic     | 0.0152023  |
| future_cluster4       | physio10_pre5_pre2_EDA_Tonic_valid_ratio     | EDA_Tonic      | 0.0152023  |
| future_cluster4       | physio10_pre5_pre2_EDA_Phasic_valid_ratio    | EDA_Phasic     | 0.0152023  |
| future_cluster4       | physio10_pre2_0_EDA_Tonic_valid_ratio        | EDA_Tonic      | 0.0152023  |
| future_cluster4       | physio10_pre2_0_EDA_Phasic_valid_ratio       | EDA_Phasic     | 0.0152023  |
| future_cluster4       | physio10_pre5_pre2_RESP_raw200_mean          | RESP_raw200    | 0.0102806  |
| future_cluster4       | physio10_pre2_0_EMG_RMS_last_minus_first     | EMG_RMS        | 0.0101001  |
| future_cluster4       | physio10_pre2_0_EMG_RMS_slope                | EMG_RMS        | 0.0100433  |
| future_cluster4       | physio10_pre20_pre10_RESP_raw200_mean        | RESP_raw200    | 0.00868572 |
| high_future_abs_q75   | physio10_pre10_pre5_RESP_raw200_p90          | RESP_raw200    | 0.019668   |
| high_future_abs_q75   | physio10_pre20_pre10_RESP_raw200_p90         | RESP_raw200    | 0.0194414  |
| high_future_abs_q75   | physio10_pre2_0_RESP_raw200_p90              | RESP_raw200    | 0.0180906  |
| high_future_abs_q75   | physio10_pre20_pre10_EDA_Tonic_valid_ratio   | EDA_Tonic      | 0.0173232  |
| high_future_abs_q75   | physio10_pre20_pre10_EDA_Phasic_valid_ratio  | EDA_Phasic     | 0.0173232  |
| high_future_abs_q75   | physio10_pre10_pre5_EDA_Tonic_valid_ratio    | EDA_Tonic      | 0.0173232  |
| high_future_abs_q75   | physio10_pre10_pre5_EDA_Phasic_valid_ratio   | EDA_Phasic     | 0.0173232  |
| high_future_abs_q75   | physio10_pre5_pre2_EDA_Tonic_valid_ratio     | EDA_Tonic      | 0.0173232  |
| high_future_abs_q75   | physio10_pre5_pre2_EDA_Phasic_valid_ratio    | EDA_Phasic     | 0.0173232  |
| high_future_abs_q75   | physio10_pre2_0_EDA_Tonic_valid_ratio        | EDA_Tonic      | 0.0173232  |
| high_future_abs_q75   | physio10_pre2_0_EDA_Phasic_valid_ratio       | EDA_Phasic     | 0.0173232  |
| high_future_abs_q75   | physio10_pre10_pre5_RESP_filt200_range       | RESP_filt200   | 0.0137623  |
| high_future_range_q75 | physio10_pre20_pre10_RESP_raw200_p90         | RESP_raw200    | 0.0282205  |
| high_future_range_q75 | physio10_pre10_pre5_RESP_raw200_p90          | RESP_raw200    | 0.0262606  |
| high_future_range_q75 | physio10_pre20_pre10_RESP_filt200_p90        | RESP_filt200   | 0.0212386  |
| high_future_range_q75 | physio10_pre20_pre10_EDA_Tonic_valid_ratio   | EDA_Tonic      | 0.0208955  |
| high_future_range_q75 | physio10_pre20_pre10_EDA_Phasic_valid_ratio  | EDA_Phasic     | 0.0208955  |
| high_future_range_q75 | physio10_pre10_pre5_EDA_Tonic_valid_ratio    | EDA_Tonic      | 0.0208955  |
| high_future_range_q75 | physio10_pre10_pre5_EDA_Phasic_valid_ratio   | EDA_Phasic     | 0.0208955  |
| high_future_range_q75 | physio10_pre5_pre2_EDA_Tonic_valid_ratio     | EDA_Tonic      | 0.0208955  |
| high_future_range_q75 | physio10_pre5_pre2_EDA_Phasic_valid_ratio    | EDA_Phasic     | 0.0208955  |
| high_future_range_q75 | physio10_pre2_0_EDA_Tonic_valid_ratio        | EDA_Tonic      | 0.0208955  |
| high_future_range_q75 | physio10_pre2_0_EDA_Phasic_valid_ratio       | EDA_Phasic     | 0.0208955  |
| high_future_range_q75 | physio10_pre10_pre5_RESP_filt200_range       | RESP_filt200   | 0.0200861  |
| recording             | physio10_pre20_pre10_RESP_BPM_mean           | RESP_BPM       | 1          |
| recording             | physio10_pre20_pre10_RESP_BPM_abs_mean       | RESP_BPM       | 1          |
| recording             | physio10_pre20_pre10_RESP_Amplitude_mean     | RESP_Amplitude | 1          |
| recording             | physio10_pre20_pre10_RESP_Amplitude_p10      | RESP_Amplitude | 1          |
| recording             | physio10_pre20_pre10_RESP_Amplitude_p50      | RESP_Amplitude | 1          |
| recording             | physio10_pre20_pre10_RESP_Amplitude_p90      | RESP_Amplitude | 1          |
| recording             | physio10_pre20_pre10_RESP_Amplitude_abs_mean | RESP_Amplitude | 1          |
| recording             | physio10_pre20_pre10_RESP_Amplitude_first    | RESP_Amplitude | 1          |
| recording             | physio10_pre20_pre10_RESP_Amplitude_last     | RESP_Amplitude | 1          |
| recording             | physio10_pre10_pre5_RESP_Amplitude_mean      | RESP_Amplitude | 1          |
| recording             | physio10_pre10_pre5_RESP_Amplitude_p10       | RESP_Amplitude | 1          |
| recording             | physio10_pre10_pre5_RESP_Amplitude_p50       | RESP_Amplitude | 1          |
| strong_steer_existing | physio10_pre20_pre10_EDA_Tonic_valid_ratio   | EDA_Tonic      | 0.0233727  |
| strong_steer_existing | physio10_pre20_pre10_EDA_Phasic_valid_ratio  | EDA_Phasic     | 0.0233727  |
| strong_steer_existing | physio10_pre10_pre5_EDA_Tonic_valid_ratio    | EDA_Tonic      | 0.0233727  |
| strong_steer_existing | physio10_pre10_pre5_EDA_Phasic_valid_ratio   | EDA_Phasic     | 0.0233727  |
| strong_steer_existing | physio10_pre5_pre2_EDA_Tonic_valid_ratio     | EDA_Tonic      | 0.0233727  |
| strong_steer_existing | physio10_pre5_pre2_EDA_Phasic_valid_ratio    | EDA_Phasic     | 0.0233727  |
| strong_steer_existing | physio10_pre2_0_EDA_Tonic_valid_ratio        | EDA_Tonic      | 0.0233727  |
| strong_steer_existing | physio10_pre2_0_EDA_Phasic_valid_ratio       | EDA_Phasic     | 0.0233727  |
| strong_steer_existing | physio10_pre20_pre10_RESP_raw200_p90         | RESP_raw200    | 0.0167631  |
| strong_steer_existing | physio10_pre5_pre2_ECG_filt200_p10           | ECG_filt200    | 0.0144732  |
| strong_steer_existing | physio10_pre10_pre5_RESP_raw200_p90          | RESP_raw200    | 0.0138691  |
| strong_steer_existing | physio10_pre2_0_EMG_RMS_range                | EMG_RMS        | 0.0138369  |
| subject               | physio10_pre20_pre10_EDA_Tonic_valid_ratio   | EDA_Tonic      | 1          |
| subject               | physio10_pre20_pre10_EDA_Phasic_valid_ratio  | EDA_Phasic     | 1          |
| subject               | physio10_pre10_pre5_EDA_Tonic_valid_ratio    | EDA_Tonic      | 1          |
| subject               | physio10_pre10_pre5_EDA_Phasic_valid_ratio   | EDA_Phasic     | 1          |
| subject               | physio10_pre5_pre2_EDA_Tonic_valid_ratio     | EDA_Tonic      | 1          |
| subject               | physio10_pre5_pre2_EDA_Phasic_valid_ratio    | EDA_Phasic     | 1          |
| subject               | physio10_pre2_0_EDA_Tonic_valid_ratio        | EDA_Tonic      | 1          |
| subject               | physio10_pre2_0_EDA_Phasic_valid_ratio       | EDA_Phasic     | 1          |
| subject               | physio10_pre20_pre10_ECG_filt200_p90         | ECG_filt200    | 0.885877   |
| subject               | physio10_pre20_pre10_RESP_BPM_mean           | RESP_BPM       | 0.867871   |
| subject               | physio10_pre20_pre10_RESP_BPM_abs_mean       | RESP_BPM       | 0.867871   |
| subject               | physio10_pre10_pre5_RESP_BPM_rms             | RESP_BPM       | 0.867871   |

## 判读规则

- 如果 `physio10hz_deep` 自己能预测未来行为标签，说明生理中有行为相关状态信号。
- 如果 `vehicle_plus_physio10hz` 明显优于 `vehicle_only`，说明生理对车辆输入有增量。
- 如果 subject/recording 的 eta^2 很高但未来行为 eta^2 很低，说明生理更像身份/记录状态，而不是当前任务可用的行为状态。
- 如果 10Hz 明显优于 1Hz，说明 v253a/v253b 的失败可能来自 1Hz 特征太粗。

## 关键图

- `figures\v254a_physio10hz_window_rows.png`
- `figures\v254a_top_physio_eta2.png`
- `figures\v254a_behavior_classification_macro_f1.png`
- `figures\v254a_future_summary_regression_r2.png`
