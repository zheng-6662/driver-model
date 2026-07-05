# v287 physiology temporal-window route audit

## 本轮目的

- v285 整体 raw 200Hz shape-state route gate 失败后，检查是否存在被混合稀释的有效时间窗口或信号族。
- 本轮不重新抽波形、不训练轨迹融合模型，只复用 v285 causal 特征和 train-only screen。
- 正式部署判断仍只看 validation 选择后的 deployable top1。

## route gate 判定

| check                                             | requirement                                                                    | pass   | evidence                    | deployable   | route_viable_now   |
|:--------------------------------------------------|:-------------------------------------------------------------------------------|:-------|:----------------------------|:-------------|:-------------------|
| deployable_top1_val_chosen_bad_top10              | validation 选出的新生理 top1 在 test bad_top10 上低于 latest                   | False  | 0.2378521702791515          | True         | False              |
| deployable_top1_val_chosen_bad_ambiguous          | validation 选出的新生理 top1 在 test bad_top10_vehicle_ambiguous 上低于 latest | False  | 0.23142589926719664         | True         | False              |
| oracle_top3_val_test_same_direction_bad_ambiguous | 非部署 top3 上限在 val/test 歧义差样本上同向改善                               | False  | val=0.054089, test=0.036843 | False        | False              |
| test_bad_top10_any_feature_corr_gt_005            | test bad_top10 至少一个新特征集的生理距离-真实误差排序相关均值 > 0.05          | True   | 0.08543510556920011         | False        | False              |
| test_best_top1_diagnostic_beats_latest            | 即使 test-best 诊断，新生理 top1 至少有一个特征集低于 latest                   | False  | 0.0941408119703594          | False        | False              |

## 各分组类型最好结果

| event_group                 | group_type    | best_top1_feature_set   | best_top1_group_value   |   best_top1_delta |   best_top3_delta | best_corr_feature_set   | best_corr_group_value   |   best_corr_mean |
|:----------------------------|:--------------|:------------------------|:------------------------|------------------:|------------------:|:------------------------|:------------------------|-----------------:|
| bad_top10                   | window_signal | combo_pre2_0_ecg_top16  | pre2_0|ecg              |         0.0941408 |        0.0089152  | combo_pre1_0_ecg_top16  | pre1_0|ecg              |        0.0854351 |
| bad_top10                   | window        | win_pre10_pre5_top32    | pre10_pre5              |         0.114419  |       -0.00272008 | win_pre5_pre2_top32     | pre5_pre2               |        0.0163337 |
| bad_top10                   | category      | category_coupling_top32 | coupling                |         0.158564  |        0.021559   | category_coupling_top32 | coupling                |        0.037308  |
| bad_top10                   | signal        | signal_resp_top32       | resp                    |         0.178303  |       -0.0158745  | signal_ecg_top32        | ecg                     |        0.0467051 |
| bad_top10_vehicle_ambiguous | window_signal | combo_pre2_0_ecg_top16  | pre2_0|ecg              |         0.100183  |        0.00903801 | combo_pre1_0_ecg_top16  | pre1_0|ecg              |        0.121185  |
| bad_top10_vehicle_ambiguous | window        | win_pre10_pre5_top32    | pre10_pre5              |         0.127392  |       -0.0133427  | win_pre5_pre2_top32     | pre5_pre2               |        0.0272898 |
| bad_top10_vehicle_ambiguous | category      | category_coupling_top32 | coupling                |         0.146092  |        0.0325913  | category_quality_top32  | quality                 |        0.0433739 |
| bad_top10_vehicle_ambiguous | signal        | signal_resp_top32       | resp                    |         0.160308  |       -0.0201666  | signal_ecg_top32        | ecg                     |        0.0588446 |

## validation 选择后的 test 泛化

| event_group                 | method          | deployable   | val_chosen_feature_set   |   val_delta_vs_latest_mean |   test_delta_vs_latest_mean |   test_corr_mean | test_passes_latest   | val_and_test_same_direction_gain   |
|:----------------------------|:----------------|:-------------|:-------------------------|---------------------------:|----------------------------:|-----------------:|:---------------------|:-----------------------------------|
| all                         | bio_top1        | True         | signal_eda_top32         |                 0.0989921  |                 0.0848877   |     -0.00667864  | False                | False                              |
| all                         | bio_top3_oracle | False        | combo_pre10_0_eda_top16  |                -0.0137594  |                 0.00184017  |      0.00188173  | False                | False                              |
| all                         | bio_top5_oracle | False        | combo_pre10_0_eda_top16  |                -0.0471875  |                -0.0225288   |      0.00188173  | True                 | True                               |
| vehicle_ambiguous           | bio_top1        | True         | signal_eda_top32         |                 0.124401   |                 0.0939845   |     -0.0144526   | False                | False                              |
| vehicle_ambiguous           | bio_top3_oracle | False        | combo_pre10_0_eda_top16  |                -0.00273023 |                 0.00620509  |     -0.00831309  | False                | False                              |
| vehicle_ambiguous           | bio_top5_oracle | False        | combo_pre10_0_eda_top16  |                -0.0349737  |                -0.0202212   |     -0.00831309  | True                 | True                               |
| bad_top10                   | bio_top1        | True         | signal_eda_top32         |                 0.338741   |                 0.237852    |     -0.057164    | False                | False                              |
| bad_top10                   | bio_top3_oracle | False        | combo_pre10_0_eda_top16  |                 0.0645711  |                 0.0388821   |     -0.032485    | False                | False                              |
| bad_top10                   | bio_top5_oracle | False        | combo_pre10_0_eda_top16  |                 0.0155419  |                 1.13359e-05 |     -0.032485    | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top1        | True         | signal_eda_top32         |                 0.370581   |                 0.231426    |     -0.0519917   | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top3_oracle | False        | combo_pre10_0_eda_top16  |                 0.0540891  |                 0.036843    |     -0.000980844 | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top5_oracle | False        | combo_pre10_0_eda_top16  |                 0.014438   |                -0.0123933   |     -0.000980844 | True                 | False                              |

## test bad_top10 top feature sets

| feature_set                              | group_type    | group_value                    |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_best_in_top3_rate |
|:-----------------------------------------|:--------------|:-------------------------------|----:|-------------------:|---------------------:|-----------------------------:|-----------------------------:|----------------:|------------------------:|
| combo_pre2_0_ecg_top16                   | window_signal | pre2_0|ecg                     |  19 |           0.695048 |             0.789189 |                    0.0941408 |                  0.0089152   |     0.0781152   |               0.157895  |
| win_pre10_pre5_top32                     | window        | pre10_pre5                     |  19 |           0.695048 |             0.809467 |                    0.114419  |                 -0.00272008  |     0.00891733  |               0.105263  |
| combo_pre10_pre5_resp_top16              | window_signal | pre10_pre5|resp                |  19 |           0.695048 |             0.812954 |                    0.117905  |                  0.0465101   |     0.0850541   |               0         |
| combo_pre10_pre5_eda_top16               | window_signal | pre10_pre5|eda                 |  19 |           0.695048 |             0.816475 |                    0.121426  |                 -0.00128375  |    -0.0119482   |               0.157895  |
| win_delta_pre2_0_minus_pre20_pre10_top32 | window        | delta_pre2_0_minus_pre20_pre10 |  19 |           0.695048 |             0.827178 |                    0.13213   |                 -0.00771858  |     0.00869541  |               0.157895  |
| win_delta_pre2_0_minus_pre10_pre5_top32  | window        | delta_pre2_0_minus_pre10_pre5  |  19 |           0.695048 |             0.827778 |                    0.13273   |                  0.0476173   |    -0.0102508   |               0.0526316 |
| combo_pre5_pre2_eda_top16                | window_signal | pre5_pre2|eda                  |  19 |           0.695048 |             0.833608 |                    0.138559  |                 -0.000471315 |    -0.021162    |               0.157895  |
| combo_pre10_0_ecg_top16                  | window_signal | pre10_0|ecg                    |  19 |           0.695048 |             0.835367 |                    0.140319  |                  0.0330568   |     0.0124617   |               0.0526316 |
| combo_pre2_0_eda_top16                   | window_signal | pre2_0|eda                     |  19 |           0.695048 |             0.841136 |                    0.146088  |                  0.0170822   |    -0.0312559   |               0.0526316 |
| combo_pre1_0_resp_top16                  | window_signal | pre1_0|resp                    |  19 |           0.695048 |             0.842543 |                    0.147495  |                  0.0416375   |    -0.0921009   |               0         |
| win_pre30_pre20_top32                    | window        | pre30_pre20                    |  19 |           0.695048 |             0.847639 |                    0.152591  |                  0.0848507   |     0.00366846  |               0.157895  |
| combo_pre10_0_eda_top16                  | window_signal | pre10_0|eda                    |  19 |           0.695048 |             0.850026 |                    0.154978  |                  0.0388821   |    -0.032485    |               0.263158  |
| category_coupling_top32                  | category      | coupling                       |  19 |           0.695048 |             0.853613 |                    0.158564  |                  0.021559    |     0.037308    |               0.105263  |
| combo_pre30_pre20_eda_top16              | window_signal | pre30_pre20|eda                |  19 |           0.695048 |             0.855127 |                    0.160078  |                  0.0417876   |    -0.0272201   |               0.105263  |
| combo_pre1_0_ecg_top16                   | window_signal | pre1_0|ecg                     |  19 |           0.695048 |             0.857474 |                    0.162426  |                  0.0376237   |     0.0854351   |               0.157895  |
| combo_pre20_pre10_eda_top16              | window_signal | pre20_pre10|eda                |  19 |           0.695048 |             0.860566 |                    0.165517  |                  0.0301632   |    -0.00228008  |               0.0526316 |
| combo_pre10_0_hr_top16                   | window_signal | pre10_0|hr                     |  19 |           0.695048 |             0.861221 |                    0.166173  |                 -0.00305844  |    -0.000960528 |               0.0526316 |
| combo_pre5_0_eda_top16                   | window_signal | pre5_0|eda                     |  19 |           0.695048 |             0.86391  |                    0.168862  |                  0.0587075   |    -0.0278391   |               0.105263  |
| category_shape_dynamic_top32             | category      | shape_dynamic                  |  19 |           0.695048 |             0.87161  |                    0.176561  |                  0.0608687   |    -0.0288227   |               0.0526316 |
| combo_pre10_pre5_hr_top16                | window_signal | pre10_pre5|hr                  |  19 |           0.695048 |             0.87169  |                    0.176642  |                  0.031796    |     0.000387933 |               0         |
| signal_resp_top32                        | signal        | resp                           |  19 |           0.695048 |             0.873352 |                    0.178303  |                 -0.0158745   |     0.00119652  |               0.105263  |
| win_pre5_pre2_top32                      | window        | pre5_pre2                      |  19 |           0.695048 |             0.877406 |                    0.182358  |                  0.0507114   |     0.0163337   |               0.263158  |
| signal_ecg_top32                         | signal        | ecg                            |  19 |           0.695048 |             0.887073 |                    0.192025  |                  0.0485953   |     0.0467051   |               0.105263  |
| win_delta_pre2_0_minus_pre5_pre2_top32   | window        | delta_pre2_0_minus_pre5_pre2   |  19 |           0.695048 |             0.903639 |                    0.20859   |                  0.0681397   |     0.015915    |               0.0526316 |
| win_pre5_0_top32                         | window        | pre5_0                         |  19 |           0.695048 |             0.906866 |                    0.211817  |                  0.056213    |    -0.107999    |               0.157895  |
| combo_pre30_pre20_hr_top16               | window_signal | pre30_pre20|hr                 |  19 |           0.695048 |             0.911955 |                    0.216907  |                  0.0740477   |    -0.0428882   |               0         |
| win_pre1_0_top32                         | window        | pre1_0                         |  19 |           0.695048 |             0.91289  |                    0.217842  |                  0.0547376   |    -0.0590415   |               0.0526316 |
| combo_pre2_0_hr_top16                    | window_signal | pre2_0|hr                      |  19 |           0.695048 |             0.916135 |                    0.221086  |                  0.0201652   |     0.0145469   |               0.105263  |
| win_delta_pre2_0_minus_pre30_pre20_top32 | window        | delta_pre2_0_minus_pre30_pre20 |  19 |           0.695048 |             0.917314 |                    0.222265  |                  0.0430581   |    -0.0458618   |               0         |
| win_pre20_pre10_top32                    | window        | pre20_pre10                    |  19 |           0.695048 |             0.920918 |                    0.22587   |                  0.0555364   |    -0.0457321   |               0.157895  |
| combo_pre2_0_resp_top16                  | window_signal | pre2_0|resp                    |  19 |           0.695048 |             0.922417 |                    0.227369  |                  0.0428328   |    -0.0279712   |               0.0526316 |
| combo_pre5_0_hr_top16                    | window_signal | pre5_0|hr                      |  19 |           0.695048 |             0.926855 |                    0.231806  |                  0.0426156   |     0.0235634   |               0.105263  |
| combo_pre20_pre10_hr_top16               | window_signal | pre20_pre10|hr                 |  19 |           0.695048 |             0.929707 |                    0.234659  |                  0.0482938   |    -0.0305746   |               0         |
| signal_hr_top32                          | signal        | hr                             |  19 |           0.695048 |             0.929775 |                    0.234726  |                  0.0614016   |     0.00666278  |               0.105263  |
| combo_pre1_0_eda_top16                   | window_signal | pre1_0|eda                     |  19 |           0.695048 |             0.929856 |                    0.234808  |                  0.0167614   |    -0.0555242   |               0.105263  |
| signal_eda_top32                         | signal        | eda                            |  19 |           0.695048 |             0.932901 |                    0.237852  |                  0.0680212   |    -0.057164    |               0         |
| category_level_dynamic_top32             | category      | level_dynamic                  |  19 |           0.695048 |             0.936472 |                    0.241424  |                  0.070695    |    -0.0907313   |               0.0526316 |
| signal_emg_top32                         | signal        | emg                            |  19 |           0.695048 |             0.937577 |                    0.242528  |                  0.0901637   |    -0.00654634  |               0.0526316 |
| category_rhythm_top32                    | category      | rhythm                         |  19 |           0.695048 |             0.942203 |                    0.247155  |                  0.0971351   |    -0.038134    |               0.157895  |
| combo_pre20_pre10_emg_top16              | window_signal | pre20_pre10|emg                |  19 |           0.695048 |             0.950165 |                    0.255116  |                  0.0533082   |    -0.0123358   |               0         |
| category_causal_past_top32               | category      | causal_past                    |  19 |           0.695048 |             0.957953 |                    0.262904  |                  0.09472     |    -0.0755739   |               0         |
| win_pre10_0_top32                        | window        | pre10_0                        |  19 |           0.695048 |             0.958881 |                    0.263833  |                  0.065513    |    -0.0503399   |               0.0526316 |
| combo_pre5_0_resp_top16                  | window_signal | pre5_0|resp                    |  19 |           0.695048 |             0.962093 |                    0.267044  |                  0.0287504   |    -0.0028548   |               0.0526316 |
| win_pre2_0_top32                         | window        | pre2_0                         |  19 |           0.695048 |             0.965852 |                    0.270803  |                  0.0310736   |    -0.0124256   |               0.210526  |
| combo_pre5_pre2_resp_top16               | window_signal | pre5_pre2|resp                 |  19 |           0.695048 |             0.966833 |                    0.271785  |                  0.0384294   |    -0.0183752   |               0.0526316 |
| combo_pre10_0_emg_top16                  | window_signal | pre10_0|emg                    |  19 |           0.695048 |             0.970071 |                    0.275023  |                  0.040003    |    -0.0461511   |               0.105263  |
| category_quality_top32                   | category      | quality                        |  19 |           0.695048 |             0.986037 |                    0.290988  |                  0.0690945   |     0.0287653   |               0.105263  |

## test bad_top10 + vehicle_ambiguous top feature sets

| feature_set                              | group_type    | group_value                    |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_best_in_top3_rate |
|:-----------------------------------------|:--------------|:-------------------------------|----:|-------------------:|---------------------:|-----------------------------:|-----------------------------:|----------------:|------------------------:|
| combo_pre2_0_ecg_top16                   | window_signal | pre2_0|ecg                     |  15 |           0.744423 |             0.844606 |                     0.100183 |                  0.00903801  |     0.0964113   |               0.2       |
| combo_pre30_pre20_eda_top16              | window_signal | pre30_pre20|eda                |  15 |           0.744423 |             0.863764 |                     0.119341 |                 -0.0129138   |     0.0263046   |               0.133333  |
| combo_pre5_pre2_eda_top16                | window_signal | pre5_pre2|eda                  |  15 |           0.744423 |             0.8704   |                     0.125977 |                 -0.0130047   |     0.0148877   |               0.2       |
| win_pre10_pre5_top32                     | window        | pre10_pre5                     |  15 |           0.744423 |             0.871815 |                     0.127392 |                 -0.0133427   |     0.0270719   |               0.133333  |
| combo_pre10_pre5_eda_top16               | window_signal | pre10_pre5|eda                 |  15 |           0.744423 |             0.879017 |                     0.134594 |                 -0.00621551  |     0.00400354  |               0.2       |
| combo_pre2_0_eda_top16                   | window_signal | pre2_0|eda                     |  15 |           0.744423 |             0.879687 |                     0.135264 |                  0.0130295   |    -0.00922225  |               0.0666667 |
| combo_pre1_0_ecg_top16                   | window_signal | pre1_0|ecg                     |  15 |           0.744423 |             0.880457 |                     0.136034 |                  0.035455    |     0.121185    |               0.2       |
| combo_pre10_0_hr_top16                   | window_signal | pre10_0|hr                     |  15 |           0.744423 |             0.883317 |                     0.138894 |                 -0.0176753   |     0.0222878   |               0.0666667 |
| combo_pre10_pre5_resp_top16              | window_signal | pre10_pre5|resp                |  15 |           0.744423 |             0.884371 |                     0.139949 |                  0.0592328   |     0.0895933   |               0         |
| category_coupling_top32                  | category      | coupling                       |  15 |           0.744423 |             0.890515 |                     0.146092 |                  0.0325913   |     0.0243816   |               0.133333  |
| combo_pre5_0_hr_top16                    | window_signal | pre5_0|hr                      |  15 |           0.744423 |             0.893094 |                     0.148671 |                  0.0331496   |     0.0465816   |               0.133333  |
| combo_pre20_pre10_eda_top16              | window_signal | pre20_pre10|eda                |  15 |           0.744423 |             0.90087  |                     0.156447 |                  0.0136622   |     0.0396007   |               0.0666667 |
| win_delta_pre2_0_minus_pre20_pre10_top32 | window        | delta_pre2_0_minus_pre20_pre10 |  15 |           0.744423 |             0.902135 |                     0.157712 |                 -0.0101491   |     0.0174497   |               0.2       |
| signal_resp_top32                        | signal        | resp                           |  15 |           0.744423 |             0.90473  |                     0.160308 |                 -0.0201666   |    -0.0103361   |               0.133333  |
| combo_pre1_0_resp_top16                  | window_signal | pre1_0|resp                    |  15 |           0.744423 |             0.905075 |                     0.160652 |                  0.039065    |    -0.09928     |               0         |
| win_delta_pre2_0_minus_pre10_pre5_top32  | window        | delta_pre2_0_minus_pre10_pre5  |  15 |           0.744423 |             0.908139 |                     0.163716 |                  0.0634031   |    -0.0138447   |               0.0666667 |
| combo_pre10_0_eda_top16                  | window_signal | pre10_0|eda                    |  15 |           0.744423 |             0.913697 |                     0.169274 |                  0.036843    |    -0.000980844 |               0.333333  |
| combo_pre2_0_hr_top16                    | window_signal | pre2_0|hr                      |  15 |           0.744423 |             0.917131 |                     0.172708 |                  0.000932668 |     0.0479837   |               0.133333  |
| win_pre5_pre2_top32                      | window        | pre5_pre2                      |  15 |           0.744423 |             0.918175 |                     0.173752 |                  0.0217862   |     0.0272898   |               0.333333  |
| combo_pre5_0_eda_top16                   | window_signal | pre5_0|eda                     |  15 |           0.744423 |             0.921424 |                     0.177001 |                  0.0580115   |     0.00685127  |               0.133333  |
| win_pre30_pre20_top32                    | window        | pre30_pre20                    |  15 |           0.744423 |             0.921694 |                     0.177271 |                  0.102391    |    -0.00361184  |               0.133333  |
| category_shape_dynamic_top32             | category      | shape_dynamic                  |  15 |           0.744423 |             0.921941 |                     0.177518 |                  0.0395945   |     0.0020236   |               0.0666667 |
| combo_pre10_0_ecg_top16                  | window_signal | pre10_0|ecg                    |  15 |           0.744423 |             0.924004 |                     0.179581 |                  0.0486188   |     0.00111847  |               0.0666667 |
| combo_pre30_pre20_hr_top16               | window_signal | pre30_pre20|hr                 |  15 |           0.744423 |             0.929014 |                     0.184591 |                  0.0612335   |    -0.0275549   |               0         |
| category_level_dynamic_top32             | category      | level_dynamic                  |  15 |           0.744423 |             0.932767 |                     0.188344 |                  0.053017    |    -0.0966604   |               0.0666667 |
| win_delta_pre2_0_minus_pre30_pre20_top32 | window        | delta_pre2_0_minus_pre30_pre20 |  15 |           0.744423 |             0.936692 |                     0.192269 |                  0.0544813   |    -0.0532697   |               0         |
| combo_pre10_pre5_hr_top16                | window_signal | pre10_pre5|hr                  |  15 |           0.744423 |             0.949696 |                     0.205273 |                  0.0283908   |     0.0171577   |               0         |
| category_rhythm_top32                    | category      | rhythm                         |  15 |           0.744423 |             0.95047  |                     0.206047 |                  0.088884    |    -0.00142377  |               0.2       |
| win_pre5_0_top32                         | window        | pre5_0                         |  15 |           0.744423 |             0.953928 |                     0.209505 |                  0.0684129   |    -0.10813     |               0.2       |
| combo_pre20_pre10_hr_top16               | window_signal | pre20_pre10|hr                 |  15 |           0.744423 |             0.954048 |                     0.209626 |                  0.0599583   |     0.000629355 |               0         |
| combo_pre2_0_resp_top16                  | window_signal | pre2_0|resp                    |  15 |           0.744423 |             0.960036 |                     0.215613 |                  0.0118009   |    -0.0353402   |               0         |
| signal_emg_top32                         | signal        | emg                            |  15 |           0.744423 |             0.96268  |                     0.218258 |                  0.0958165   |    -0.00444711  |               0.0666667 |
| combo_pre1_0_eda_top16                   | window_signal | pre1_0|eda                     |  15 |           0.744423 |             0.974629 |                     0.230206 |                  0.0215826   |    -0.0371891   |               0.133333  |
| signal_eda_top32                         | signal        | eda                            |  15 |           0.744423 |             0.975849 |                     0.231426 |                  0.0672819   |    -0.0519917   |               0         |
| signal_ecg_top32                         | signal        | ecg                            |  15 |           0.744423 |             0.976693 |                     0.23227  |                  0.063551    |     0.0588446   |               0.133333  |
| win_pre1_0_top32                         | window        | pre1_0                         |  15 |           0.744423 |             0.984032 |                     0.239609 |                  0.0756072   |    -0.0384353   |               0.0666667 |
| win_pre2_0_top32                         | window        | pre2_0                         |  15 |           0.744423 |             0.985889 |                     0.241466 |                  0.0179907   |    -0.00761802  |               0.266667  |
| win_delta_pre2_0_minus_pre5_pre2_top32   | window        | delta_pre2_0_minus_pre5_pre2   |  15 |           0.744423 |             0.987898 |                     0.243476 |                  0.0746697   |     0.00586108  |               0.0666667 |
| signal_hr_top32                          | signal        | hr                             |  15 |           0.744423 |             0.98979  |                     0.245367 |                  0.0344026   |     0.0345951   |               0.133333  |
| combo_pre20_pre10_emg_top16              | window_signal | pre20_pre10|emg                |  15 |           0.744423 |             0.991125 |                     0.246702 |                  0.0497838   |     0.0126262   |               0         |
| category_quality_top32                   | category      | quality                        |  15 |           0.744423 |             0.99175  |                     0.247327 |                  0.0753568   |     0.0433739   |               0.133333  |
| category_causal_past_top32               | category      | causal_past                    |  15 |           0.744423 |             0.992346 |                     0.247923 |                  0.09597     |    -0.0744248   |               0         |
| combo_pre5_0_resp_top16                  | window_signal | pre5_0|resp                    |  15 |           0.744423 |             1.00138  |                     0.256953 |                  0.0383424   |    -0.0073884   |               0         |
| win_pre20_pre10_top32                    | window        | pre20_pre10                    |  15 |           0.744423 |             1.00404  |                     0.259616 |                  0.070918    |    -0.0204645   |               0.133333  |
| win_pre10_0_top32                        | window        | pre10_0                        |  15 |           0.744423 |             1.02435  |                     0.27993  |                  0.0781581   |    -0.0308713   |               0.0666667 |
| combo_pre10_0_emg_top16                  | window_signal | pre10_0|emg                    |  15 |           0.744423 |             1.03498  |                     0.290559 |                  0.0505071   |    -0.0184646   |               0.133333  |
| combo_pre5_pre2_resp_top16               | window_signal | pre5_pre2|resp                 |  15 |           0.744423 |             1.05075  |                     0.30633  |                  0.0335287   |    -0.0298472   |               0.0666667 |

## feature set 审计

| feature_set                              | group_type    | group_value                    |   candidate_feature_n |   feature_n |   rank_score_max |   behavior_eta_max |   bad_eta_max |   identity_eta_median |   feature_n_eval |
|:-----------------------------------------|:--------------|:-------------------------------|----------------------:|------------:|-----------------:|-------------------:|--------------:|----------------------:|-----------------:|
| win_pre30_pre20_top32                    | window        | pre30_pre20                    |                   115 |          32 |         0.736402 |          0.046282  |    0.0452955  |             0.135465  |               32 |
| win_pre20_pre10_top32                    | window        | pre20_pre10                    |                   115 |          32 |         0.75804  |          0.046282  |    0.0400816  |             0.134344  |               32 |
| win_pre10_pre5_top32                     | window        | pre10_pre5                     |                   115 |          32 |         0.303826 |          0.046282  |    0.0114558  |             0.121622  |               32 |
| win_pre5_pre2_top32                      | window        | pre5_pre2                      |                   115 |          32 |         0.354405 |          0.046282  |    0.0154792  |             0.12994   |               32 |
| win_pre2_0_top32                         | window        | pre2_0                         |                   142 |          32 |         0.361865 |          0.046282  |    0.0377538  |             0.14241   |               32 |
| win_pre1_0_top32                         | window        | pre1_0                         |                   113 |          32 |         0.355568 |          0.046282  |    0.0278426  |             0.144592  |               32 |
| win_pre5_0_top32                         | window        | pre5_0                         |                   142 |          32 |         0.359774 |          0.046282  |    0.0169339  |             0.158819  |               32 |
| win_pre10_0_top32                        | window        | pre10_0                        |                   142 |          32 |         0.308448 |          0.046282  |    0.0145786  |             0.163018  |               32 |
| win_delta_pre2_0_minus_pre30_pre20_top32 | window        | delta_pre2_0_minus_pre30_pre20 |                    35 |          32 |         0.290552 |          0.0360705 |    0.0360705  |             0.109138  |               32 |
| win_delta_pre2_0_minus_pre20_pre10_top32 | window        | delta_pre2_0_minus_pre20_pre10 |                    35 |          32 |         0.779786 |          0.0356985 |    0.0356985  |             0.10229   |               32 |
| win_delta_pre2_0_minus_pre10_pre5_top32  | window        | delta_pre2_0_minus_pre10_pre5  |                    35 |          32 |         0.391016 |          0.043357  |    0.043357   |             0.0835442 |               32 |
| win_delta_pre2_0_minus_pre5_pre2_top32   | window        | delta_pre2_0_minus_pre5_pre2   |                    35 |          32 |         0.49423  |          0.0423308 |    0.0423308  |             0.065236  |               32 |
| signal_ecg_top32                         | signal        | ecg                            |                   234 |          32 |         0.49423  |          0.0457388 |    0.043357   |             0.324758  |               32 |
| signal_eda_top32                         | signal        | eda                            |                   228 |          32 |         0.779786 |          0.046282  |    0.0452955  |             0.0810619 |               32 |
| signal_emg_top32                         | signal        | emg                            |                   240 |          32 |         0.215491 |          0.0318017 |    0.0178396  |             0.125963  |               32 |
| signal_hr_top32                          | signal        | hr                             |                   198 |          32 |         0.303826 |          0.0361572 |    0.0149863  |             0.117748  |               32 |
| signal_resp_top32                        | signal        | resp                           |                   244 |          32 |         0.348359 |          0.0210765 |    0.0208779  |             0.140923  |               32 |
| category_causal_past_top32               | category      | causal_past                    |                    45 |          32 |         0.299391 |          0.0310685 |    0.0124932  |             0.204767  |               32 |
| category_coupling_top32                  | category      | coupling                       |                    36 |          32 |         0.188353 |          0.0224064 |    0.0116306  |             0.0964368 |               32 |
| category_level_dynamic_top32             | category      | level_dynamic                  |                   280 |          32 |         0.736402 |          0.0457388 |    0.0452955  |             0.135252  |               32 |
| category_quality_top32                   | category      | quality                        |                   125 |          32 |         0.328917 |          0.046282  |    0.0208779  |             0.208234  |               32 |
| category_rhythm_top32                    | category      | rhythm                         |                   118 |          32 |         0.215491 |          0.021527  |    0.0172847  |             0.201702  |               32 |
| category_shape_dynamic_top32             | category      | shape_dynamic                  |                   540 |          32 |         0.779786 |          0.043357  |    0.043357   |             0.11261   |               32 |
| combo_pre20_pre10_eda_top16              | window_signal | pre20_pre10|eda                |                    23 |          16 |         0.75804  |          0.046282  |    0.0400816  |             0.121737  |               16 |
| combo_pre30_pre20_eda_top16              | window_signal | pre30_pre20|eda                |                    23 |          16 |         0.736402 |          0.046282  |    0.0452955  |             0.096939  |               16 |
| combo_pre2_0_eda_top16                   | window_signal | pre2_0|eda                     |                    28 |          16 |         0.361865 |          0.046282  |    0.0073078  |             0.081502  |               16 |
| combo_pre5_0_eda_top16                   | window_signal | pre5_0|eda                     |                    28 |          16 |         0.359774 |          0.046282  |    0.00684005 |             0.0808976 |               16 |
| combo_pre1_0_eda_top16                   | window_signal | pre1_0|eda                     |                    23 |          16 |         0.355568 |          0.046282  |    0.00477868 |             0.0586751 |               16 |
| combo_pre5_pre2_eda_top16                | window_signal | pre5_pre2|eda                  |                    23 |          16 |         0.354405 |          0.046282  |    0.0100188  |             0.0572523 |               16 |
| combo_pre5_pre2_resp_top16               | window_signal | pre5_pre2|resp                 |                    25 |          16 |         0.348359 |          0.0210765 |    0.00942364 |             0.151645  |               16 |
| combo_pre1_0_resp_top16                  | window_signal | pre1_0|resp                    |                    25 |          16 |         0.328917 |          0.0208779 |    0.0208779  |             0.0886056 |               16 |
| combo_pre5_0_resp_top16                  | window_signal | pre5_0|resp                    |                    30 |          16 |         0.318453 |          0.0165146 |    0.0165146  |             0.159147  |               16 |
| combo_pre10_0_eda_top16                  | window_signal | pre10_0|eda                    |                    28 |          16 |         0.308448 |          0.046282  |    0.00770815 |             0.0829561 |               16 |
| combo_pre10_pre5_hr_top16                | window_signal | pre10_pre5|hr                  |                    20 |          16 |         0.303826 |          0.0361572 |    0.00657606 |             0.0912645 |               16 |
| combo_pre5_0_hr_top16                    | window_signal | pre5_0|hr                      |                    23 |          16 |         0.299391 |          0.0310685 |    0.0149863  |             0.140154  |               16 |
| combo_pre2_0_hr_top16                    | window_signal | pre2_0|hr                      |                    23 |          16 |         0.263233 |          0.0266306 |    0.0113298  |             0.132755  |               16 |
| combo_pre20_pre10_emg_top16              | window_signal | pre20_pre10|emg                |                    23 |          16 |         0.215491 |          0.0318017 |    0.0178396  |             0.104342  |               16 |
| combo_pre10_pre5_resp_top16              | window_signal | pre10_pre5|resp                |                    25 |          16 |         0.193731 |          0.0144003 |    0.0110205  |             0.128219  |               16 |
| combo_pre1_0_ecg_top16                   | window_signal | pre1_0|ecg                     |                    22 |          16 |         0.193599 |          0.0278426 |    0.0278426  |             0.293685  |               16 |
| combo_pre10_0_emg_top16                  | window_signal | pre10_0|emg                    |                    32 |          16 |         0.188353 |          0.0272858 |    0.0145786  |             0.154352  |               16 |
| combo_pre2_0_resp_top16                  | window_signal | pre2_0|resp                    |                    30 |          16 |         0.184546 |          0.0152969 |    0.0152969  |             0.115558  |               16 |
| combo_pre20_pre10_hr_top16               | window_signal | pre20_pre10|hr                 |                    20 |          16 |         0.177233 |          0.0227568 |    0.00620624 |             0.100428  |               16 |
| combo_pre10_0_ecg_top16                  | window_signal | pre10_0|ecg                    |                    29 |          16 |         0.175606 |          0.0214811 |    0.00935737 |             0.392228  |               16 |
| combo_pre10_pre5_eda_top16               | window_signal | pre10_pre5|eda                 |                    23 |          16 |         0.175495 |          0.046282  |    0.00849422 |             0.0750216 |               16 |
| combo_pre10_0_hr_top16                   | window_signal | pre10_0|hr                     |                    23 |          16 |         0.175041 |          0.0235512 |    0.00840273 |             0.129982  |               16 |
| combo_pre2_0_ecg_top16                   | window_signal | pre2_0|ecg                     |                    29 |          16 |         0.172903 |          0.0377538 |    0.0377538  |             0.324729  |               16 |
| combo_pre30_pre20_hr_top16               | window_signal | pre30_pre20|hr                 |                    20 |          16 |         0.169901 |          0.0119828 |    0.00595683 |             0.0821519 |               16 |

## 关键判读

- route gate 未通过：没有发现单独时间窗口、信号族或特征类型能够把生理信号转成可部署 top1 收益。
- 如果连窗口/信号族拆分都没有通过，下一步不应继续在同一 v285 特征层上做复杂融合。
- 若仍坚持生理路线，应先回到源信号清洗/事件同步证据，而不是继续换 selector。

## 关键图

- `figures\v287_window_badtop10_top1_delta.png`
- `figures\v287_signal_bad_ambiguous_corr.png`
- `figures\v287_group_type_winners.png`

## guardrail

```json
{
  "pass": true,
  "zip_testzip": true,
  "event_n": 1167,
  "candidate_rows": 46680,
  "screen_feature_n": 1144,
  "feature_set_n": 47,
  "v285_source_guardrail_pass": true,
  "v285_source_uses_post_observation_any": false,
  "route_viable_now": false,
  "deployable_top1_badtop10_pass": false,
  "deployable_top1_bad_ambiguous_pass": false,
  "test_best_top1_diagnostic_pass": false,
  "best_test_badtop10_top1_delta": 0.0941408119703594,
  "best_test_badtop10_corr": 0.08543510556920011,
  "test_used_for_feature_selection": false
}
```