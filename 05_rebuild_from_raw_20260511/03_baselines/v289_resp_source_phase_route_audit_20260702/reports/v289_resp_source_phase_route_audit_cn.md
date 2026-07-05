# v289 RESP source phase route audit

## 本轮目的

- 承接 v288：ECG 源信号未形成可部署 top1 改善。
- 本轮回到 cleaned 200Hz RESP 源信号，重建呼吸周期、相位、幅值、质量和因果同步偏移特征。
- 不使用 RESP_BPM / RESP_Amplitude 这类已知弱派生列；仍然只做 route gate，不训练轨迹融合模型。

## route gate 判定

| check                                             | requirement                                                                    | pass   | evidence                    | deployable   | route_viable_now   |
|:--------------------------------------------------|:-------------------------------------------------------------------------------|:-------|:----------------------------|:-------------|:-------------------|
| deployable_top1_val_chosen_bad_top10              | validation 选出的新生理 top1 在 test bad_top10 上低于 latest                   | False  | 0.1552875779177013          | True         | False              |
| deployable_top1_val_chosen_bad_ambiguous          | validation 选出的新生理 top1 在 test bad_top10_vehicle_ambiguous 上低于 latest | False  | 0.12514613270759584         | True         | False              |
| oracle_top3_val_test_same_direction_bad_ambiguous | 非部署 top3 上限在 val/test 歧义差样本上同向改善                               | False  | val=0.055131, test=0.010645 | False        | False              |
| test_bad_top10_any_feature_corr_gt_005            | test bad_top10 至少一个新特征集的生理距离-真实误差排序相关均值 > 0.05          | False  | 0.04627740497548323         | False        | False              |
| test_best_top1_diagnostic_beats_latest            | 即使 test-best 诊断，新生理 top1 至少有一个特征集低于 latest                   | False  | 0.06250646788822975         | False        | False              |

## validation 选择后的 test 泛化

| event_group                 | method          | deployable   | val_chosen_feature_set          |   val_delta_vs_latest_mean |   test_delta_vs_latest_mean |   test_corr_mean | test_passes_latest   | val_and_test_same_direction_gain   |
|:----------------------------|:----------------|:-------------|:--------------------------------|---------------------------:|----------------------------:|-----------------:|:---------------------|:-----------------------------------|
| all                         | bio_top1        | True         | resp_window_dur2_endm1_top24    |                0.0996328   |                  0.0611607  |     -0.0189469   | False                | False                              |
| all                         | bio_top3_oracle | False        | resp_duration_pre20_pre10_top32 |               -0.0120708   |                 -0.00674316 |      0.000841061 | True                 | True                               |
| all                         | bio_top5_oracle | False        | resp_category_phase_cycle_top48 |               -0.0505264   |                 -0.0299865  |     -0.00953198  | True                 | True                               |
| vehicle_ambiguous           | bio_top1        | True         | resp_window_dur2_endm1_top24    |                0.128105    |                  0.0632815  |     -0.0127123   | False                | False                              |
| vehicle_ambiguous           | bio_top3_oracle | False        | resp_window_dur2_endm1_top24    |                0.000928682 |                 -0.00774441 |     -0.0127123   | True                 | False                              |
| vehicle_ambiguous           | bio_top5_oracle | False        | resp_category_phase_cycle_top48 |               -0.0397135   |                 -0.0291221  |     -2.68716e-05 | True                 | True                               |
| bad_top10                   | bio_top1        | True         | resp_offset_endm0p5_top32       |                0.382644    |                  0.155288   |      0.00582297  | False                | False                              |
| bad_top10                   | bio_top3_oracle | False        | resp_window_dur3_endm1_top24    |                0.0622864   |                  0.00696409 |      0.0362925   | False                | False                              |
| bad_top10                   | bio_top5_oracle | False        | resp_window_dur3_endm1_top24    |               -0.00833071  |                 -0.0287658  |      0.0362925   | True                 | True                               |
| bad_top10_vehicle_ambiguous | bio_top1        | True         | resp_offset_endm0p5_top32       |                0.394395    |                  0.125146   |      0.029698    | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top3_oracle | False        | resp_window_dur3_endm1_top24    |                0.0551311   |                  0.0106453  |      0.0242122   | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top5_oracle | False        | resp_window_dur3_endm1_top24    |               -0.00938591  |                 -0.0292752  |      0.0242122   | True                 | True                               |

## test bad_top10 最佳 top1 诊断

| feature_set                     |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:--------------------------------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| resp_window_dur3_endm1_top24    |  19 |           0.695048 |             0.757555 |                    0.0625065 |                    0.702013 |                  0.00696409  |      0.0362925  |
| resp_offset_end0_top32          |  19 |           0.695048 |             0.772157 |                    0.0771089 |                    0.694774 |                 -0.000274829 |     -0.0162373  |
| resp_window_dur3_end0_top24     |  19 |           0.695048 |             0.773885 |                    0.0788371 |                    0.705369 |                  0.010321    |     -0.017363   |
| resp_category_phase_cycle_top48 |  19 |           0.695048 |             0.792353 |                    0.0973047 |                    0.691926 |                 -0.00312225  |     -0.00917276 |
| resp_window_dur5_end0_top24     |  19 |           0.695048 |             0.798768 |                    0.10372   |                    0.700174 |                  0.00512603  |     -0.0424972  |
| resp_duration_dur3_top32        |  19 |           0.695048 |             0.809089 |                    0.11404   |                    0.687863 |                 -0.00718589  |      0.00873121 |
| resp_duration_pre10_pre5_top32  |  19 |           0.695048 |             0.811594 |                    0.116545  |                    0.726081 |                  0.0310326   |      0.0458194  |
| resp_offset_endm5_top32         |  19 |           0.695048 |             0.811594 |                    0.116545  |                    0.726081 |                  0.0310326   |      0.0458194  |

## test bad_top10 排序相关最高特征集

| feature_set                     |   n |   bio_top1_minus_latest_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_corr_positive_rate |
|:--------------------------------|----:|-----------------------------:|-----------------------------:|----------------:|-------------------------:|
| resp_category_quality_top48     |  19 |                    0.1659    |                   0.015573   |       0.0462774 |                 0.631579 |
| resp_offset_endm5_top32         |  19 |                    0.116545  |                   0.0310326  |       0.0458194 |                 0.631579 |
| resp_duration_pre10_pre5_top32  |  19 |                    0.116545  |                   0.0310326  |       0.0458194 |                 0.631579 |
| resp_window_dur3_endm1_top24    |  19 |                    0.0625065 |                   0.00696409 |       0.0362925 |                 0.368421 |
| resp_category_morph_level_top48 |  19 |                    0.17196   |                   0.0340538  |       0.0310799 |                 0.631579 |
| resp_window_dur5_endm1_top24    |  19 |                    0.138726  |                  -0.00721356 |       0.0251742 |                 0.631579 |
| resp_all_top64                  |  19 |                    0.14403   |                   0.0281578  |       0.0111599 |                 0.473684 |
| resp_offset_endm1_top32         |  19 |                    0.122674  |                   0.024592   |       0.0107341 |                 0.473684 |

## 因果同步偏移组

| feature_set               | group_value   |   n |   bio_top1_minus_latest_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_corr_positive_rate |
|:--------------------------|:--------------|----:|-----------------------------:|-----------------------------:|----------------:|-------------------------:|
| resp_offset_endm5_top32   | endm5         |  15 |                    0.135351  |                    0.0322573 |      0.0389748  |                 0.6      |
| resp_offset_endm0p5_top32 | endm0p5       |  15 |                    0.125146  |                    0.0350809 |      0.029698   |                 0.533333 |
| resp_offset_endm2_top32   | endm2         |  15 |                    0.163244  |                    0.0549199 |      0.0163799  |                 0.533333 |
| resp_offset_endm1_top32   | endm1         |  15 |                    0.0888811 |                    0.021804  |      0.0155541  |                 0.466667 |
| resp_offset_endm10_top32  | endm10        |  15 |                    0.221405  |                    0.0112228 |     -0.00917351 |                 0.466667 |
| resp_offset_end0_top32    | end0          |  15 |                    0.0271259 |                   -0.0267886 |     -0.013329   |                 0.466667 |
| resp_offset_delta_top32   | delta         |  15 |                    0.0790455 |                    0.0294069 |     -0.0140755  |                 0.4      |

## bad_top10 分层

| feature_set                        | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:-----------------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| resp_window_dur3_endm1_top24       | test    |  19 |           0.695048 |             0.757555 |                    0.0625065 |                    0.702013 |                  0.00696409  |      0.0362925  |
| resp_offset_end0_top32             | test    |  19 |           0.695048 |             0.772157 |                    0.0771089 |                    0.694774 |                 -0.000274829 |     -0.0162373  |
| resp_window_dur3_end0_top24        | test    |  19 |           0.695048 |             0.773885 |                    0.0788371 |                    0.705369 |                  0.010321    |     -0.017363   |
| resp_category_phase_cycle_top48    | test    |  19 |           0.695048 |             0.792353 |                    0.0973047 |                    0.691926 |                 -0.00312225  |     -0.00917276 |
| resp_window_dur5_end0_top24        | test    |  19 |           0.695048 |             0.798768 |                    0.10372   |                    0.700174 |                  0.00512603  |     -0.0424972  |
| resp_duration_dur3_top32           | test    |  19 |           0.695048 |             0.809089 |                    0.11404   |                    0.687863 |                 -0.00718589  |      0.00873121 |
| resp_duration_pre10_pre5_top32     | test    |  19 |           0.695048 |             0.811594 |                    0.116545  |                    0.726081 |                  0.0310326   |      0.0458194  |
| resp_offset_endm5_top32            | test    |  19 |           0.695048 |             0.811594 |                    0.116545  |                    0.726081 |                  0.0310326   |      0.0458194  |
| resp_offset_endm1_top32            | test    |  19 |           0.695048 |             0.817722 |                    0.122674  |                    0.71964  |                  0.024592    |      0.0107341  |
| resp_window_dur2_end0_top24        | test    |  19 |           0.695048 |             0.829939 |                    0.13489   |                    0.722468 |                  0.0274201   |     -0.0468101  |
| resp_duration_delta_top32          | test    |  19 |           0.695048 |             0.83097  |                    0.135921  |                    0.742655 |                  0.0476071   |      0.00184124 |
| resp_offset_delta_top32            | test    |  19 |           0.695048 |             0.83097  |                    0.135921  |                    0.742655 |                  0.0476071   |      0.00184124 |
| resp_window_dur5_endm1_top24       | test    |  19 |           0.695048 |             0.833774 |                    0.138726  |                    0.687835 |                 -0.00721356  |      0.0251742  |
| resp_all_top64                     | test    |  19 |           0.695048 |             0.839078 |                    0.14403   |                    0.723206 |                  0.0281578   |      0.0111599  |
| resp_low_identity_top48            | test    |  19 |           0.695048 |             0.843309 |                    0.148261  |                    0.730066 |                  0.0350173   |     -0.00111999 |
| resp_duration_dur2_top32           | test    |  19 |           0.695048 |             0.844962 |                    0.149914  |                    0.750939 |                  0.055891    |     -0.0170296  |
| resp_window_dur2_endm1_top24       | test    |  19 |           0.695048 |             0.848741 |                    0.153693  |                    0.682836 |                 -0.0122125   |     -0.0121929  |
| resp_offset_endm0p5_top32          | test    |  19 |           0.695048 |             0.850336 |                    0.155288  |                    0.725422 |                  0.0303731   |      0.00582297 |
| resp_duration_dur5_top32           | test    |  19 |           0.695048 |             0.85927  |                    0.164222  |                    0.7206   |                  0.025552    |      0.00136359 |
| resp_duration_dur10_top32          | test    |  19 |           0.695048 |             0.860807 |                    0.165758  |                    0.730631 |                  0.0355822   |     -0.0291926  |
| resp_category_quality_top48        | test    |  19 |           0.695048 |             0.860948 |                    0.1659    |                    0.710621 |                  0.015573    |      0.0462774  |
| resp_category_morph_level_top48    | test    |  19 |           0.695048 |             0.867009 |                    0.17196   |                    0.729102 |                  0.0340538   |      0.0310799  |
| resp_category_temporal_delta_top48 | test    |  19 |           0.695048 |             0.898399 |                    0.203351  |                    0.763816 |                  0.0687677   |     -0.0222054  |
| resp_offset_endm2_top32            | test    |  19 |           0.695048 |             0.90052  |                    0.205471  |                    0.742691 |                  0.0476428   |     -0.00371743 |
| resp_duration_pre20_pre10_top32    | test    |  19 |           0.695048 |             0.903155 |                    0.208107  |                    0.700119 |                  0.00507043  |     -0.0134661  |
| resp_offset_endm10_top32           | test    |  19 |           0.695048 |             0.903155 |                    0.208107  |                    0.700119 |                  0.00507043  |     -0.0134661  |
| resp_category_morph_dynamic_top48  | test    |  19 |           0.695048 |             0.91333  |                    0.218281  |                    0.767756 |                  0.0727072   |     -0.0281399  |
| resp_offset_endm0p5_top32          | val     |  31 |           1.07279  |             1.45543  |                    0.382644  |                    1.22507  |                  0.152281    |      0.0315022  |
| resp_window_dur3_endm1_top24       | val     |  31 |           1.07279  |             1.45586  |                    0.383071  |                    1.13507  |                  0.0622864   |      0.0366485  |
| resp_low_identity_top48            | val     |  31 |           1.07279  |             1.47629  |                    0.403505  |                    1.23409  |                  0.161301    |      0.0237532  |
| resp_window_dur2_endm1_top24       | val     |  31 |           1.07279  |             1.49153  |                    0.418745  |                    1.20041  |                  0.127622    |      0.0455553  |
| resp_offset_endm1_top32            | val     |  31 |           1.07279  |             1.52108  |                    0.448296  |                    1.21014  |                  0.137353    |      0.0156859  |
| resp_duration_delta_top32          | val     |  31 |           1.07279  |             1.52258  |                    0.449789  |                    1.21881  |                  0.146026    |      0.0243243  |
| resp_offset_delta_top32            | val     |  31 |           1.07279  |             1.52258  |                    0.449789  |                    1.21881  |                  0.146026    |      0.0243243  |
| resp_duration_dur2_top32           | val     |  31 |           1.07279  |             1.56707  |                    0.494283  |                    1.2146   |                  0.141814    |      0.0382667  |
| resp_duration_dur5_top32           | val     |  31 |           1.07279  |             1.57707  |                    0.504284  |                    1.24266  |                  0.169877    |      0.0329478  |
| resp_window_dur5_endm1_top24       | val     |  31 |           1.07279  |             1.57876  |                    0.505968  |                    1.31959  |                  0.246801    |      0.00373954 |
| resp_category_temporal_delta_top48 | val     |  31 |           1.07279  |             1.58929  |                    0.516501  |                    1.25341  |                  0.180623    |      0.0283799  |
| resp_all_top64                     | val     |  31 |           1.07279  |             1.62123  |                    0.548437  |                    1.26067  |                  0.187887    |      0.0196402  |
| resp_category_morph_level_top48    | val     |  31 |           1.07279  |             1.62183  |                    0.549043  |                    1.23859  |                  0.165804    |      0.0721956  |

## bad_top10_vehicle_ambiguous 分层

| feature_set                        | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:-----------------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| resp_offset_end0_top32             | test    |  15 |           0.744423 |             0.771549 |                    0.0271259 |                    0.717634 |                 -0.0267886   |     -0.013329   |
| resp_window_dur3_endm1_top24       | test    |  15 |           0.744423 |             0.806652 |                    0.0622295 |                    0.755068 |                  0.0106453   |      0.0242122  |
| resp_window_dur5_endm1_top24       | test    |  15 |           0.744423 |             0.813053 |                    0.0686299 |                    0.722097 |                 -0.0223264   |      0.0372606  |
| resp_duration_delta_top32          | test    |  15 |           0.744423 |             0.823468 |                    0.0790455 |                    0.77383  |                  0.0294069   |     -0.0140755  |
| resp_offset_delta_top32            | test    |  15 |           0.744423 |             0.823468 |                    0.0790455 |                    0.77383  |                  0.0294069   |     -0.0140755  |
| resp_window_dur3_end0_top24        | test    |  15 |           0.744423 |             0.830141 |                    0.0857182 |                    0.758178 |                  0.0137551   |     -0.016092   |
| resp_offset_endm1_top32            | test    |  15 |           0.744423 |             0.833304 |                    0.0888811 |                    0.766227 |                  0.021804    |      0.0155541  |
| resp_window_dur5_end0_top24        | test    |  15 |           0.744423 |             0.841711 |                    0.0972879 |                    0.749711 |                  0.00528774  |     -0.0228142  |
| resp_category_phase_cycle_top48    | test    |  15 |           0.744423 |             0.852748 |                    0.108325  |                    0.744483 |                  5.97884e-05 |      0.0247608  |
| resp_window_dur2_end0_top24        | test    |  15 |           0.744423 |             0.863034 |                    0.118611  |                    0.773919 |                  0.029496    |     -0.0417751  |
| resp_offset_endm0p5_top32          | test    |  15 |           0.744423 |             0.869569 |                    0.125146  |                    0.779504 |                  0.0350809   |      0.029698   |
| resp_duration_dur2_top32           | test    |  15 |           0.744423 |             0.879729 |                    0.135306  |                    0.764707 |                  0.0202837   |     -0.0150909  |
| resp_duration_pre10_pre5_top32     | test    |  15 |           0.744423 |             0.879774 |                    0.135351  |                    0.77668  |                  0.0322573   |      0.0389748  |
| resp_offset_endm5_top32            | test    |  15 |           0.744423 |             0.879774 |                    0.135351  |                    0.77668  |                  0.0322573   |      0.0389748  |
| resp_window_dur2_endm1_top24       | test    |  15 |           0.744423 |             0.886805 |                    0.142382  |                    0.734244 |                 -0.0101787   |     -0.0301983  |
| resp_duration_dur3_top32           | test    |  15 |           0.744423 |             0.886973 |                    0.14255   |                    0.742482 |                 -0.00194072  |      0.041718   |
| resp_all_top64                     | test    |  15 |           0.744423 |             0.892037 |                    0.147614  |                    0.776608 |                  0.0321846   |      0.0276884  |
| resp_low_identity_top48            | test    |  15 |           0.744423 |             0.897396 |                    0.152973  |                    0.785296 |                  0.0408733   |      0.00703507 |
| resp_duration_dur5_top32           | test    |  15 |           0.744423 |             0.905947 |                    0.161524  |                    0.77673  |                  0.032307    |      0.0407729  |
| resp_category_temporal_delta_top48 | test    |  15 |           0.744423 |             0.907313 |                    0.16289   |                    0.787433 |                  0.0430103   |     -0.0444934  |
| resp_offset_endm2_top32            | test    |  15 |           0.744423 |             0.907667 |                    0.163244  |                    0.799343 |                  0.0549199   |      0.0163799  |
| resp_category_morph_dynamic_top48  | test    |  15 |           0.744423 |             0.914493 |                    0.17007   |                    0.794033 |                  0.0496103   |     -0.0220955  |
| resp_category_quality_top48        | test    |  15 |           0.744423 |             0.91785  |                    0.173427  |                    0.730795 |                 -0.0136275   |      0.0351104  |
| resp_duration_dur10_top32          | test    |  15 |           0.744423 |             0.950051 |                    0.205628  |                    0.790531 |                  0.0461076   |     -0.0568805  |
| resp_category_morph_level_top48    | test    |  15 |           0.744423 |             0.963276 |                    0.218853  |                    0.792688 |                  0.0482654   |      0.0138943  |
| resp_duration_pre20_pre10_top32    | test    |  15 |           0.744423 |             0.965828 |                    0.221405  |                    0.755646 |                  0.0112228   |     -0.00917351 |
| resp_offset_endm10_top32           | test    |  15 |           0.744423 |             0.965828 |                    0.221405  |                    0.755646 |                  0.0112228   |     -0.00917351 |
| resp_offset_endm0p5_top32          | val     |  27 |           1.02949  |             1.42389  |                    0.394395  |                    1.18845  |                  0.158954    |      0.0375123  |
| resp_window_dur3_endm1_top24       | val     |  27 |           1.02949  |             1.44387  |                    0.414376  |                    1.08462  |                  0.0551311   |      0.0507065  |
| resp_window_dur2_endm1_top24       | val     |  27 |           1.02949  |             1.4754   |                    0.445906  |                    1.17044  |                  0.14095     |      0.0535929  |
| resp_low_identity_top48            | val     |  27 |           1.02949  |             1.4815   |                    0.452008  |                    1.21408  |                  0.184591    |      0.0393188  |
| resp_duration_delta_top32          | val     |  27 |           1.02949  |             1.51641  |                    0.486911  |                    1.18799  |                  0.158499    |      0.0428484  |
| resp_offset_delta_top32            | val     |  27 |           1.02949  |             1.51641  |                    0.486911  |                    1.18799  |                  0.158499    |      0.0428484  |
| resp_offset_endm1_top32            | val     |  27 |           1.02949  |             1.52167  |                    0.492177  |                    1.17036  |                  0.140868    |      0.0223791  |
| resp_window_dur5_endm1_top24       | val     |  27 |           1.02949  |             1.57425  |                    0.544761  |                    1.28992  |                  0.260424    |      0.0151602  |
| resp_duration_dur5_top32           | val     |  27 |           1.02949  |             1.58677  |                    0.557273  |                    1.22321  |                  0.193712    |      0.0433393  |
| resp_category_temporal_delta_top48 | val     |  27 |           1.02949  |             1.593    |                    0.563506  |                    1.22807  |                  0.19858     |      0.0429715  |
| resp_duration_dur2_top32           | val     |  27 |           1.02949  |             1.5965   |                    0.56701   |                    1.19182  |                  0.162323    |      0.0339911  |
| resp_all_top64                     | val     |  27 |           1.02949  |             1.63337  |                    0.603873  |                    1.23524  |                  0.205744    |      0.0361205  |
| resp_duration_dur3_top32           | val     |  27 |           1.02949  |             1.64469  |                    0.6152    |                    1.28507  |                  0.255577    |      0.0555454  |

## feature set 审计

| feature_set                        | group_type   | group_value    |   candidate_feature_n |   feature_n |   behavior_eta_max |   bad_eta_max |   identity_eta_median |
|:-----------------------------------|:-------------|:---------------|----------------------:|------------:|-------------------:|--------------:|----------------------:|
| resp_all_top64                     | all          | all            |                   477 |          64 |          0.0238073 |    0.0196948  |             0.096135  |
| resp_low_identity_top48            | identity     | low_identity   |                   251 |          48 |          0.0216395 |    0.0196948  |             0.0714951 |
| resp_category_phase_cycle_top48    | category     | phase_cycle    |                   167 |          48 |          0.018951  |    0.0175158  |             0.103648  |
| resp_category_morph_dynamic_top48  | category     | morph_dynamic  |                   126 |          48 |          0.0216395 |    0.0188653  |             0.157199  |
| resp_category_morph_level_top48    | category     | morph_level    |                    56 |          48 |          0.0219894 |    0.0120241  |             0.137242  |
| resp_category_quality_top48        | category     | quality        |                    59 |          48 |          0.0238073 |    0.0120902  |             0         |
| resp_category_temporal_delta_top48 | category     | temporal_delta |                    69 |          48 |          0.0196948 |    0.0196948  |             0.0730824 |
| resp_offset_end0_top32             | offset       | end0           |                   117 |          32 |          0.0219894 |    0.0188653  |             0.11497   |
| resp_offset_endm0p5_top32          | offset       | endm0p5        |                    83 |          32 |          0.0216395 |    0.0120902  |             0.109566  |
| resp_offset_endm1_top32            | offset       | endm1          |                    83 |          32 |          0.0203713 |    0.0128464  |             0.0945293 |
| resp_offset_endm2_top32            | offset       | endm2          |                    58 |          32 |          0.0173851 |    0.00845124 |             0.106949  |
| resp_offset_endm5_top32            | offset       | endm5          |                    29 |          29 |          0.0175158 |    0.0175158  |             0.103694  |
| resp_offset_endm10_top32           | offset       | endm10         |                    34 |          32 |          0.0168325 |    0.0168325  |             0.179953  |
| resp_offset_delta_top32            | offset       | delta          |                    69 |          32 |          0.0196948 |    0.0196948  |             0.0730824 |
| resp_duration_dur2_top32           | duration     | dur2           |                    75 |          32 |          0.0219894 |    0.0120902  |             0.0942334 |
| resp_duration_dur3_top32           | duration     | dur3           |                   116 |          32 |          0.0212292 |    0.0118571  |             0.102793  |
| resp_duration_dur5_top32           | duration     | dur5           |                   116 |          32 |          0.0188653 |    0.0188653  |             0.124204  |
| resp_duration_dur10_top32          | duration     | dur10          |                    34 |          32 |          0.0155992 |    0.00723528 |             0.149529  |
| resp_duration_pre10_pre5_top32     | duration     | pre10_pre5     |                    29 |          29 |          0.0175158 |    0.0175158  |             0.103694  |
| resp_duration_pre20_pre10_top32    | duration     | pre20_pre10    |                    34 |          32 |          0.0168325 |    0.0168325  |             0.179953  |
| resp_duration_delta_top32          | duration     | delta          |                    69 |          32 |          0.0196948 |    0.0196948  |             0.0730824 |
| resp_window_dur2_end0_top24        | window       | dur2_end0      |                    25 |          24 |          0.0219894 |    0.00991357 |             0.0944708 |
| resp_window_dur3_end0_top24        | window       | dur3_end0      |                    29 |          24 |          0.0170373 |    0.00948876 |             0.102038  |
| resp_window_dur5_end0_top24        | window       | dur5_end0      |                    29 |          24 |          0.0188653 |    0.0188653  |             0.139402  |
| resp_window_dur2_endm1_top24       | window       | dur2_endm1     |                    25 |          24 |          0.0203713 |    0.00869575 |             0.0942305 |
| resp_window_dur3_endm1_top24       | window       | dur3_endm1     |                    29 |          24 |          0.0152823 |    0.0118571  |             0.100943  |
| resp_window_dur5_endm1_top24       | window       | dur5_endm1     |                    29 |          24 |          0.0163874 |    0.0128464  |             0.117024  |

## train-only RESP feature screen 摘要

| feature_category   | offset_group   | duration_group   |   feature_n |   behavior_eta_max |   bad_eta_max |   identity_eta_median |   behavior_identity_score_max |
|:-------------------|:---------------|:-----------------|------------:|-------------------:|--------------:|----------------------:|------------------------------:|
| phase_cycle        | endm2          | dur3             |          17 |         0.308804   |    0.308804   |             0.0934979 |                     0.633903  |
| phase_cycle        | endm1          | dur5             |          17 |         0.0320806  |    0.0320806  |             0.146642  |                     0.615265  |
| temporal_delta     | delta          | delta            |          95 |         0.0545089  |    0.0254496  |             0.0757508 |                     0.334947  |
| phase_cycle        | end0           | dur2             |          17 |         0.0292782  |    0.00838853 |             0.0891965 |                     0.261528  |
| morph_dynamic      | end0           | dur5             |           9 |         0.0188653  |    0.0188653  |             0.200211  |                     0.245018  |
| phase_cycle        | endm0p5        | dur3             |          17 |         0.0859067  |    0.0714286  |             0.109566  |                     0.240941  |
| morph_dynamic      | endm0p5        | dur2             |           9 |         0.0216395  |    0.00981646 |             0.115016  |                     0.22633   |
| phase_cycle        | endm2          | dur5             |          17 |         0.0371717  |    0.0279956  |             0.173836  |                     0.220961  |
| morph_level        | endm2          | dur3             |           4 |         0.0173851  |    0.00489912 |             0.12738   |                     0.21305   |
| phase_cycle        | end0           | dur10            |          17 |         0.0155992  |    0.00691663 |             0.226634  |                     0.212569  |
| morph_dynamic      | endm1          | dur2             |           9 |         0.0203713  |    0.00752106 |             0.126908  |                     0.208844  |
| phase_cycle        | end0           | dur5             |          17 |         0.0148623  |    0.00831978 |             0.188893  |                     0.193813  |
| phase_cycle        | endm5          | pre10_pre5       |          17 |         0.0344981  |    0.0344981  |             0.169135  |                     0.188873  |
| morph_level        | end0           | dur2             |           4 |         0.0219894  |    0.00799296 |             0.116767  |                     0.182405  |
| phase_cycle        | endm10         | pre20_pre10      |          17 |         0.0168325  |    0.0168325  |             0.253801  |                     0.180591  |
| morph_dynamic      | endm0p5        | dur3             |           9 |         0.0212292  |    0.00803745 |             0.155723  |                     0.178322  |
| phase_cycle        | endm1          | dur3             |          17 |         0.0799271  |    0.0118571  |             0.0832537 |                     0.1652    |
| morph_dynamic      | endm5          | pre10_pre5       |           9 |         0.014919   |    0.0127004  |             0.172541  |                     0.163474  |
| morph_dynamic      | endm0p5        | dur5             |           9 |         0.0173412  |    0.0082922  |             0.193892  |                     0.157452  |
| phase_cycle        | end0           | dur3             |          17 |         0.0616182  |    0.024244   |             0.122677  |                     0.155967  |
| morph_dynamic      | endm2          | dur5             |           9 |         0.00985782 |    0.00629016 |             0.189276  |                     0.154974  |
| morph_dynamic      | end0           | dur3             |           9 |         0.0170373  |    0.00706549 |             0.156134  |                     0.149601  |
| phase_cycle        | endm0p5        | dur5             |          17 |         0.0190862  |    0.00983124 |             0.157875  |                     0.145237  |
| phase_cycle        | endm1          | dur2             |          17 |         0.0161906  |    0.0106102  |             0.0764084 |                     0.139415  |
| phase_cycle        | endm0p5        | dur2             |          17 |         0.0156304  |    0.0110352  |             0.0891675 |                     0.136978  |
| morph_dynamic      | endm1          | dur5             |           9 |         0.0163874  |    0.00483811 |             0.198207  |                     0.129623  |
| morph_dynamic      | endm1          | dur3             |           9 |         0.0152823  |    0.00900807 |             0.172596  |                     0.129623  |
| morph_dynamic      | end0           | dur2             |           9 |         0.0125788  |    0.00991357 |             0.148952  |                     0.128594  |
| morph_level        | endm0p5        | dur3             |           4 |         0.0137349  |    0.00735677 |             0.118952  |                     0.120962  |
| morph_level        | endm0p5        | dur2             |           4 |         0.0120241  |    0.0120241  |             0.0928781 |                     0.118175  |
| quality            | end0           | dur2             |           4 |         0.0117578  |    0.00504664 |             0         |                     0.110781  |
| quality            | endm5          | pre10_pre5       |           4 |         0.00615516 |    0.00300431 |             0         |                     0.105423  |
| morph_level        | endm1          | dur2             |           4 |         0.0121327  |    0.0072151  |             0.100467  |                     0.101528  |
| morph_level        | endm1          | dur3             |           4 |         0.0110042  |    0.00920616 |             0.128826  |                     0.0991872 |
| quality            | endm0p5        | dur2             |           4 |         0.0120902  |    0.0120902  |             0         |                     0.0988777 |
| morph_level        | endm0p5        | dur5             |           4 |         0.0149156  |    0.00675032 |             0.164209  |                     0.0968081 |
| quality            | end0           | dur3             |           4 |         0.0102     |    0.00527143 |             0         |                     0.0937882 |
| morph_dynamic      | endm2          | dur3             |           9 |         0.008404   |    0.00773632 |             0.160503  |                     0.090822  |
| morph_level        | end0           | dur3             |           4 |         0.00948876 |    0.00948876 |             0.108504  |                     0.0902581 |
| quality            | end0           | dur10            |           4 |         0.00723528 |    0.00723528 |             0         |                     0.0882589 |
| morph_dynamic      | endm10         | pre20_pre10      |           9 |         0.00943425 |    0.00943425 |             0.22239   |                     0.0859833 |
| morph_level        | endm1          | dur5             |           4 |         0.0130108  |    0.00716202 |             0.168472  |                     0.0856612 |
| morph_dynamic      | end0           | dur10            |           9 |         0.0110627  |    0.00288004 |             0.227698  |                     0.0834606 |
| morph_level        | end0           | dur5             |           4 |         0.0145228  |    0.00707581 |             0.172142  |                     0.0816821 |
| quality            | endm0p5        | dur5             |           4 |         0.00600021 |    0.00600021 |             0         |                     0.0754079 |
| quality            | endm2          | dur3             |           4 |         0.00471189 |    0.00180581 |             0         |                     0.0703433 |
| quality            | endm1          | dur2             |           4 |         0.00864011 |    0.00701673 |             0         |                     0.0688259 |
| morph_level        | endm10         | pre20_pre10      |           4 |         0.0127382  |    0.00760064 |             0.193624  |                     0.0680064 |
| quality            | endm1          | dur3             |           4 |         0.00667039 |    0.00667039 |             0         |                     0.0671107 |
| quality            | endm0p5        | dur3             |           4 |         0.00667039 |    0.00667039 |             0         |                     0.066985  |

## RESP 质量摘要

| subject   | recording                            | split   |   event_n |   ok_rate |   bio289_baseline_valid_ratio_median |   bio289_context_period_s_median |   bio289_context_bpm_median |   bio289_w_dur5_end0_resp_zero_up_n_median |   bio289_w_dur5_end0_resp_period_plausible_rate_median |
|:----------|:-------------------------------------|:--------|----------:|----------:|-------------------------------------:|---------------------------------:|----------------------------:|-------------------------------------------:|-------------------------------------------------------:|
| cwh       | Entity_Recording_2025_09_26_19_35_47 | test    |        10 |         1 |                                    1 |                          2.74617 |                     21.8644 |                                        3   |                                               0.333333 |
| cwh       | Entity_Recording_2025_09_26_19_45_40 | test    |        11 |         1 |                                    1 |                          2.89682 |                     20.7124 |                                        3   |                                               0.5      |
| cwh       | Entity_Recording_2025_09_26_19_56_16 | test    |        11 |         1 |                                    1 |                          2.82522 |                     21.2373 |                                        2   |                                               0.5      |
| cwh       | Entity_Recording_2025_09_26_20_06_19 | test    |        14 |         1 |                                    1 |                          2.56569 |                     23.7182 |                                        3   |                                               0.5      |
| lx        | Entity_Recording_2025_09_26_08_58_43 | test    |         2 |         1 |                                    1 |                          3.10049 |                     19.852  |                                        0.5 |                                             nan        |
| lx        | Entity_Recording_2025_09_26_09_17_22 | test    |        11 |         1 |                                    1 |                          4.06833 |                     14.7481 |                                        1   |                                               0.5      |
| rjy       | Entity_Recording_2025_09_28_19_33_26 | test    |        17 |         0 |                                  nan |                        nan       |                    nan      |                                      nan   |                                             nan        |
| rjy       | Entity_Recording_2025_09_28_19_44_42 | test    |         2 |         0 |                                  nan |                        nan       |                    nan      |                                      nan   |                                             nan        |
| rjy       | Entity_Recording_2025_09_28_19_51_44 | test    |        19 |         1 |                                    1 |                          2.95112 |                     20.3312 |                                        2   |                                               1        |
| rjy       | Entity_Recording_2025_09_28_20_02_20 | test    |        25 |         1 |                                    1 |                          3.50868 |                     17.1004 |                                        2   |                                               0        |
| rjy       | Entity_Recording_2025_09_28_20_15_42 | test    |        19 |         1 |                                    1 |                          3.07404 |                     19.5183 |                                        2   |                                               1        |
| tyy       | Entity_Recording_2025_09_28_14_23_43 | test    |        24 |         1 |                                    1 |                          2.60925 |                     22.9963 |                                        2   |                                               1        |
| tyy       | Entity_Recording_2025_09_28_14_40_01 | test    |         7 |         1 |                                    1 |                          2.81003 |                     21.3521 |                                        2   |                                               0.5      |
| tyy       | Entity_Recording_2025_09_28_14_57_17 | test    |        12 |         1 |                                    1 |                          2.85836 |                     20.9911 |                                        2   |                                               1        |
| byx       | Entity_Recording_2025_09_28_17_05_51 | train   |        19 |         1 |                                    1 |                          3.43436 |                     17.4705 |                                        1   |                                               1        |
| byx       | Entity_Recording_2025_09_28_17_15_52 | train   |        18 |         1 |                                    1 |                          3.24291 |                     18.5019 |                                        1   |                                               1        |
| byx       | Entity_Recording_2025_09_28_17_25_18 | train   |        23 |         1 |                                    1 |                          3.41133 |                     17.5884 |                                        1   |                                               1        |
| byx       | Entity_Recording_2025_09_28_17_35_43 | train   |        25 |         1 |                                    1 |                          3.42337 |                     17.5266 |                                        2   |                                               1        |
| byx       | Entity_Recording_2025_09_28_17_46_00 | train   |        17 |         1 |                                    1 |                          3.31185 |                     18.1167 |                                        1   |                                               1        |
| gf        | Entity_Recording_2025_09_26_10_03_00 | train   |         1 |         1 |                                    1 |                          2.55736 |                     23.4617 |                                        1   |                                             nan        |
| gf        | Entity_Recording_2025_09_26_10_18_49 | train   |         9 |         1 |                                    1 |                          2.37971 |                     25.2131 |                                        3   |                                               0        |
| gf        | Entity_Recording_2025_09_26_10_30_12 | train   |         9 |         1 |                                    1 |                          2.54569 |                     23.5692 |                                        2   |                                               0.5      |
| gf        | Entity_Recording_2025_09_26_10_40_59 | train   |         9 |         1 |                                    1 |                          2.46102 |                     24.3802 |                                        2   |                                               0.5      |
| gf        | Entity_Recording_2025_09_26_10_52_57 | train   |         8 |         1 |                                    1 |                          2.47885 |                     24.2508 |                                        2   |                                               0        |
| hzh       | Entity_Recording_2025_09_26_20_50_27 | train   |        27 |         1 |                                    1 |                          2.66413 |                     22.5215 |                                        2   |                                               1        |
| hzh       | Entity_Recording_2025_09_26_21_03_19 | train   |        24 |         1 |                                    1 |                          3.18439 |                     18.8424 |                                        1.5 |                                               1        |
| hzh       | Entity_Recording_2025_09_26_21_17_02 | train   |        10 |         1 |                                    1 |                          3.18918 |                     18.8149 |                                        1   |                                               1        |
| hzh       | Entity_Recording_2025_09_27_19_22_27 | train   |        17 |         1 |                                    1 |                          2.97709 |                     20.1539 |                                        2   |                                               1        |
| hzh       | Entity_Recording_2025_09_27_19_33_25 | train   |        19 |         1 |                                    1 |                          3.12823 |                     19.1802 |                                        2   |                                               1        |
| hzh       | Entity_Recording_2025_09_27_19_44_05 | train   |        21 |         1 |                                    1 |                          3.14765 |                     19.0619 |                                        2   |                                               0.5      |
| jy        | Entity_Recording_2025_09_26_17_17_11 | train   |         8 |         1 |                                    1 |                          3.21626 |                     18.6574 |                                        1   |                                               1        |
| jy        | Entity_Recording_2025_09_26_17_29_44 | train   |        11 |         1 |                                    1 |                          2.88571 |                     20.7921 |                                        1   |                                               1        |
| jy        | Entity_Recording_2025_09_26_17_40_51 | train   |        10 |         1 |                                    1 |                          2.84644 |                     21.0838 |                                        2   |                                               0.75     |
| jy        | Entity_Recording_2025_09_26_17_51_46 | train   |         3 |         1 |                                    1 |                          2.54477 |                     23.5778 |                                        2   |                                               1        |
| jy        | Entity_Recording_2025_09_26_18_01_40 | train   |        10 |         1 |                                    1 |                          2.92233 |                     20.5342 |                                        2   |                                               1        |
| xst       | Entity_Recording_2025_09_26_11_34_18 | train   |         6 |         1 |                                    1 |                          3.02689 |                     19.8234 |                                        1.5 |                                               1        |
| yyl       | Entity_Recording_2025_09_28_09_14_23 | train   |        21 |         1 |                                    1 |                          3.02783 |                     19.8162 |                                        1   |                                               1        |
| yyl       | Entity_Recording_2025_09_28_09_29_01 | train   |        28 |         1 |                                    1 |                          3.02441 |                     19.8386 |                                        2   |                                               1        |
| yyl       | Entity_Recording_2025_09_28_09_39_01 | train   |        20 |         1 |                                    1 |                          3.07137 |                     19.5353 |                                        2   |                                               1        |
| yyl       | Entity_Recording_2025_09_28_09_49_11 | train   |        18 |         1 |                                    1 |                          3.10689 |                     19.312  |                                        2   |                                               1        |
| yzy       | Entity_Recording_2025_09_27_14_13_03 | train   |        23 |         1 |                                    1 |                          2.9428  |                     20.3887 |                                        2   |                                               1        |
| yzy       | Entity_Recording_2025_09_27_14_26_04 | train   |        21 |         1 |                                    1 |                          3.07518 |                     19.511  |                                        2   |                                               1        |
| yzy       | Entity_Recording_2025_09_27_14_37_08 | train   |        21 |         1 |                                    1 |                          3.74505 |                     16.0211 |                                        2   |                                               0.75     |
| yzy       | Entity_Recording_2025_09_27_15_04_26 | train   |         1 |         1 |                                    1 |                          4.21236 |                     14.2438 |                                        2   |                                               1        |
| yzy       | Entity_Recording_2025_09_27_15_07_57 | train   |        13 |         1 |                                    1 |                          3.18548 |                     18.8354 |                                        1   |                                               1        |
| zt        | Entity_Recording_2025_09_28_11_20_08 | train   |        15 |         1 |                                    1 |                          2.82639 |                     21.2285 |                                        2   |                                               1        |
| zx        | Entity_Recording_2025_09_27_16_32_00 | train   |        24 |         0 |                                  nan |                        nan       |                    nan      |                                      nan   |                                             nan        |
| zx        | Entity_Recording_2025_09_27_16_46_13 | train   |        26 |         0 |                                  nan |                        nan       |                    nan      |                                      nan   |                                             nan        |
| zx        | Entity_Recording_2025_09_27_17_14_07 | train   |        21 |         0 |                                  nan |                        nan       |                    nan      |                                      nan   |                                             nan        |
| zx        | Entity_Recording_2025_09_27_17_25_16 | train   |         3 |         0 |                                  nan |                        nan       |                    nan      |                                      nan   |                                             nan        |
| zx        | Entity_Recording_2025_09_27_17_29_08 | train   |         1 |         0 |                                  nan |                        nan       |                    nan      |                                      nan   |                                             nan        |
| zx        | Entity_Recording_2025_09_27_17_45_11 | train   |        27 |         1 |                                    1 |                          3.31256 |                     18.1129 |                                        2   |                                               1        |
| zx        | Entity_Recording_2025_09_27_17_56_42 | train   |         3 |         1 |                                    1 |                          3.57777 |                     16.7702 |                                        1   |                                             nan        |
| zx        | Entity_Recording_2025_09_27_18_07_01 | train   |        23 |         1 |                                    1 |                          3.61488 |                     16.5981 |                                        1   |                                               1        |
| zx        | Entity_Recording_2025_09_27_18_17_48 | train   |        25 |         1 |                                    1 |                          3.83169 |                     15.6589 |                                        1   |                                               0        |
| zxy       | Entity_Recording_2025_09_28_15_57_38 | train   |         4 |         1 |                                    1 |                          2.24628 |                     26.7298 |                                        2   |                                               0.5      |
| zxy       | Entity_Recording_2025_09_28_16_01_55 | train   |         1 |         1 |                                    1 |                          2.13726 |                     28.0733 |                                        2   |                                               0        |
| zxy       | Entity_Recording_2025_09_28_16_12_11 | train   |        12 |         1 |                                    1 |                          2.24257 |                     26.7562 |                                        2.5 |                                               0        |
| zxy       | Entity_Recording_2025_09_28_16_25_51 | train   |        12 |         1 |                                    1 |                          2.26582 |                     26.4823 |                                        2   |                                               0.75     |
| zxy       | Entity_Recording_2025_09_28_16_35_30 | train   |         7 |         1 |                                    1 |                          2.27152 |                     26.4141 |                                        2   |                                               0.5      |

## 图表

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v289_resp_source_phase_route_audit_20260702\figures\v289_badtop10_val_test_delta.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v289_resp_source_phase_route_audit_20260702\figures\v289_resp_offset_group_summary.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v289_resp_source_phase_route_audit_20260702\figures\v289_resp_feature_screen_summary.png`

## 解释

- route gate 未通过：即使重建 RESP 周期/相位，当前呼吸源信号仍没有形成可部署候选选择收益。
- 若只有 top3 oracle 或 corr 有弱苗头，而 validation 选出的 top1 不赢 latest，不能写成差样本本质改善。

## guardrail

```json
{
  "pass": true,
  "zip_testzip": true,
  "event_n": 1167,
  "candidate_rows": 46680,
  "resp_source_feature_n": 575,
  "feature_set_n": 27,
  "uses_post_observation_any": false,
  "ok_rate": 0.919451585261354,
  "baseline_valid_ratio_median": 1.0,
  "context_period_s_median": 3.026337680887167,
  "context_bpm_median": 19.82594354190213,
  "fixed_wait_latest_badtop10": 0.6950484153471495,
  "route_viable_now": false,
  "deployable_top1_badtop10_pass": false,
  "deployable_top1_bad_ambiguous_pass": false,
  "test_best_top1_diagnostic_pass": false,
  "best_test_badtop10_top1_delta": 0.06250646788822975,
  "best_test_badtop10_corr": 0.04627740497548323,
  "reused_v260_feature_table": false,
  "test_used_for_current_feature_selection": false,
  "v288_source_guardrail_pass": true,
  "v288_source_route_viable_now": false
}
```