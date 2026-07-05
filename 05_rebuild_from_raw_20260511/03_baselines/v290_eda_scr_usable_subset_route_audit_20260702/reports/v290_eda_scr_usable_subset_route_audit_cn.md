# v290 EDA/SCR usable-subset source route audit

## 本轮目的

- 承接 v288/v289：ECG 与 RESP 源信号都没有形成可部署 top1 改善。
- 本轮回到 cleaned 200Hz EDA 源信号，重建 tonic/phasic/SCR 特征，并显式区分 EDA 可用子集。
- 仍然只做 vehicle top40 route gate，不训练轨迹融合模型。

## 标准 route gate 判定

| check                                             | requirement                                                                    | pass   | evidence                    | deployable   | route_viable_now   |
|:--------------------------------------------------|:-------------------------------------------------------------------------------|:-------|:----------------------------|:-------------|:-------------------|
| deployable_top1_val_chosen_bad_top10              | validation 选出的新生理 top1 在 test bad_top10 上低于 latest                   | False  | 0.17595165183669642         | True         | False              |
| deployable_top1_val_chosen_bad_ambiguous          | validation 选出的新生理 top1 在 test bad_top10_vehicle_ambiguous 上低于 latest | False  | 0.16008606553077698         | True         | False              |
| oracle_top3_val_test_same_direction_bad_ambiguous | 非部署 top3 上限在 val/test 歧义差样本上同向改善                               | False  | val=0.185578, test=0.067658 | False        | False              |
| test_bad_top10_any_feature_corr_gt_005            | test bad_top10 至少一个新特征集的生理距离-真实误差排序相关均值 > 0.05          | False  | 0.030608940917421466        | False        | False              |
| test_best_top1_diagnostic_beats_latest            | 即使 test-best 诊断，新生理 top1 至少有一个特征集低于 latest                   | False  | 0.14087613240668648         | False        | False              |

## EDA 可用子集 route gate 判定

| check                                       | pass   |   evidence | deployable   | val_chosen_feature_set   |   test_n | eda_subset_route_viable_now   |
|:--------------------------------------------|:-------|-----------:|:-------------|:-------------------------|---------:|:------------------------------|
| eda_usable_top1                             | False  |  0.0761956 | True         | eda_duration_dur2_top32  |      165 | False                         |
| bad_top10_eda_usable_top1                   | False  |  0.175952  | True         | eda_duration_dur2_top32  |       19 | False                         |
| bad_top10_vehicle_ambiguous_eda_usable_top1 | False  |  0.160086  | True         | eda_duration_dur2_top32  |       15 | False                         |

## validation 选择后的 test 泛化

| event_group                            | method          | deployable   | val_chosen_feature_set   |   val_delta_vs_latest_mean |   test_delta_vs_latest_mean |   test_corr_mean | test_passes_latest   | val_and_test_same_direction_gain   |
|:---------------------------------------|:----------------|:-------------|:-------------------------|---------------------------:|----------------------------:|-----------------:|:---------------------|:-----------------------------------|
| all                                    | bio_top1        | True         | eda_duration_dur2_top32  |                0.103916    |                  0.0717104  |      -0.0440954  | False                | False                              |
| all                                    | bio_top3_oracle | False        | eda_duration_dur10_top32 |               -0.000519623 |                  0.00327399 |       0.00382045 | False                | False                              |
| all                                    | bio_top5_oracle | False        | eda_duration_dur10_top32 |               -0.0410265   |                 -0.0250825  |       0.00382045 | True                 | True                               |
| vehicle_ambiguous                      | bio_top1        | True         | eda_duration_dur2_top32  |                0.131298    |                  0.0767164  |      -0.0355648  | False                | False                              |
| vehicle_ambiguous                      | bio_top3_oracle | False        | eda_all_top64            |                0.0166833   |                  0.00264397 |      -0.0131448  | False                | False                              |
| vehicle_ambiguous                      | bio_top5_oracle | False        | eda_duration_dur10_top32 |               -0.0264622   |                 -0.0220493  |      -0.0127832  | True                 | True                               |
| bad_top10                              | bio_top1        | True         | eda_duration_dur2_top32  |                0.403947    |                  0.175952   |      -0.0561081  | False                | False                              |
| bad_top10                              | bio_top3_oracle | False        | eda_all_top64            |                0.175936    |                  0.0617903  |      -0.067727   | False                | False                              |
| bad_top10                              | bio_top5_oracle | False        | eda_duration_dur10_top32 |                0.0507909   |                  0.0383648  |      -0.029209   | False                | False                              |
| bad_top10_vehicle_ambiguous            | bio_top1        | True         | eda_duration_dur2_top32  |                0.430939    |                  0.160086   |      -0.0573416  | False                | False                              |
| bad_top10_vehicle_ambiguous            | bio_top3_oracle | False        | eda_duration_dur30_top32 |                0.185578    |                  0.0676582  |      -0.0284275  | False                | False                              |
| bad_top10_vehicle_ambiguous            | bio_top5_oracle | False        | eda_duration_dur10_top32 |                0.0581592   |                  0.0389505  |      -0.0495085  | False                | False                              |
| eda_usable                             | bio_top1        | True         | eda_duration_dur2_top32  |                0.114966    |                  0.0761956  |      -0.0515526  | False                | False                              |
| eda_usable                             | bio_top3_oracle | False        | eda_duration_dur10_top32 |                0.00745129  |                  0.00636844 |       0.00500522 | False                | False                              |
| eda_usable                             | bio_top5_oracle | False        | eda_duration_dur10_top32 |               -0.0377406   |                 -0.0232745  |       0.00500522 | True                 | True                               |
| vehicle_ambiguous_eda_usable           | bio_top1        | True         | eda_duration_dur2_top32  |                0.136168    |                  0.0824028  |      -0.0403586  | False                | False                              |
| vehicle_ambiguous_eda_usable           | bio_top3_oracle | False        | eda_all_top64            |                0.023011    |                  0.00290151 |      -0.0110952  | False                | False                              |
| vehicle_ambiguous_eda_usable           | bio_top5_oracle | False        | eda_duration_dur10_top32 |               -0.0256429   |                 -0.0205176  |      -0.0112355  | True                 | True                               |
| bad_top10_eda_usable                   | bio_top1        | True         | eda_duration_dur2_top32  |                0.41697     |                  0.175952   |      -0.0561081  | False                | False                              |
| bad_top10_eda_usable                   | bio_top3_oracle | False        | eda_all_top64            |                0.18641     |                  0.0617903  |      -0.067727   | False                | False                              |
| bad_top10_eda_usable                   | bio_top5_oracle | False        | eda_duration_dur10_top32 |                0.0530535   |                  0.0383648  |      -0.029209   | False                | False                              |
| bad_top10_vehicle_ambiguous_eda_usable | bio_top1        | True         | eda_duration_dur2_top32  |                0.434968    |                  0.160086   |      -0.0573416  | False                | False                              |
| bad_top10_vehicle_ambiguous_eda_usable | bio_top3_oracle | False        | eda_duration_dur30_top32 |                0.192716    |                  0.0676582  |      -0.0284275  | False                | False                              |
| bad_top10_vehicle_ambiguous_eda_usable | bio_top5_oracle | False        | eda_duration_dur10_top32 |                0.0603961   |                  0.0389505  |      -0.0495085  | False                | False                              |

## test bad_top10 最佳 top1 诊断

| feature_set                       |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:----------------------------------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| eda_duration_delta_top32          |  19 |           0.695048 |             0.835925 |                     0.140876 |                    0.742048 |                    0.047     |       0.0306089 |
| eda_offset_delta_top32            |  19 |           0.695048 |             0.835925 |                     0.140876 |                    0.742048 |                    0.047     |       0.0306089 |
| eda_category_temporal_delta_top48 |  19 |           0.695048 |             0.858583 |                     0.163534 |                    0.744686 |                    0.0496378 |       0.0058794 |
| eda_category_quality_top48        |  19 |           0.695048 |             0.865592 |                     0.170544 |                    0.766052 |                    0.0710039 |      -0.0395695 |
| eda_duration_dur2_top32           |  19 |           0.695048 |             0.871    |                     0.175952 |                    0.75208  |                    0.0570317 |      -0.0561081 |
| eda_window_pre5_0_top24           |  19 |           0.695048 |             0.875607 |                     0.180558 |                    0.731154 |                    0.0361051 |      -0.0710518 |
| eda_window_pre2_0_top24           |  19 |           0.695048 |             0.878341 |                     0.183292 |                    0.755439 |                    0.0603905 |      -0.0640935 |
| eda_offset_endm20_top32           |  19 |           0.695048 |             0.882377 |                     0.187328 |                    0.759835 |                    0.0647867 |      -0.0395358 |

## test bad_top10_eda_usable 最佳 top1 诊断

| feature_set                       |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:----------------------------------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| eda_duration_delta_top32          |  19 |           0.695048 |             0.835925 |                     0.140876 |                    0.742048 |                    0.047     |       0.0306089 |
| eda_offset_delta_top32            |  19 |           0.695048 |             0.835925 |                     0.140876 |                    0.742048 |                    0.047     |       0.0306089 |
| eda_category_temporal_delta_top48 |  19 |           0.695048 |             0.858583 |                     0.163534 |                    0.744686 |                    0.0496378 |       0.0058794 |
| eda_category_quality_top48        |  19 |           0.695048 |             0.865592 |                     0.170544 |                    0.766052 |                    0.0710039 |      -0.0395695 |
| eda_duration_dur2_top32           |  19 |           0.695048 |             0.871    |                     0.175952 |                    0.75208  |                    0.0570317 |      -0.0561081 |
| eda_window_pre5_0_top24           |  19 |           0.695048 |             0.875607 |                     0.180558 |                    0.731154 |                    0.0361051 |      -0.0710518 |
| eda_window_pre2_0_top24           |  19 |           0.695048 |             0.878341 |                     0.183292 |                    0.755439 |                    0.0603905 |      -0.0640935 |
| eda_offset_endm20_top32           |  19 |           0.695048 |             0.882377 |                     0.187328 |                    0.759835 |                    0.0647867 |      -0.0395358 |

## bad_top10 全体分层

| feature_set                       | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:----------------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| eda_duration_delta_top32          | test    |  19 |           0.695048 |             0.835925 |                     0.140876 |                    0.742048 |                    0.047     |      0.0306089  |
| eda_offset_delta_top32            | test    |  19 |           0.695048 |             0.835925 |                     0.140876 |                    0.742048 |                    0.047     |      0.0306089  |
| eda_category_temporal_delta_top48 | test    |  19 |           0.695048 |             0.858583 |                     0.163534 |                    0.744686 |                    0.0496378 |      0.0058794  |
| eda_category_quality_top48        | test    |  19 |           0.695048 |             0.865592 |                     0.170544 |                    0.766052 |                    0.0710039 |     -0.0395695  |
| eda_duration_dur2_top32           | test    |  19 |           0.695048 |             0.871    |                     0.175952 |                    0.75208  |                    0.0570317 |     -0.0561081  |
| eda_window_pre5_0_top24           | test    |  19 |           0.695048 |             0.875607 |                     0.180558 |                    0.731154 |                    0.0361051 |     -0.0710518  |
| eda_window_pre2_0_top24           | test    |  19 |           0.695048 |             0.878341 |                     0.183292 |                    0.755439 |                    0.0603905 |     -0.0640935  |
| eda_offset_endm20_top32           | test    |  19 |           0.695048 |             0.882377 |                     0.187328 |                    0.759835 |                    0.0647867 |     -0.0395358  |
| eda_duration_dur5_top32           | test    |  19 |           0.695048 |             0.886106 |                     0.191057 |                    0.766619 |                    0.0715704 |     -0.0186438  |
| eda_duration_dur3_top32           | test    |  19 |           0.695048 |             0.887706 |                     0.192658 |                    0.727428 |                    0.0323798 |     -0.050104   |
| eda_duration_dur20_top32          | test    |  19 |           0.695048 |             0.889619 |                     0.194571 |                    0.762938 |                    0.0678896 |     -0.107496   |
| eda_window_dur5_endm1_top24       | test    |  19 |           0.695048 |             0.902357 |                     0.207308 |                    0.705995 |                    0.0109461 |     -0.0693564  |
| eda_category_scr_phasic_top48     | test    |  19 |           0.695048 |             0.909904 |                     0.214855 |                    0.72774  |                    0.0326919 |     -0.0312874  |
| eda_offset_endm1_top32            | test    |  19 |           0.695048 |             0.918614 |                     0.223565 |                    0.737696 |                    0.042648  |     -0.0848449  |
| eda_window_pre20_0_top24          | test    |  19 |           0.695048 |             0.924655 |                     0.229606 |                    0.7645   |                    0.0694518 |     -0.0917848  |
| eda_low_identity_top48            | test    |  19 |           0.695048 |             0.925889 |                     0.230841 |                    0.758611 |                    0.0635627 |     -0.00105922 |
| eda_category_morph_dynamic_top48  | test    |  19 |           0.695048 |             0.933548 |                     0.2385   |                    0.741369 |                    0.0463205 |     -0.00601332 |
| eda_all_top64                     | test    |  19 |           0.695048 |             0.937641 |                     0.242593 |                    0.756839 |                    0.0617903 |     -0.067727   |
| eda_offset_endm2_top32            | test    |  19 |           0.695048 |             0.945791 |                     0.250742 |                    0.735393 |                    0.0403442 |     -0.08622    |
| eda_duration_dur10_top32          | test    |  19 |           0.695048 |             0.950432 |                     0.255383 |                    0.819387 |                    0.124339  |     -0.029209   |
| eda_category_tonic_top48          | test    |  19 |           0.695048 |             0.950796 |                     0.255748 |                    0.719181 |                    0.0241323 |     -0.0194469  |
| eda_window_pre10_0_top24          | test    |  19 |           0.695048 |             0.952605 |                     0.257557 |                    0.789934 |                    0.0948851 |     -0.115509   |
| eda_offset_end0_top32             | test    |  19 |           0.695048 |             0.959821 |                     0.264773 |                    0.76547  |                    0.070422  |     -0.0406063  |
| eda_window_dur10_endm1_top24      | test    |  19 |           0.695048 |             0.965721 |                     0.270673 |                    0.77905  |                    0.0840018 |     -0.088252   |
| eda_offset_endm5_top32            | test    |  19 |           0.695048 |             0.987923 |                     0.292874 |                    0.716216 |                    0.0211677 |     -0.017055   |
| eda_offset_endm10_top32           | test    |  19 |           0.695048 |             1.001    |                     0.305953 |                    0.765157 |                    0.0701091 |     -0.0409066  |
| eda_duration_dur30_top32          | test    |  19 |           0.695048 |             1.01077  |                     0.31572  |                    0.753878 |                    0.0588294 |     -0.045451   |
| eda_offset_endm30_top32           | test    |  19 |           0.695048 |             1.01077  |                     0.31572  |                    0.753878 |                    0.0588294 |     -0.045451   |
| eda_category_level_top48          | test    |  19 |           0.695048 |             1.02824  |                     0.333188 |                    0.842046 |                    0.146998  |     -0.148862   |
| eda_duration_dur2_top32           | val     |  31 |           1.07279  |             1.47674  |                     0.403947 |                    1.27853  |                    0.20574   |     -0.0596535  |
| eda_window_pre20_0_top24          | val     |  31 |           1.07279  |             1.52635  |                     0.453566 |                    1.2821   |                    0.209315  |     -0.110157   |
| eda_duration_dur20_top32          | val     |  31 |           1.07279  |             1.52654  |                     0.453754 |                    1.31272  |                    0.239928  |     -0.107433   |
| eda_window_pre2_0_top24           | val     |  31 |           1.07279  |             1.57167  |                     0.498885 |                    1.30982  |                    0.237031  |     -0.0643902  |
| eda_category_scr_phasic_top48     | val     |  31 |           1.07279  |             1.6613   |                     0.588517 |                    1.40649  |                    0.333707  |     -0.0696574  |
| eda_all_top64                     | val     |  31 |           1.07279  |             1.66321  |                     0.590425 |                    1.24872  |                    0.175936  |      0.0215073  |
| eda_window_pre5_0_top24           | val     |  31 |           1.07279  |             1.66341  |                     0.59062  |                    1.30794  |                    0.235151  |     -0.0633958  |
| eda_category_level_top48          | val     |  31 |           1.07279  |             1.70886  |                     0.636077 |                    1.28022  |                    0.207429  |     -0.00474743 |
| eda_offset_endm2_top32            | val     |  31 |           1.07279  |             1.72802  |                     0.655235 |                    1.35738  |                    0.284589  |     -0.0764775  |
| eda_offset_endm10_top32           | val     |  31 |           1.07279  |             1.7314   |                     0.658611 |                    1.25599  |                    0.183203  |     -0.0584767  |
| eda_duration_dur3_top32           | val     |  31 |           1.07279  |             1.73168  |                     0.658892 |                    1.27741  |                    0.204625  |     -0.0661804  |

## bad_top10 EDA 可用子集

| feature_set                       | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:----------------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| eda_duration_delta_top32          | test    |  19 |           0.695048 |             0.835925 |                     0.140876 |                    0.742048 |                    0.047     |      0.0306089  |
| eda_offset_delta_top32            | test    |  19 |           0.695048 |             0.835925 |                     0.140876 |                    0.742048 |                    0.047     |      0.0306089  |
| eda_category_temporal_delta_top48 | test    |  19 |           0.695048 |             0.858583 |                     0.163534 |                    0.744686 |                    0.0496378 |      0.0058794  |
| eda_category_quality_top48        | test    |  19 |           0.695048 |             0.865592 |                     0.170544 |                    0.766052 |                    0.0710039 |     -0.0395695  |
| eda_duration_dur2_top32           | test    |  19 |           0.695048 |             0.871    |                     0.175952 |                    0.75208  |                    0.0570317 |     -0.0561081  |
| eda_window_pre5_0_top24           | test    |  19 |           0.695048 |             0.875607 |                     0.180558 |                    0.731154 |                    0.0361051 |     -0.0710518  |
| eda_window_pre2_0_top24           | test    |  19 |           0.695048 |             0.878341 |                     0.183292 |                    0.755439 |                    0.0603905 |     -0.0640935  |
| eda_offset_endm20_top32           | test    |  19 |           0.695048 |             0.882377 |                     0.187328 |                    0.759835 |                    0.0647867 |     -0.0395358  |
| eda_duration_dur5_top32           | test    |  19 |           0.695048 |             0.886106 |                     0.191057 |                    0.766619 |                    0.0715704 |     -0.0186438  |
| eda_duration_dur3_top32           | test    |  19 |           0.695048 |             0.887706 |                     0.192658 |                    0.727428 |                    0.0323798 |     -0.050104   |
| eda_duration_dur20_top32          | test    |  19 |           0.695048 |             0.889619 |                     0.194571 |                    0.762938 |                    0.0678896 |     -0.107496   |
| eda_window_dur5_endm1_top24       | test    |  19 |           0.695048 |             0.902357 |                     0.207308 |                    0.705995 |                    0.0109461 |     -0.0693564  |
| eda_category_scr_phasic_top48     | test    |  19 |           0.695048 |             0.909904 |                     0.214855 |                    0.72774  |                    0.0326919 |     -0.0312874  |
| eda_offset_endm1_top32            | test    |  19 |           0.695048 |             0.918614 |                     0.223565 |                    0.737696 |                    0.042648  |     -0.0848449  |
| eda_window_pre20_0_top24          | test    |  19 |           0.695048 |             0.924655 |                     0.229606 |                    0.7645   |                    0.0694518 |     -0.0917848  |
| eda_low_identity_top48            | test    |  19 |           0.695048 |             0.925889 |                     0.230841 |                    0.758611 |                    0.0635627 |     -0.00105922 |
| eda_category_morph_dynamic_top48  | test    |  19 |           0.695048 |             0.933548 |                     0.2385   |                    0.741369 |                    0.0463205 |     -0.00601332 |
| eda_all_top64                     | test    |  19 |           0.695048 |             0.937641 |                     0.242593 |                    0.756839 |                    0.0617903 |     -0.067727   |
| eda_offset_endm2_top32            | test    |  19 |           0.695048 |             0.945791 |                     0.250742 |                    0.735393 |                    0.0403442 |     -0.08622    |
| eda_duration_dur10_top32          | test    |  19 |           0.695048 |             0.950432 |                     0.255383 |                    0.819387 |                    0.124339  |     -0.029209   |
| eda_category_tonic_top48          | test    |  19 |           0.695048 |             0.950796 |                     0.255748 |                    0.719181 |                    0.0241323 |     -0.0194469  |
| eda_window_pre10_0_top24          | test    |  19 |           0.695048 |             0.952605 |                     0.257557 |                    0.789934 |                    0.0948851 |     -0.115509   |
| eda_offset_end0_top32             | test    |  19 |           0.695048 |             0.959821 |                     0.264773 |                    0.76547  |                    0.070422  |     -0.0406063  |
| eda_window_dur10_endm1_top24      | test    |  19 |           0.695048 |             0.965721 |                     0.270673 |                    0.77905  |                    0.0840018 |     -0.088252   |
| eda_offset_endm5_top32            | test    |  19 |           0.695048 |             0.987923 |                     0.292874 |                    0.716216 |                    0.0211677 |     -0.017055   |
| eda_offset_endm10_top32           | test    |  19 |           0.695048 |             1.001    |                     0.305953 |                    0.765157 |                    0.0701091 |     -0.0409066  |
| eda_duration_dur30_top32          | test    |  19 |           0.695048 |             1.01077  |                     0.31572  |                    0.753878 |                    0.0588294 |     -0.045451   |
| eda_offset_endm30_top32           | test    |  19 |           0.695048 |             1.01077  |                     0.31572  |                    0.753878 |                    0.0588294 |     -0.045451   |
| eda_category_level_top48          | test    |  19 |           0.695048 |             1.02824  |                     0.333188 |                    0.842046 |                    0.146998  |     -0.148862   |
| eda_duration_dur2_top32           | val     |  29 |           1.03619  |             1.45316  |                     0.41697  |                    1.25446  |                    0.218269  |     -0.0651189  |
| eda_window_pre20_0_top24          | val     |  29 |           1.03619  |             1.5062   |                     0.470011 |                    1.25829  |                    0.222091  |     -0.110418   |
| eda_duration_dur20_top32          | val     |  29 |           1.03619  |             1.50641  |                     0.470212 |                    1.29101  |                    0.254815  |     -0.106449   |
| eda_window_pre2_0_top24           | val     |  29 |           1.03619  |             1.55465  |                     0.518455 |                    1.28791  |                    0.251719  |     -0.0715844  |
| eda_category_scr_phasic_top48     | val     |  29 |           1.03619  |             1.65046  |                     0.614269 |                    1.39126  |                    0.355061  |     -0.0735861  |
| eda_all_top64                     | val     |  29 |           1.03619  |             1.6525   |                     0.616308 |                    1.2226   |                    0.18641   |      0.0241942  |
| eda_window_pre5_0_top24           | val     |  29 |           1.03619  |             1.65271  |                     0.616516 |                    1.2859   |                    0.249708  |     -0.0672054  |
| eda_category_level_top48          | val     |  29 |           1.03619  |             1.7013   |                     0.665108 |                    1.25627  |                    0.220075  |      0.00458757 |
| eda_offset_endm2_top32            | val     |  29 |           1.03619  |             1.72178  |                     0.685588 |                    1.33875  |                    0.302556  |     -0.072027   |
| eda_offset_endm10_top32           | val     |  29 |           1.03619  |             1.72539  |                     0.689196 |                    1.23037  |                    0.194178  |     -0.0591734  |
| eda_duration_dur3_top32           | val     |  29 |           1.03619  |             1.72569  |                     0.689497 |                    1.25327  |                    0.217077  |     -0.0650294  |

## bad_top10_vehicle_ambiguous EDA 可用子集

| feature_set                       | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:----------------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| eda_duration_delta_top32          | test    |  15 |           0.744423 |             0.857199 |                     0.112776 |                    0.802148 |                   0.0577255  |      0.0253571  |
| eda_offset_delta_top32            | test    |  15 |           0.744423 |             0.857199 |                     0.112776 |                    0.802148 |                   0.0577255  |      0.0253571  |
| eda_category_temporal_delta_top48 | test    |  15 |           0.744423 |             0.889704 |                     0.145281 |                    0.790083 |                   0.0456603  |      0.0114703  |
| eda_duration_dur2_top32           | test    |  15 |           0.744423 |             0.904509 |                     0.160086 |                    0.812254 |                   0.0678314  |     -0.0573416  |
| eda_category_quality_top48        | test    |  15 |           0.744423 |             0.908752 |                     0.164329 |                    0.825653 |                   0.0812304  |     -0.0341467  |
| eda_window_pre2_0_top24           | test    |  15 |           0.744423 |             0.913807 |                     0.169384 |                    0.814638 |                   0.0702148  |     -0.0730079  |
| eda_window_dur5_endm1_top24       | test    |  15 |           0.744423 |             0.916306 |                     0.171883 |                    0.757643 |                   0.0132198  |     -0.0582718  |
| eda_window_pre5_0_top24           | test    |  15 |           0.744423 |             0.922312 |                     0.177889 |                    0.785747 |                   0.0413244  |     -0.0683818  |
| eda_offset_endm20_top32           | test    |  15 |           0.744423 |             0.923138 |                     0.178715 |                    0.813276 |                   0.0688534  |     -0.0259294  |
| eda_category_scr_phasic_top48     | test    |  15 |           0.744423 |             0.924031 |                     0.179608 |                    0.781442 |                   0.0370194  |     -0.0164739  |
| eda_offset_endm2_top32            | test    |  15 |           0.744423 |             0.926363 |                     0.18194  |                    0.788166 |                   0.043743   |     -0.0620006  |
| eda_duration_dur3_top32           | test    |  15 |           0.744423 |             0.92702  |                     0.182597 |                    0.765174 |                   0.0207512  |     -0.0319162  |
| eda_duration_dur20_top32          | test    |  15 |           0.744423 |             0.932642 |                     0.188219 |                    0.826594 |                   0.0821712  |     -0.0899103  |
| eda_category_morph_dynamic_top48  | test    |  15 |           0.744423 |             0.955444 |                     0.211021 |                    0.786719 |                   0.0422956  |     -0.0276286  |
| eda_window_pre20_0_top24          | test    |  15 |           0.744423 |             0.962822 |                     0.218399 |                    0.835483 |                   0.0910601  |     -0.0796714  |
| eda_duration_dur5_top32           | test    |  15 |           0.744423 |             0.970059 |                     0.225636 |                    0.829175 |                   0.0847523  |     -0.00218652 |
| eda_all_top64                     | test    |  15 |           0.744423 |             0.982747 |                     0.238324 |                    0.805476 |                   0.0610535  |     -0.0600932  |
| eda_window_pre10_0_top24          | test    |  15 |           0.744423 |             0.986952 |                     0.242529 |                    0.793306 |                   0.0488829  |     -0.0781193  |
| eda_category_tonic_top48          | test    |  15 |           0.744423 |             0.99082  |                     0.246397 |                    0.760706 |                   0.0162832  |     -0.00509692 |
| eda_low_identity_top48            | test    |  15 |           0.744423 |             0.991504 |                     0.247081 |                    0.819766 |                   0.0753434  |     -0.0171849  |
| eda_duration_dur10_top32          | test    |  15 |           0.744423 |             0.999832 |                     0.255409 |                    0.848378 |                   0.103955   |     -0.0495085  |
| eda_offset_endm1_top32            | test    |  15 |           0.744423 |             1.00686  |                     0.262436 |                    0.793752 |                   0.0493287  |     -0.0727787  |
| eda_window_dur10_endm1_top24      | test    |  15 |           0.744423 |             1.02201  |                     0.277588 |                    0.832529 |                   0.0881056  |     -0.0690042  |
| eda_offset_end0_top32             | test    |  15 |           0.744423 |             1.02226  |                     0.277836 |                    0.829215 |                   0.0847925  |     -0.0605645  |
| eda_offset_endm5_top32            | test    |  15 |           0.744423 |             1.02814  |                     0.283718 |                    0.753299 |                   0.00887564 |      0.0120379  |
| eda_duration_dur30_top32          | test    |  15 |           0.744423 |             1.03472  |                     0.290299 |                    0.812081 |                   0.0676582  |     -0.0284275  |
| eda_offset_endm30_top32           | test    |  15 |           0.744423 |             1.03472  |                     0.290299 |                    0.812081 |                   0.0676582  |     -0.0284275  |
| eda_offset_endm10_top32           | test    |  15 |           0.744423 |             1.05297  |                     0.308544 |                    0.817967 |                   0.0735443  |     -0.010179   |
| eda_category_level_top48          | test    |  15 |           0.744423 |             1.08496  |                     0.340533 |                    0.88943  |                   0.145007   |     -0.143517   |
| eda_duration_dur2_top32           | val     |  26 |           1.00556  |             1.44052  |                     0.434968 |                    1.23934  |                   0.233787   |     -0.0671671  |
| eda_duration_dur20_top32          | val     |  26 |           1.00556  |             1.51308  |                     0.507522 |                    1.28014  |                   0.27458    |     -0.103209   |
| eda_window_pre20_0_top24          | val     |  26 |           1.00556  |             1.52079  |                     0.515234 |                    1.24426  |                   0.238708   |     -0.106044   |
| eda_window_pre2_0_top24           | val     |  26 |           1.00556  |             1.55372  |                     0.548163 |                    1.25685  |                   0.251297   |     -0.0729943  |
| eda_window_pre5_0_top24           | val     |  26 |           1.00556  |             1.64102  |                     0.635466 |                    1.28366  |                   0.278102   |     -0.0671946  |
| eda_all_top64                     | val     |  26 |           1.00556  |             1.64911  |                     0.643556 |                    1.21131  |                   0.205753   |      0.030964   |
| eda_category_scr_phasic_top48     | val     |  26 |           1.00556  |             1.65471  |                     0.649149 |                    1.38494  |                   0.379385   |     -0.0718788  |
| eda_category_level_top48          | val     |  26 |           1.00556  |             1.70448  |                     0.698926 |                    1.23543  |                   0.229874   |      0.0132657  |
| eda_window_pre10_0_top24          | val     |  26 |           1.00556  |             1.73256  |                     0.727008 |                    1.29352  |                   0.28796    |     -0.0887264  |
| eda_offset_endm2_top32            | val     |  26 |           1.00556  |             1.74995  |                     0.744392 |                    1.32769  |                   0.322138   |     -0.0673126  |
| eda_offset_endm10_top32           | val     |  26 |           1.00556  |             1.75698  |                     0.751425 |                    1.22214  |                   0.216584   |     -0.0598251  |

## feature set 审计

| feature_set                       | group_type   | group_value    |   candidate_feature_n |   feature_n |   behavior_eta_max |   bad_eta_max |   identity_eta_median |
|:----------------------------------|:-------------|:---------------|----------------------:|------------:|-------------------:|--------------:|----------------------:|
| eda_all_top64                     | all          | all            |                   431 |          64 |          0.046282  |    0.0157978  |             0.0898218 |
| eda_low_identity_top48            | identity     | low_identity   |                   243 |          48 |          0.0157978 |    0.0157978  |             0.065153  |
| eda_category_scr_phasic_top48     | category     | scr_phasic     |                    54 |          48 |          0.026966  |    0.00491103 |             0.176234  |
| eda_category_tonic_top48          | category     | tonic          |                    97 |          48 |          0.046282  |    0.00347792 |             0.0846178 |
| eda_category_morph_dynamic_top48  | category     | morph_dynamic  |                   132 |          48 |          0.0157978 |    0.0157978  |             0.0821157 |
| eda_category_level_top48          | category     | level          |                    51 |          48 |          0.0109942 |    0.0109942  |             0.0753014 |
| eda_category_quality_top48        | category     | quality        |                    34 |          34 |          0.046282  |    0.00347792 |             1         |
| eda_category_temporal_delta_top48 | category     | temporal_delta |                    63 |          48 |          0.0154759 |    0.0154759  |             0.0493686 |
| eda_offset_end0_top32             | offset       | end0           |                   119 |          32 |          0.046282  |    0.00729097 |             0.109164  |
| eda_offset_endm1_top32            | offset       | endm1          |                    58 |          32 |          0.046282  |    0.00347792 |             0.0966913 |
| eda_offset_endm2_top32            | offset       | endm2          |                    61 |          32 |          0.046282  |    0.00734114 |             0.0832846 |
| eda_offset_endm5_top32            | offset       | endm5          |                    29 |          29 |          0.046282  |    0.00347792 |             0.0898051 |
| eda_offset_endm10_top32           | offset       | endm10         |                    29 |          29 |          0.046282  |    0.0134719  |             0.0821025 |
| eda_offset_endm20_top32           | offset       | endm20         |                    29 |          29 |          0.046282  |    0.0157978  |             0.0719158 |
| eda_offset_endm30_top32           | offset       | endm30         |                    32 |          32 |          0.046282  |    0.00808669 |             0.11601   |
| eda_offset_delta_top32            | offset       | delta          |                    63 |          32 |          0.0154759 |    0.0154759  |             0.0493686 |
| eda_duration_dur2_top32           | duration     | dur2           |                    29 |          29 |          0.046282  |    0.00347792 |             0.10556   |
| eda_duration_dur3_top32           | duration     | dur3           |                    29 |          29 |          0.046282  |    0.00347792 |             0.070385  |
| eda_duration_dur5_top32           | duration     | dur5           |                    87 |          32 |          0.046282  |    0.00347792 |             0.0949991 |
| eda_duration_dur10_top32          | duration     | dur10          |                   116 |          32 |          0.046282  |    0.0157978  |             0.0851762 |
| eda_duration_dur20_top32          | duration     | dur20          |                    64 |          32 |          0.046282  |    0.00734114 |             0.113035  |
| eda_duration_dur30_top32          | duration     | dur30          |                    32 |          32 |          0.046282  |    0.00808669 |             0.11601   |
| eda_duration_delta_top32          | duration     | delta          |                    63 |          32 |          0.0154759 |    0.0154759  |             0.0493686 |
| eda_window_pre2_0_top24           | window       | pre2_0         |                    29 |          24 |          0.046282  |    0.00347792 |             0.10556   |
| eda_window_pre5_0_top24           | window       | pre5_0         |                    29 |          24 |          0.046282  |    0.00347792 |             0.104713  |
| eda_window_pre10_0_top24          | window       | pre10_0        |                    29 |          24 |          0.046282  |    0.00347792 |             0.109164  |
| eda_window_pre20_0_top24          | window       | pre20_0        |                    32 |          24 |          0.046282  |    0.00729097 |             0.119794  |
| eda_window_dur5_endm1_top24       | window       | dur5_endm1     |                    29 |          24 |          0.046282  |    0.00347792 |             0.0949991 |
| eda_window_dur10_endm1_top24      | window       | dur10_endm1    |                    29 |          24 |          0.046282  |    0.00347792 |             0.0981736 |

## train-only EDA feature screen 摘要

| feature_category   | offset_group   | duration_group   |   feature_n |   behavior_eta_max |   bad_eta_max |   identity_eta_median |   behavior_identity_score_max |
|:-------------------|:---------------|:-----------------|------------:|-------------------:|--------------:|----------------------:|------------------------------:|
| temporal_delta     | delta          | delta            |          78 |         0.0549536  |    0.0288934  |             0.0679729 |                     0.635759  |
| level              | endm2          | dur20            |           5 |         0.010618   |    0.00244372 |             0.0743202 |                     0.465426  |
| tonic              | end0           | dur10            |           8 |         0.046282   |    0.00347792 |             0.0872607 |                     0.389036  |
| tonic              | endm1          | dur5             |           8 |         0.046282   |    0.00347792 |             0.0906931 |                     0.374928  |
| tonic              | endm2          | dur3             |           8 |         0.046282   |    0.00347792 |             0.0876074 |                     0.360718  |
| tonic              | endm1          | dur10            |           8 |         0.046282   |    0.00347792 |             0.0809551 |                     0.347742  |
| tonic              | end0           | dur5             |           8 |         0.046282   |    0.00347792 |             0.10119   |                     0.334257  |
| morph_dynamic      | endm2          | dur3             |          11 |         0.00973333 |    0.0010195  |             0.0631385 |                     0.268562  |
| morph_dynamic      | endm20         | dur10            |          11 |         0.0157978  |    0.0157978  |             0.0635202 |                     0.224831  |
| level              | endm20         | dur10            |           5 |         0.0109942  |    0.0109942  |             0.0666621 |                     0.146291  |
| level              | endm1          | dur10            |           5 |         0.0146145  |    0.0039719  |             0.10569   |                     0.145118  |
| tonic              | endm30         | dur30            |           8 |         0.046282   |    0.00347792 |             0.107985  |                     0.142377  |
| tonic              | endm5          | dur5             |           8 |         0.046282   |    0.00347792 |             0.0669691 |                     0.141254  |
| level              | end0           | dur20            |           5 |         0.0109126  |    0.00235224 |             0.0817076 |                     0.140668  |
| level              | end0           | dur10            |           5 |         0.018715   |    0.00725953 |             0.135283  |                     0.137403  |
| morph_dynamic      | endm10         | dur10            |          11 |         0.0134719  |    0.0134719  |             0.0821025 |                     0.126965  |
| tonic              | end0           | dur2             |           8 |         0.046282   |    0.00347792 |             0.114821  |                     0.120113  |
| morph_dynamic      | end0           | dur10            |          11 |         0.00501012 |    0.0016471  |             0.0821289 |                     0.11983   |
| level              | endm2          | dur3             |           5 |         0.00759412 |    0.00379242 |             0.0598749 |                     0.106541  |
| level              | endm5          | dur5             |           5 |         0.0225343  |    0.00506319 |             0.0898051 |                     0.103813  |
| level              | endm10         | dur10            |           5 |         0.0123882  |    0.00670526 |             0.0717039 |                     0.103656  |
| level              | endm30         | dur30            |           5 |         0.0105141  |    0.00808669 |             0.086558  |                     0.102442  |
| morph_dynamic      | end0           | dur2             |          11 |         0.00857207 |    0.00106984 |             0.0969837 |                     0.088903  |
| morph_dynamic      | endm30         | dur30            |          11 |         0.011551   |    0.00145554 |             0.0855297 |                     0.0864246 |
| morph_dynamic      | endm1          | dur5             |          11 |         0.00686785 |    0.00118044 |             0.0733636 |                     0.0823843 |
| level              | endm1          | dur5             |           5 |         0.0174612  |    0.0174612  |             0.0949991 |                     0.0819526 |
| morph_dynamic      | end0           | dur20            |          11 |         0.00729097 |    0.00729097 |             0.079614  |                     0.0813597 |
| morph_dynamic      | endm2          | dur20            |          11 |         0.00734114 |    0.00734114 |             0.0741988 |                     0.0812805 |
| level              | end0           | dur5             |           5 |         0.0128584  |    0.00573919 |             0.0910345 |                     0.0802392 |
| scr_phasic         | endm20         | dur10            |           6 |         0.0188668  |    0.0091177  |             0.108915  |                     0.0764283 |
| morph_dynamic      | end0           | dur5             |          11 |         0.00568799 |    0.00120898 |             0.0942003 |                     0.0747676 |
| tonic              | endm20         | dur10            |           8 |         0.046282   |    0.00347792 |             0.07914   |                     0.0741214 |
| scr_phasic         | endm30         | dur30            |           6 |         0.0266041  |    0.00355987 |             0.242012  |                     0.0687916 |
| level              | end0           | dur2             |           5 |         0.00645044 |    0.00645044 |             0.0866897 |                     0.0676463 |
| scr_phasic         | endm2          | dur20            |           6 |         0.0218884  |    0.00338644 |             0.119742  |                     0.0567166 |
| scr_phasic         | end0           | dur2             |           6 |         0.0154119  |    0.00238292 |             0.255438  |                     0.0564545 |
| morph_dynamic      | endm1          | dur10            |          11 |         0.00601058 |    0.00130515 |             0.0981736 |                     0.0564469 |
| scr_phasic         | endm5          | dur5             |           6 |         0.020211   |    0.00100237 |             0.150686  |                     0.053595  |
| tonic              | endm10         | dur10            |           8 |         0.046282   |    0.00347792 |             0.06133   |                     0.05012   |
| scr_phasic         | endm1          | dur10            |           6 |         0.026966   |    0.00283403 |             0.147534  |                     0.0486138 |
| scr_phasic         | end0           | dur10            |           6 |         0.0269107  |    0.0012354  |             0.217997  |                     0.0479029 |
| quality            | end0           | dur10            |           2 |         0.046282   |    0.00347792 |             1         |                     0.0458237 |
| quality            | end0           | dur2             |           2 |         0.046282   |    0.00347792 |             1         |                     0.0458237 |
| quality            | end0           | dur20            |           2 |         0.046282   |    0.00347792 |             1         |                     0.0458237 |
| quality            | end0           | dur5             |           2 |         0.046282   |    0.00347792 |             1         |                     0.0458237 |
| quality            | endm1          | dur10            |           2 |         0.046282   |    0.00347792 |             1         |                     0.0458237 |
| quality            | endm1          | dur5             |           2 |         0.046282   |    0.00347792 |             1         |                     0.0458237 |
| quality            | endm10         | dur10            |           2 |         0.046282   |    0.00347792 |             1         |                     0.0458237 |
| quality            | endm2          | dur20            |           2 |         0.046282   |    0.00347792 |             1         |                     0.0458237 |
| quality            | endm2          | dur3             |           2 |         0.046282   |    0.00347792 |             1         |                     0.0458237 |

## EDA 质量摘要

| subject   | recording                            | split   |   event_n |   ok_rate | recording_usable   |   event_usable_rate |   bio290_recording_raw_spread_median |   bio290_baseline_raw_spread_median |
|:----------|:-------------------------------------|:--------|----------:|----------:|:-------------------|--------------------:|-------------------------------------:|------------------------------------:|
| cwh       | Entity_Recording_2025_09_26_19_35_47 | test    |        10 |         1 | True               |                   1 |                           2.04587    |                          1.41913    |
| cwh       | Entity_Recording_2025_09_26_19_45_40 | test    |        11 |         1 | True               |                   1 |                           2.42118    |                          1.89213    |
| cwh       | Entity_Recording_2025_09_26_19_56_16 | test    |        11 |         1 | True               |                   1 |                           3.10928    |                          1.87419    |
| cwh       | Entity_Recording_2025_09_26_20_06_19 | test    |        14 |         1 | True               |                   1 |                           2.98269    |                          1.59224    |
| lx        | Entity_Recording_2025_09_26_08_58_43 | test    |         2 |         1 | True               |                   1 |                          12.1264     |                         10.3715     |
| lx        | Entity_Recording_2025_09_26_09_17_22 | test    |        11 |         1 | True               |                   1 |                           8.82642    |                          6.59552    |
| rjy       | Entity_Recording_2025_09_28_19_33_26 | test    |        17 |         0 | False              |                   0 |                         nan          |                        nan          |
| rjy       | Entity_Recording_2025_09_28_19_44_42 | test    |         2 |         0 | False              |                   0 |                         nan          |                        nan          |
| rjy       | Entity_Recording_2025_09_28_19_51_44 | test    |        19 |         1 | True               |                   1 |                           4.35253    |                          2.30046    |
| rjy       | Entity_Recording_2025_09_28_20_02_20 | test    |        25 |         1 | True               |                   1 |                           6.37786    |                          3.81686    |
| rjy       | Entity_Recording_2025_09_28_20_15_42 | test    |        19 |         1 | True               |                   1 |                           5.39985    |                          3.58669    |
| tyy       | Entity_Recording_2025_09_28_14_23_43 | test    |        24 |         1 | True               |                   1 |                          24.7281     |                          0.20516    |
| tyy       | Entity_Recording_2025_09_28_14_40_01 | test    |         7 |         1 | True               |                   1 |                           0.00384865 |                          0.00438619 |
| tyy       | Entity_Recording_2025_09_28_14_57_17 | test    |        12 |         1 | True               |                   1 |                          12.8092     |                          4.36592    |
| byx       | Entity_Recording_2025_09_28_17_05_51 | train   |        19 |         1 | False              |                   0 |                           0          |                          0          |
| byx       | Entity_Recording_2025_09_28_17_15_52 | train   |        18 |         1 | False              |                   0 |                           0          |                          0          |
| byx       | Entity_Recording_2025_09_28_17_25_18 | train   |        23 |         1 | False              |                   0 |                           0          |                          0          |
| byx       | Entity_Recording_2025_09_28_17_35_43 | train   |        25 |         1 | False              |                   0 |                           0          |                          0          |
| byx       | Entity_Recording_2025_09_28_17_46_00 | train   |        17 |         1 | False              |                   0 |                           0          |                          0          |
| gf        | Entity_Recording_2025_09_26_10_03_00 | train   |         1 |         1 | True               |                   1 |                           4.18292    |                          1.48702    |
| gf        | Entity_Recording_2025_09_26_10_18_49 | train   |         9 |         1 | True               |                   1 |                           3.8229     |                          3.36691    |
| gf        | Entity_Recording_2025_09_26_10_30_12 | train   |         9 |         1 | True               |                   1 |                           4.80835    |                          2.55717    |
| gf        | Entity_Recording_2025_09_26_10_40_59 | train   |         9 |         1 | True               |                   1 |                           3.69373    |                          1.77804    |
| gf        | Entity_Recording_2025_09_26_10_52_57 | train   |         8 |         1 | True               |                   1 |                           4.1539     |                          2.39047    |
| hzh       | Entity_Recording_2025_09_26_20_50_27 | train   |        27 |         1 | True               |                   1 |                           0.409315   |                          0.00698144 |
| hzh       | Entity_Recording_2025_09_26_21_03_19 | train   |        24 |         1 | True               |                   1 |                           0.0723546  |                          0.0172031  |
| hzh       | Entity_Recording_2025_09_26_21_17_02 | train   |        10 |         1 | True               |                   1 |                           0.049089   |                          0.0170657  |
| hzh       | Entity_Recording_2025_09_27_19_22_27 | train   |        17 |         1 | True               |                   1 |                          10.2748     |                          1.89215    |
| hzh       | Entity_Recording_2025_09_27_19_33_25 | train   |        19 |         1 | True               |                   1 |                           0.773789   |                          0.00245191 |
| hzh       | Entity_Recording_2025_09_27_19_44_05 | train   |        21 |         1 | True               |                   1 |                           2.0974     |                          0.002367   |
| jy        | Entity_Recording_2025_09_26_17_17_11 | train   |         8 |         1 | True               |                   1 |                           5.94677    |                          2.27206    |
| jy        | Entity_Recording_2025_09_26_17_29_44 | train   |        11 |         1 | True               |                   1 |                           4.78111    |                          2.51089    |
| jy        | Entity_Recording_2025_09_26_17_40_51 | train   |        10 |         1 | True               |                   1 |                           3.53758    |                          1.67435    |
| jy        | Entity_Recording_2025_09_26_17_51_46 | train   |         3 |         1 | True               |                   1 |                           4.06859    |                          2.47355    |
| jy        | Entity_Recording_2025_09_26_18_01_40 | train   |        10 |         1 | True               |                   1 |                           3.71035    |                          1.52323    |
| xst       | Entity_Recording_2025_09_26_11_34_18 | train   |         6 |         1 | True               |                   1 |                           3.99631    |                          2.27745    |
| yyl       | Entity_Recording_2025_09_28_09_14_23 | train   |        21 |         1 | True               |                   1 |                           4.8648     |                          2.10808    |
| yyl       | Entity_Recording_2025_09_28_09_29_01 | train   |        28 |         1 | True               |                   1 |                           5.98108    |                          2.81315    |
| yyl       | Entity_Recording_2025_09_28_09_39_01 | train   |        20 |         1 | True               |                   1 |                           4.83443    |                          3.08552    |
| yyl       | Entity_Recording_2025_09_28_09_49_11 | train   |        18 |         1 | True               |                   1 |                           5.78239    |                          4.25758    |
| yzy       | Entity_Recording_2025_09_27_14_13_03 | train   |        23 |         1 | True               |                   1 |                           3.34694    |                          2.3027     |
| yzy       | Entity_Recording_2025_09_27_14_26_04 | train   |        21 |         1 | True               |                   1 |                           4.18177    |                          2.66987    |
| yzy       | Entity_Recording_2025_09_27_14_37_08 | train   |        21 |         1 | True               |                   1 |                           5.01968    |                          3.03253    |
| yzy       | Entity_Recording_2025_09_27_15_04_26 | train   |         1 |         1 | True               |                   1 |                           5.49285    |                          5.14141    |
| yzy       | Entity_Recording_2025_09_27_15_07_57 | train   |        13 |         1 | True               |                   1 |                           7.8206     |                          2.42254    |
| zt        | Entity_Recording_2025_09_28_11_20_08 | train   |        15 |         1 | True               |                   1 |                           7.43619    |                          3.07737    |
| zx        | Entity_Recording_2025_09_27_16_32_00 | train   |        24 |         0 | False              |                   0 |                         nan          |                        nan          |
| zx        | Entity_Recording_2025_09_27_16_46_13 | train   |        26 |         0 | False              |                   0 |                         nan          |                        nan          |
| zx        | Entity_Recording_2025_09_27_17_14_07 | train   |        21 |         0 | False              |                   0 |                         nan          |                        nan          |
| zx        | Entity_Recording_2025_09_27_17_25_16 | train   |         3 |         0 | False              |                   0 |                         nan          |                        nan          |
| zx        | Entity_Recording_2025_09_27_17_29_08 | train   |         1 |         0 | False              |                   0 |                         nan          |                        nan          |
| zx        | Entity_Recording_2025_09_27_17_45_11 | train   |        27 |         1 | True               |                   1 |                           2.12979    |                          0.917463   |
| zx        | Entity_Recording_2025_09_27_17_56_42 | train   |         3 |         1 | True               |                   1 |                           1.7961     |                          0.687449   |
| zx        | Entity_Recording_2025_09_27_18_07_01 | train   |        23 |         1 | True               |                   1 |                           1.27731    |                          0.597228   |
| zx        | Entity_Recording_2025_09_27_18_17_48 | train   |        25 |         1 | True               |                   1 |                           0.931578   |                          0.51082    |
| zxy       | Entity_Recording_2025_09_28_15_57_38 | train   |         4 |         1 | True               |                   1 |                           6.65733    |                          3.86612    |
| zxy       | Entity_Recording_2025_09_28_16_01_55 | train   |         1 |         1 | True               |                   1 |                           4.72787    |                          2.55998    |
| zxy       | Entity_Recording_2025_09_28_16_12_11 | train   |        12 |         1 | True               |                   1 |                           9.23652    |                          7.07523    |
| zxy       | Entity_Recording_2025_09_28_16_25_51 | train   |        12 |         1 | True               |                   1 |                           9.23093    |                          3.73281    |
| zxy       | Entity_Recording_2025_09_28_16_35_30 | train   |         7 |         1 | True               |                   1 |                           6.96591    |                          5.07201    |
| gzj       | Entity_Recording_2025_09_27_11_38_49 | val     |         1 |         1 | True               |                   1 |                           2.06365    |                          1.76708    |
| gzj       | Entity_Recording_2025_09_27_11_41_47 | val     |        26 |         1 | True               |                   1 |                           3.78423    |                          1.98789    |
| gzj       | Entity_Recording_2025_09_27_11_53_25 | val     |        17 |         1 | True               |                   1 |                           2.71184    |                          1.12126    |
| gzj       | Entity_Recording_2025_09_27_12_04_23 | val     |         7 |         1 | True               |                   1 |                           1.90648    |                          0.63801    |
| gzj       | Entity_Recording_2025_09_27_12_17_12 | val     |        23 |         1 | True               |                   1 |                           2.38695    |                          0.650841   |
| gzj       | Entity_Recording_2025_09_27_12_28_14 | val     |        31 |         1 | True               |                   1 |                           4.47285    |                          2.25564    |
| lxy       | Entity_Recording_2025_09_28_17_55_52 | val     |        22 |         1 | False              |                   0 |                           0          |                          0          |
| lxy       | Entity_Recording_2025_09_28_18_06_16 | val     |        19 |         1 | False              |                   0 |                           0          |                          0          |
| lxy       | Entity_Recording_2025_09_28_18_19_35 | val     |        24 |         1 | False              |                   0 |                           0          |                          0          |
| txj       | Entity_Recording_2025_09_27_08_40_46 | val     |        23 |         1 | True               |                   1 |                          16.29       |                          4.75299    |
| txj       | Entity_Recording_2025_09_27_08_53_44 | val     |        31 |         1 | True               |                   1 |                           6.25839    |                          3.59333    |
| txj       | Entity_Recording_2025_09_27_09_06_19 | val     |        23 |         1 | True               |                   1 |                           4.62747    |                          2.41986    |
| txj       | Entity_Recording_2025_09_27_09_17_11 | val     |        14 |         1 | True               |                   1 |                           2.8566     |                          2.09187    |
| zdq       | Entity_Recording_2025_09_26_15_14_51 | val     |         8 |         1 | True               |                   1 |                           9.61927    |                          3.16879    |
| zdq       | Entity_Recording_2025_09_26_15_21_51 | val     |         3 |         1 | True               |                   1 |                           2.54121    |                          1.211      |
| zdq       | Entity_Recording_2025_09_26_15_27_09 | val     |         8 |         1 | True               |                   1 |                           5.60403    |                          1.62944    |
| zdq       | Entity_Recording_2025_09_26_15_37_30 | val     |        10 |         1 | True               |                   1 |                           2.17898    |                          1.14677    |
| zdq       | Entity_Recording_2025_09_26_15_52_46 | val     |         8 |         1 | True               |                   1 |                           6.07225    |                          3.8737     |
| zdq       | Entity_Recording_2025_09_26_16_03_48 | val     |        11 |         1 | True               |                   1 |                           1.13992    |                          0.371119   |

## 图表

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v290_eda_scr_usable_subset_route_audit_20260702\figures\v290_badtop10_val_test_delta.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v290_eda_scr_usable_subset_route_audit_20260702\figures\v290_eda_usable_subset_delta.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v290_eda_scr_usable_subset_route_audit_20260702\figures\v290_eda_feature_screen_summary.png`

## 解释

- 标准 route gate 未通过：EDA/SCR 未形成全体样本上的可部署 top1 改善。
- EDA 可用子集 route gate 也未通过：近常数/缺失记录不是唯一瓶颈。
- 若只有 top3 oracle 或 test-best 诊断变好，不能写成可部署模型改善。

## guardrail

```json
{
  "pass": true,
  "zip_testzip": true,
  "event_n": 1167,
  "candidate_rows": 46680,
  "eda_source_feature_n": 473,
  "feature_set_n": 29,
  "uses_post_observation_any": false,
  "ok_rate": 0.919451585261354,
  "eda_recording_usable_event_rate": 0.7763496143958869,
  "eda_event_usable_rate": 0.7763496143958869,
  "eda_event_usable_n": 906,
  "fixed_wait_latest_badtop10": 0.6950484153471495,
  "route_viable_now": false,
  "eda_subset_route_viable_now": false,
  "deployable_top1_badtop10_pass": false,
  "deployable_top1_bad_ambiguous_pass": false,
  "test_best_top1_diagnostic_pass": false,
  "best_test_badtop10_top1_delta": 0.14087613240668648,
  "best_test_badtop10_corr": 0.030608940917421466,
  "best_test_badtop10_eda_usable_top1_delta": 0.14087613240668648,
  "reused_v260_feature_table": false,
  "test_used_for_current_feature_selection": false,
  "v289_source_guardrail_pass": true,
  "v289_source_route_viable_now": false
}
```