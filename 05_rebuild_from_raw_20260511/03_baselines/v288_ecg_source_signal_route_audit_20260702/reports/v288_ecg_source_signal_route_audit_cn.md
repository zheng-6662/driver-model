# v288 ECG source-signal route audit

## 本轮目的

- 承接 v287：ECG 最近 1-2 秒有弱诊断苗头，但 v287 的现成 shape-state 特征没有形成可部署改善。
- 本轮回到 cleaned 200Hz ECG 源信号，重新提取 R 峰/RR、形态、质量和因果同步偏移特征。
- 仍然使用 v284/v285 同一个 vehicle top40 route gate；本轮不训练轨迹融合模型。

## route gate 判定

| check                                             | requirement                                                                    | pass   | evidence                    | deployable   | route_viable_now   |
|:--------------------------------------------------|:-------------------------------------------------------------------------------|:-------|:----------------------------|:-------------|:-------------------|
| deployable_top1_val_chosen_bad_top10              | validation 选出的新生理 top1 在 test bad_top10 上低于 latest                   | False  | 0.15560306843958402         | True         | False              |
| deployable_top1_val_chosen_bad_ambiguous          | validation 选出的新生理 top1 在 test bad_top10_vehicle_ambiguous 上低于 latest | False  | 0.15097085038820904         | True         | False              |
| oracle_top3_val_test_same_direction_bad_ambiguous | 非部署 top3 上限在 val/test 歧义差样本上同向改善                               | False  | val=0.175004, test=0.048393 | False        | False              |
| test_bad_top10_any_feature_corr_gt_005            | test bad_top10 至少一个新特征集的生理距离-真实误差排序相关均值 > 0.05          | True   | 0.06203064366124616         | False        | False              |
| test_best_top1_diagnostic_beats_latest            | 即使 test-best 诊断，新生理 top1 至少有一个特征集低于 latest                   | False  | 0.0903037010054839          | False        | False              |

## validation 选择后的 test 泛化

| event_group                 | method          | deployable   | val_chosen_feature_set        |   val_delta_vs_latest_mean |   test_delta_vs_latest_mean |   test_corr_mean | test_passes_latest   | val_and_test_same_direction_gain   |
|:----------------------------|:----------------|:-------------|:------------------------------|---------------------------:|----------------------------:|-----------------:|:---------------------|:-----------------------------------|
| all                         | bio_top1        | True         | ecg_category_quality_top48    |                 0.116043   |                   0.0598198 |      0.00350309  | False                | False                              |
| all                         | bio_top3_oracle | False        | ecg_window_dur2_end0_top24    |                -0.00271998 |                  -0.0105243 |      0.000130505 | True                 | True                               |
| all                         | bio_top5_oracle | False        | ecg_duration_dur1_top32       |                -0.0435468  |                  -0.0288172 |     -0.0176127   | True                 | True                               |
| vehicle_ambiguous           | bio_top1        | True         | ecg_window_dur1_end0_top24    |                 0.149281   |                   0.0767018 |     -0.0049848   | False                | False                              |
| vehicle_ambiguous           | bio_top3_oracle | False        | ecg_window_dur2_end0_top24    |                 0.0193195  |                  -0.0119792 |     -0.00249147  | True                 | False                              |
| vehicle_ambiguous           | bio_top5_oracle | False        | ecg_window_dur1_end0_top24    |                -0.0299514  |                  -0.0221543 |     -0.0049848   | True                 | True                               |
| bad_top10                   | bio_top1        | True         | ecg_window_dur1_end0_top24    |                 0.45105    |                   0.155603  |      0.0220224   | False                | False                              |
| bad_top10                   | bio_top3_oracle | False        | ecg_window_dur1_end0_top24    |                 0.158724   |                   0.0332531 |      0.0220224   | False                | False                              |
| bad_top10                   | bio_top5_oracle | False        | ecg_duration_pre10_pre5_top32 |                 0.0453102  |                   0.015074  |     -0.0158539   | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top1        | True         | ecg_window_dur1_end0_top24    |                 0.501792   |                   0.150971  |      0.0537887   | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top3_oracle | False        | ecg_window_dur1_end0_top24    |                 0.175004   |                   0.0483935 |      0.0537887   | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top5_oracle | False        | ecg_duration_pre10_pre5_top32 |                 0.0505596  |                   0.0153402 |     -0.00847061  | False                | False                              |

## test bad_top10 的最佳诊断结果

| feature_set                      |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:---------------------------------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| ecg_category_morph_dynamic_top48 |  19 |           0.695048 |             0.785352 |                    0.0903037 |                    0.708932 |                   0.0138834  |      0.0601455  |
| ecg_duration_pre10_pre5_top32    |  19 |           0.695048 |             0.79289  |                    0.0978419 |                    0.73474  |                   0.0396918  |     -0.0158539  |
| ecg_offset_endm5_top32           |  19 |           0.695048 |             0.79289  |                    0.0978419 |                    0.73474  |                   0.0396918  |     -0.0158539  |
| ecg_duration_dur1_top32          |  19 |           0.695048 |             0.800207 |                    0.105159  |                    0.677102 |                  -0.0179468  |      0.0620306  |
| ecg_window_dur1_endm1_top24      |  19 |           0.695048 |             0.816257 |                    0.121208  |                    0.699896 |                   0.00484718 |      0.0553891  |
| ecg_offset_end0_top32            |  19 |           0.695048 |             0.816836 |                    0.121787  |                    0.711501 |                   0.016453   |      0.0492619  |
| ecg_window_dur2_endm1_top24      |  19 |           0.695048 |             0.822343 |                    0.127295  |                    0.723425 |                   0.0283766  |      0.00966661 |
| ecg_offset_endm1_top32           |  19 |           0.695048 |             0.82785  |                    0.132802  |                    0.687694 |                  -0.00735407 |      0.0527647  |

## test bad_top10 排序相关最高的 ECG 特征集

| feature_set                      |   n |   bio_top1_minus_latest_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_corr_positive_rate |
|:---------------------------------|----:|-----------------------------:|-----------------------------:|----------------:|-------------------------:|
| ecg_duration_dur1_top32          |  19 |                    0.105159  |                  -0.0179468  |       0.0620306 |                 0.631579 |
| ecg_category_morph_dynamic_top48 |  19 |                    0.0903037 |                   0.0138834  |       0.0601455 |                 0.684211 |
| ecg_category_morph_level_top48   |  19 |                    0.16551   |                   0.0259329  |       0.0560593 |                 0.736842 |
| ecg_window_dur1_endm1_top24      |  19 |                    0.121208  |                   0.00484718 |       0.0553891 |                 0.684211 |
| ecg_offset_endm2_top32           |  19 |                    0.206126  |                   0.0503928  |       0.0529665 |                 0.789474 |
| ecg_offset_endm1_top32           |  19 |                    0.132802  |                  -0.00735407 |       0.0527647 |                 0.684211 |
| ecg_offset_end0_top32            |  19 |                    0.121787  |                   0.016453   |       0.0492619 |                 0.736842 |
| ecg_duration_dur2_top32          |  19 |                    0.260166  |                   0.0408576  |       0.0437316 |                 0.578947 |

## 因果同步偏移组

| feature_set              | group_value   |   n |   bio_top1_minus_latest_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_corr_positive_rate |
|:-------------------------|:--------------|----:|-----------------------------:|-----------------------------:|----------------:|-------------------------:|
| ecg_offset_end0_top32    | end0          |  15 |                    0.127494  |                    0.0239258 |      0.0723626  |                 0.8      |
| ecg_offset_endm1_top32   | endm1         |  15 |                    0.0906392 |                   -0.0232423 |      0.0699103  |                 0.733333 |
| ecg_offset_endm2_top32   | endm2         |  15 |                    0.22514   |                    0.0622111 |      0.0685269  |                 0.8      |
| ecg_offset_endm0p5_top32 | endm0p5       |  15 |                    0.0910424 |                    0.0261483 |      0.0567447  |                 0.6      |
| ecg_offset_delta_top32   | delta         |  15 |                    0.180246  |                   -0.0120338 |      0.0154014  |                 0.533333 |
| ecg_offset_endm10_top32  | endm10        |  15 |                    0.245231  |                    0.0692356 |      0.00905749 |                 0.666667 |
| ecg_offset_endm5_top32   | endm5         |  15 |                    0.0854522 |                    0.0408882 |     -0.00847061 |                 0.666667 |

## bad_top10 分层

| feature_set                       | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:----------------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| ecg_category_morph_dynamic_top48  | test    |  19 |           0.695048 |             0.785352 |                    0.0903037 |                    0.708932 |                   0.0138834  |      0.0601455  |
| ecg_duration_pre10_pre5_top32     | test    |  19 |           0.695048 |             0.79289  |                    0.0978419 |                    0.73474  |                   0.0396918  |     -0.0158539  |
| ecg_offset_endm5_top32            | test    |  19 |           0.695048 |             0.79289  |                    0.0978419 |                    0.73474  |                   0.0396918  |     -0.0158539  |
| ecg_duration_dur1_top32           | test    |  19 |           0.695048 |             0.800207 |                    0.105159  |                    0.677102 |                  -0.0179468  |      0.0620306  |
| ecg_window_dur1_endm1_top24       | test    |  19 |           0.695048 |             0.816257 |                    0.121208  |                    0.699896 |                   0.00484718 |      0.0553891  |
| ecg_offset_end0_top32             | test    |  19 |           0.695048 |             0.816836 |                    0.121787  |                    0.711501 |                   0.016453   |      0.0492619  |
| ecg_window_dur2_endm1_top24       | test    |  19 |           0.695048 |             0.822343 |                    0.127295  |                    0.723425 |                   0.0283766  |      0.00966661 |
| ecg_offset_endm1_top32            | test    |  19 |           0.695048 |             0.82785  |                    0.132802  |                    0.687694 |                  -0.00735407 |      0.0527647  |
| ecg_window_dur1_endm0p5_top24     | test    |  19 |           0.695048 |             0.827973 |                    0.132924  |                    0.714708 |                   0.0196591  |      0.0138579  |
| ecg_window_dur2_endm0p5_top24     | test    |  19 |           0.695048 |             0.829589 |                    0.13454   |                    0.743563 |                   0.0485142  |      0.0376034  |
| ecg_category_rr_peak_top48        | test    |  19 |           0.695048 |             0.83254  |                    0.137492  |                    0.7471   |                   0.0520512  |      0.013068   |
| ecg_offset_endm0p5_top32          | test    |  19 |           0.695048 |             0.834162 |                    0.139114  |                    0.717331 |                   0.0222822  |      0.0328806  |
| ecg_window_dur1_end0_top24        | test    |  19 |           0.695048 |             0.850651 |                    0.155603  |                    0.728302 |                   0.0332531  |      0.0220224  |
| ecg_duration_delta_top32          | test    |  19 |           0.695048 |             0.85126  |                    0.156211  |                    0.679089 |                  -0.0159595  |      0.0189255  |
| ecg_offset_delta_top32            | test    |  19 |           0.695048 |             0.85126  |                    0.156211  |                    0.679089 |                  -0.0159595  |      0.0189255  |
| ecg_category_morph_level_top48    | test    |  19 |           0.695048 |             0.860559 |                    0.16551   |                    0.720981 |                   0.0259329  |      0.0560593  |
| ecg_category_temporal_delta_top48 | test    |  19 |           0.695048 |             0.869806 |                    0.174757  |                    0.701278 |                   0.0062297  |      0.0308642  |
| ecg_duration_dur5_top32           | test    |  19 |           0.695048 |             0.893404 |                    0.198356  |                    0.740707 |                   0.0456586  |      0.00710455 |
| ecg_duration_dur3_top32           | test    |  19 |           0.695048 |             0.8961   |                    0.201051  |                    0.726145 |                   0.031097   |      0.0324318  |
| ecg_offset_endm2_top32            | test    |  19 |           0.695048 |             0.901175 |                    0.206126  |                    0.745441 |                   0.0503928  |      0.0529665  |
| ecg_category_quality_top48        | test    |  19 |           0.695048 |             0.905853 |                    0.210804  |                    0.750311 |                   0.0552628  |     -0.0515111  |
| ecg_window_dur2_end0_top24        | test    |  19 |           0.695048 |             0.912981 |                    0.217933  |                    0.712813 |                   0.0177641  |      0.0179616  |
| ecg_all_top64                     | test    |  19 |           0.695048 |             0.913547 |                    0.218498  |                    0.733951 |                   0.0389024  |      0.0232393  |
| ecg_duration_pre20_pre10_top32    | test    |  19 |           0.695048 |             0.925563 |                    0.230515  |                    0.772849 |                   0.0778008  |     -0.011311   |
| ecg_offset_endm10_top32           | test    |  19 |           0.695048 |             0.925563 |                    0.230515  |                    0.772849 |                   0.0778008  |     -0.011311   |
| ecg_duration_dur2_top32           | test    |  19 |           0.695048 |             0.955215 |                    0.260166  |                    0.735906 |                   0.0408576  |      0.0437316  |
| ecg_low_identity_top48            | test    |  19 |           0.695048 |             0.970257 |                    0.275208  |                    0.750458 |                   0.0554101  |      0.0177268  |
| ecg_window_dur1_end0_top24        | val     |  31 |           1.07279  |             1.52384  |                    0.45105   |                    1.23151  |                   0.158724   |      0.0163635  |
| ecg_category_temporal_delta_top48 | val     |  31 |           1.07279  |             1.55266  |                    0.479868  |                    1.23983  |                   0.167046   |      0.0550115  |
| ecg_low_identity_top48            | val     |  31 |           1.07279  |             1.6121   |                    0.539316  |                    1.23857  |                   0.165782   |      0.0523896  |
| ecg_category_quality_top48        | val     |  31 |           1.07279  |             1.66859  |                    0.595801  |                    1.29479  |                   0.222001   |     -0.0188873  |
| ecg_duration_dur3_top32           | val     |  31 |           1.07279  |             1.72336  |                    0.650576  |                    1.4716   |                   0.398816   |     -0.00421636 |
| ecg_window_dur2_end0_top24        | val     |  31 |           1.07279  |             1.73485  |                    0.662061  |                    1.24616  |                   0.173371   |      0.0047754  |
| ecg_offset_endm0p5_top32          | val     |  31 |           1.07279  |             1.7397   |                    0.66691   |                    1.32419  |                   0.251398   |     -0.00577369 |
| ecg_duration_dur1_top32           | val     |  31 |           1.07279  |             1.74086  |                    0.668077  |                    1.3148   |                   0.242016   |     -0.0122932  |
| ecg_window_dur1_endm1_top24       | val     |  31 |           1.07279  |             1.74143  |                    0.668638  |                    1.30926  |                   0.236473   |     -0.0125827  |
| ecg_window_dur1_endm0p5_top24     | val     |  31 |           1.07279  |             1.74766  |                    0.674875  |                    1.25186  |                   0.179075   |      0.0112228  |
| ecg_duration_pre10_pre5_top32     | val     |  31 |           1.07279  |             1.74948  |                    0.676697  |                    1.28854  |                   0.215757   |      0.0129519  |
| ecg_offset_endm5_top32            | val     |  31 |           1.07279  |             1.74948  |                    0.676697  |                    1.28854  |                   0.215757   |      0.0129519  |
| ecg_duration_delta_top32          | val     |  31 |           1.07279  |             1.75421  |                    0.681425  |                    1.29518  |                   0.222396   |      0.0589966  |

## bad_top10_vehicle_ambiguous 分层

| feature_set                       | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |
|:----------------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|
| ecg_duration_dur1_top32           | test    |  15 |           0.744423 |             0.790156 |                    0.0457331 |                    0.715738 |                  -0.0286852  |     0.10107     |
| ecg_window_dur2_endm1_top24       | test    |  15 |           0.744423 |             0.819487 |                    0.0750644 |                    0.749042 |                   0.00461875 |     0.0131532   |
| ecg_duration_pre10_pre5_top32     | test    |  15 |           0.744423 |             0.829875 |                    0.0854522 |                    0.785311 |                   0.0408882  |    -0.00847061  |
| ecg_offset_endm5_top32            | test    |  15 |           0.744423 |             0.829875 |                    0.0854522 |                    0.785311 |                   0.0408882  |    -0.00847061  |
| ecg_offset_endm1_top32            | test    |  15 |           0.744423 |             0.835062 |                    0.0906392 |                    0.721181 |                  -0.0232423  |     0.0699103   |
| ecg_offset_endm0p5_top32          | test    |  15 |           0.744423 |             0.835465 |                    0.0910424 |                    0.770571 |                   0.0261483  |     0.0567447   |
| ecg_category_rr_peak_top48        | test    |  15 |           0.744423 |             0.836822 |                    0.0923987 |                    0.79229  |                   0.0478673  |     0.0534749   |
| ecg_category_morph_dynamic_top48  | test    |  15 |           0.744423 |             0.837406 |                    0.0929834 |                    0.760242 |                   0.0158191  |     0.0581238   |
| ecg_window_dur1_endm1_top24       | test    |  15 |           0.744423 |             0.850153 |                    0.10573   |                    0.740536 |                  -0.00388646 |     0.0667382   |
| ecg_offset_end0_top32             | test    |  15 |           0.744423 |             0.871917 |                    0.127494  |                    0.768349 |                   0.0239258  |     0.0723626   |
| ecg_window_dur2_endm0p5_top24     | test    |  15 |           0.744423 |             0.872176 |                    0.127753  |                    0.794282 |                   0.0498587  |     0.0603402   |
| ecg_window_dur1_endm0p5_top24     | test    |  15 |           0.744423 |             0.879192 |                    0.134769  |                    0.769576 |                   0.0251527  |     0.0424298   |
| ecg_duration_dur3_top32           | test    |  15 |           0.744423 |             0.89478  |                    0.150357  |                    0.778573 |                   0.0341499  |     0.0417535   |
| ecg_window_dur1_end0_top24        | test    |  15 |           0.744423 |             0.895394 |                    0.150971  |                    0.792816 |                   0.0483935  |     0.0537887   |
| ecg_category_morph_level_top48    | test    |  15 |           0.744423 |             0.916695 |                    0.172272  |                    0.773446 |                   0.0290235  |     0.0728299   |
| ecg_duration_dur5_top32           | test    |  15 |           0.744423 |             0.924095 |                    0.179672  |                    0.771741 |                   0.0273178  |     0.0318441   |
| ecg_duration_delta_top32          | test    |  15 |           0.744423 |             0.924669 |                    0.180246  |                    0.732389 |                  -0.0120338  |     0.0154014   |
| ecg_offset_delta_top32            | test    |  15 |           0.744423 |             0.924669 |                    0.180246  |                    0.732389 |                  -0.0120338  |     0.0154014   |
| ecg_category_quality_top48        | test    |  15 |           0.744423 |             0.932388 |                    0.187965  |                    0.767594 |                   0.0231706  |    -0.0672295   |
| ecg_window_dur2_end0_top24        | test    |  15 |           0.744423 |             0.954883 |                    0.21046   |                    0.753758 |                   0.00933498 |     0.0449423   |
| ecg_offset_endm2_top32            | test    |  15 |           0.744423 |             0.969563 |                    0.22514   |                    0.806634 |                   0.0622111  |     0.0685269   |
| ecg_category_temporal_delta_top48 | test    |  15 |           0.744423 |             0.973964 |                    0.229541  |                    0.760495 |                   0.0160726  |     0.0268693   |
| ecg_duration_pre20_pre10_top32    | test    |  15 |           0.744423 |             0.989654 |                    0.245231  |                    0.813658 |                   0.0692356  |     0.00905749  |
| ecg_offset_endm10_top32           | test    |  15 |           0.744423 |             0.989654 |                    0.245231  |                    0.813658 |                   0.0692356  |     0.00905749  |
| ecg_all_top64                     | test    |  15 |           0.744423 |             1.01506  |                    0.27064   |                    0.799972 |                   0.0555492  |     0.0350894   |
| ecg_duration_dur2_top32           | test    |  15 |           0.744423 |             1.03814  |                    0.293716  |                    0.783847 |                   0.0394244  |     0.0549293   |
| ecg_low_identity_top48            | test    |  15 |           0.744423 |             1.07886  |                    0.334433  |                    0.822175 |                   0.0777517  |     0.021922    |
| ecg_window_dur1_end0_top24        | val     |  27 |           1.02949  |             1.53129  |                    0.501792  |                    1.2045   |                   0.175004   |     0.00943605  |
| ecg_category_temporal_delta_top48 | val     |  27 |           1.02949  |             1.56184  |                    0.532348  |                    1.21116  |                   0.181663   |     0.053653    |
| ecg_low_identity_top48            | val     |  27 |           1.02949  |             1.58273  |                    0.553237  |                    1.21754  |                   0.188048   |     0.0508316   |
| ecg_category_quality_top48        | val     |  27 |           1.02949  |             1.70315  |                    0.673654  |                    1.2807   |                   0.251205   |    -0.0216018   |
| ecg_window_dur2_end0_top24        | val     |  27 |           1.02949  |             1.75833  |                    0.728835  |                    1.22359  |                   0.194099   |     0.000585852 |
| ecg_duration_dur3_top32           | val     |  27 |           1.02949  |             1.764    |                    0.734506  |                    1.48606  |                   0.456568   |    -0.00810254  |
| ecg_duration_dur1_top32           | val     |  27 |           1.02949  |             1.77414  |                    0.744647  |                    1.30686  |                   0.27737    |    -0.0120169   |
| ecg_duration_delta_top32          | val     |  27 |           1.02949  |             1.77452  |                    0.745027  |                    1.28039  |                   0.250895   |     0.0572084   |
| ecg_offset_delta_top32            | val     |  27 |           1.02949  |             1.77452  |                    0.745027  |                    1.28039  |                   0.250895   |     0.0572084   |
| ecg_duration_pre10_pre5_top32     | val     |  27 |           1.02949  |             1.7751   |                    0.745605  |                    1.26655  |                   0.237055   |     0.0154769   |
| ecg_offset_endm5_top32            | val     |  27 |           1.02949  |             1.7751   |                    0.745605  |                    1.26655  |                   0.237055   |     0.0154769   |
| ecg_offset_endm0p5_top32          | val     |  27 |           1.02949  |             1.78464  |                    0.755149  |                    1.3098   |                   0.280304   |    -0.00249847  |
| ecg_window_dur1_endm1_top24       | val     |  27 |           1.02949  |             1.78471  |                    0.755219  |                    1.29323  |                   0.263732   |    -0.0146516   |

## feature set 审计

| feature_set                       | group_type   | group_value    |   candidate_feature_n |   feature_n |   behavior_eta_max |   bad_eta_max |   identity_eta_median |
|:----------------------------------|:-------------|:---------------|----------------------:|------------:|-------------------:|--------------:|----------------------:|
| ecg_all_top64                     | all          | all            |                   477 |          64 |          0.0526711 |    0.0371144  |             0.356579  |
| ecg_low_identity_top48            | identity     | low_identity   |                   128 |          48 |          0.0353967 |    0.0325284  |             0.0683974 |
| ecg_category_rr_peak_top48        | category     | rr_peak        |                   110 |          48 |          0.0515114 |    0.0163235  |             0.506542  |
| ecg_category_morph_dynamic_top48  | category     | morph_dynamic  |                   112 |          48 |          0.0342905 |    0.0318837  |             0.338595  |
| ecg_category_morph_level_top48    | category     | morph_level    |                    95 |          48 |          0.0526711 |    0.020943   |             0.661     |
| ecg_category_quality_top48        | category     | quality        |                    72 |          48 |          0.0353701 |    0.00861137 |             0.0631594 |
| ecg_category_temporal_delta_top48 | category     | temporal_delta |                    88 |          48 |          0.0371144 |    0.0371144  |             0.0925886 |
| ecg_offset_end0_top32             | offset       | end0           |                   110 |          32 |          0.0526711 |    0.0318837  |             0.502662  |
| ecg_offset_endm0p5_top32          | offset       | endm0p5        |                    79 |          32 |          0.0452044 |    0.0172989  |             0.495965  |
| ecg_offset_endm1_top32            | offset       | endm1          |                    80 |          32 |          0.0444046 |    0.0180507  |             0.464464  |
| ecg_offset_endm2_top32            | offset       | endm2          |                    58 |          32 |          0.0461073 |    0.0163235  |             0.506317  |
| ecg_offset_endm5_top32            | offset       | endm5          |                    30 |          30 |          0.0444543 |    0.0124229  |             0.490039  |
| ecg_offset_endm10_top32           | offset       | endm10         |                    30 |          30 |          0.0515114 |    0.011338   |             0.536844  |
| ecg_offset_delta_top32            | offset       | delta          |                    88 |          32 |          0.0371144 |    0.0371144  |             0.0925886 |
| ecg_duration_dur1_top32           | duration     | dur1           |                    66 |          32 |          0.0526711 |    0.0277859  |             0.400913  |
| ecg_duration_dur2_top32           | duration     | dur2           |                   112 |          32 |          0.0492809 |    0.0318837  |             0.541278  |
| ecg_duration_dur3_top32           | duration     | dur3           |                    89 |          32 |          0.04712   |    0.0203183  |             0.553814  |
| ecg_duration_dur5_top32           | duration     | dur5           |                    60 |          32 |          0.0495015 |    0.0163235  |             0.491058  |
| ecg_duration_pre10_pre5_top32     | duration     | pre10_pre5     |                    30 |          30 |          0.0444543 |    0.0124229  |             0.490039  |
| ecg_duration_pre20_pre10_top32    | duration     | pre20_pre10    |                    30 |          30 |          0.0515114 |    0.011338   |             0.536844  |
| ecg_duration_delta_top32          | duration     | delta          |                    88 |          32 |          0.0371144 |    0.0371144  |             0.0925886 |
| ecg_window_dur1_end0_top24        | window       | dur1_end0      |                    22 |          22 |          0.0526711 |    0.0277859  |             0.375734  |
| ecg_window_dur2_end0_top24        | window       | dur2_end0      |                    28 |          24 |          0.0492809 |    0.0318837  |             0.525619  |
| ecg_window_dur1_endm0p5_top24     | window       | dur1_endm0p5   |                    22 |          22 |          0.0443879 |    0.0172989  |             0.430845  |
| ecg_window_dur2_endm0p5_top24     | window       | dur2_endm0p5   |                    28 |          24 |          0.0452044 |    0.0114958  |             0.562918  |
| ecg_window_dur1_endm1_top24       | window       | dur1_endm1     |                    22 |          22 |          0.0435147 |    0.0180507  |             0.39425   |
| ecg_window_dur2_endm1_top24       | window       | dur2_endm1     |                    28 |          24 |          0.0444046 |    0.0120442  |             0.550741  |

## train-only ECG feature screen 摘要

| feature_category   | offset_group   | duration_group   |   feature_n |   behavior_eta_max |   bad_eta_max |   identity_eta_median |   behavior_identity_score_max |
|:-------------------|:---------------|:-----------------|------------:|-------------------:|--------------:|----------------------:|------------------------------:|
| temporal_delta     | delta          | delta            |          96 |          0.0801845 |    0.0801845  |             0.0962447 |                     0.585108  |
| rr_peak            | endm0p5        | dur1             |          10 |          0.243467  |    0.0474831  |             0.383225  |                     0.526871  |
| rr_peak            | end0           | dur1             |          10 |          0.217118  |    0.0374578  |             0.351581  |                     0.397251  |
| rr_peak            | endm1          | dur1             |          10 |          0.18809   |    0.0151897  |             0.354388  |                     0.391626  |
| rr_peak            | end0           | dur5             |          10 |          0.0495015 |    0.0146414  |             0.556859  |                     0.191598  |
| rr_peak            | endm5          | pre10_pre5       |          10 |          0.0415404 |    0.0124229  |             0.520348  |                     0.177298  |
| rr_peak            | endm2          | dur2             |          10 |          0.051346  |    0.02052    |             0.432449  |                     0.15147   |
| morph_dynamic      | endm5          | pre10_pre5       |           8 |          0.0271175 |    0.0104264  |             0.241656  |                     0.124642  |
| quality            | end0           | dur1             |           5 |          0.0102886 |    0.0022964  |             0.0539762 |                     0.117503  |
| rr_peak            | endm0p5        | dur2             |          10 |          0.0411903 |    0.0087027  |             0.474739  |                     0.116191  |
| quality            | endm1          | dur1             |           5 |          0.0353701 |    0.00791796 |             0.0611911 |                     0.112876  |
| quality            | endm0p5        | dur1             |           5 |          0.0281538 |    0.00377022 |             0.0611911 |                     0.112876  |
| quality            | end0           | dur2             |           5 |          0.016603  |    0.00486609 |             0.0796189 |                     0.109973  |
| rr_peak            | endm1          | dur2             |          10 |          0.0444046 |    0.0146092  |             0.451177  |                     0.104292  |
| morph_dynamic      | end0           | dur1             |           8 |          0.0277859 |    0.0277859  |             0.323194  |                     0.103139  |
| morph_dynamic      | endm0p5        | dur1             |           8 |          0.02681   |    0.0115412  |             0.25781   |                     0.100108  |
| rr_peak            | endm10         | pre20_pre10      |          10 |          0.0515114 |    0.00944832 |             0.631467  |                     0.099462  |
| rr_peak            | endm1          | dur3             |          10 |          0.044329  |    0.0130541  |             0.474146  |                     0.0988443 |
| quality            | endm10         | pre20_pre10      |           5 |          0.0182111 |    0.00517125 |             0.077055  |                     0.0978338 |
| rr_peak            | endm2          | dur5             |          10 |          0.0461073 |    0.0163235  |             0.57422   |                     0.0953448 |
| morph_dynamic      | endm1          | dur1             |           8 |          0.0281109 |    0.0150728  |             0.295711  |                     0.0930055 |
| rr_peak            | end0           | dur3             |          10 |          0.0433786 |    0.0120458  |             0.440941  |                     0.0889061 |
| morph_level        | end0           | dur1             |           7 |          0.0526711 |    0.020943   |             0.624558  |                     0.0812357 |
| quality            | end0           | dur5             |           5 |          0.0173788 |    0.00299989 |             0.0777019 |                     0.0798659 |
| morph_dynamic      | endm1          | dur2             |           8 |          0.019983  |    0.0120442  |             0.372183  |                     0.0779941 |
| quality            | endm2          | dur2             |           5 |          0.0207613 |    0.00328776 |             0.0456925 |                     0.0767124 |
| morph_level        | endm0p5        | dur1             |           7 |          0.0443879 |    0.0172989  |             0.626957  |                     0.0759517 |
| morph_level        | end0           | dur2             |           7 |          0.0492809 |    0.0207634  |             0.659231  |                     0.0736381 |
| quality            | endm0p5        | dur3             |           5 |          0.0257974 |    0.00342997 |             0.0696047 |                     0.0713762 |
| quality            | end0           | dur3             |           5 |          0.0160806 |    0.00240488 |             0.0696047 |                     0.0713762 |
| quality            | endm0p5        | dur2             |           5 |          0.0299478 |    0.00382225 |             0.0696047 |                     0.0713762 |
| morph_level        | endm10         | pre20_pre10      |           7 |          0.0461086 |    0.00671058 |             0.683672  |                     0.0710841 |
| morph_level        | endm0p5        | dur3             |           7 |          0.0429466 |    0.00989148 |             0.699035  |                     0.070431  |
| rr_peak            | endm0p5        | dur3             |          10 |          0.04323   |    0.0106643  |             0.460739  |                     0.0699503 |
| morph_level        | end0           | dur3             |           7 |          0.04712   |    0.016698   |             0.69506   |                     0.0696486 |
| morph_level        | end0           | dur5             |           7 |          0.0456948 |    0.0132327  |             0.683785  |                     0.0680996 |
| morph_level        | endm5          | pre10_pre5       |           7 |          0.0444543 |    0.00670337 |             0.666314  |                     0.0662895 |
| morph_level        | endm1          | dur1             |           7 |          0.0435147 |    0.0180507  |             0.634027  |                     0.0659792 |
| morph_level        | endm0p5        | dur2             |           7 |          0.0452044 |    0.0114958  |             0.655577  |                     0.0651504 |
| morph_dynamic      | end0           | dur2             |           8 |          0.0318837 |    0.0318837  |             0.352849  |                     0.0649636 |
| quality            | endm1          | dur2             |           5 |          0.0277038 |    0.00245554 |             0.0534748 |                     0.0640251 |
| morph_level        | endm1          | dur3             |           7 |          0.0424572 |    0.00893304 |             0.69249   |                     0.0638173 |
| morph_level        | endm1          | dur2             |           7 |          0.0423934 |    0.0118644  |             0.656733  |                     0.0635838 |
| morph_dynamic      | endm10         | pre20_pre10      |           8 |          0.0342905 |    0.011338   |             0.315825  |                     0.0609512 |
| morph_level        | endm2          | dur5             |           7 |          0.0429358 |    0.00871927 |             0.655323  |                     0.0603457 |
| morph_dynamic      | endm0p5        | dur2             |           8 |          0.0220289 |    0.0104479  |             0.37382   |                     0.0600932 |
| morph_dynamic      | end0           | dur3             |           8 |          0.0316432 |    0.0203183  |             0.408522  |                     0.0599903 |
| rr_peak            | end0           | dur2             |          10 |          0.0328804 |    0.0116657  |             0.368551  |                     0.0599507 |
| morph_level        | endm2          | dur2             |           7 |          0.0384881 |    0.0062774  |             0.650465  |                     0.0582742 |
| morph_dynamic      | endm0p5        | dur3             |           8 |          0.0159552 |    0.0091116  |             0.357886  |                     0.0549786 |

## ECG 质量摘要

| subject   | recording                            | split   |   event_n |   ok_rate |   bio288_baseline_valid_ratio_median |   bio288_w_dur2_end0_ecg_valid_ratio_median |   bio288_w_dur2_end0_ecg_peak_n_median |   bio288_w_dur2_end0_ecg_rr_plausible_rate_median |
|:----------|:-------------------------------------|:--------|----------:|----------:|-------------------------------------:|--------------------------------------------:|---------------------------------------:|--------------------------------------------------:|
| cwh       | Entity_Recording_2025_09_26_19_35_47 | test    |        10 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| cwh       | Entity_Recording_2025_09_26_19_45_40 | test    |        11 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| cwh       | Entity_Recording_2025_09_26_19_56_16 | test    |        11 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| cwh       | Entity_Recording_2025_09_26_20_06_19 | test    |        14 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| lx        | Entity_Recording_2025_09_26_08_58_43 | test    |         2 |         1 |                                    1 |                                           1 |                                    3.5 |                                          1        |
| lx        | Entity_Recording_2025_09_26_09_17_22 | test    |        11 |         1 |                                    1 |                                           1 |                                    4   |                                          0.666667 |
| rjy       | Entity_Recording_2025_09_28_19_33_26 | test    |        17 |         0 |                                  nan |                                         nan |                                  nan   |                                        nan        |
| rjy       | Entity_Recording_2025_09_28_19_44_42 | test    |         2 |         0 |                                  nan |                                         nan |                                  nan   |                                        nan        |
| rjy       | Entity_Recording_2025_09_28_19_51_44 | test    |        19 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| rjy       | Entity_Recording_2025_09_28_20_02_20 | test    |        25 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| rjy       | Entity_Recording_2025_09_28_20_15_42 | test    |        19 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| tyy       | Entity_Recording_2025_09_28_14_23_43 | test    |        24 |         1 |                                    1 |                                           1 |                                    5   |                                          0.6      |
| tyy       | Entity_Recording_2025_09_28_14_40_01 | test    |         7 |         1 |                                    1 |                                           1 |                                    5   |                                          0.8      |
| tyy       | Entity_Recording_2025_09_28_14_57_17 | test    |        12 |         1 |                                    1 |                                           1 |                                    5.5 |                                          0.8      |
| byx       | Entity_Recording_2025_09_28_17_05_51 | train   |        19 |         1 |                                    1 |                                           1 |                                    2   |                                          1        |
| byx       | Entity_Recording_2025_09_28_17_15_52 | train   |        18 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| byx       | Entity_Recording_2025_09_28_17_25_18 | train   |        23 |         1 |                                    1 |                                           1 |                                    2   |                                          1        |
| byx       | Entity_Recording_2025_09_28_17_35_43 | train   |        25 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| byx       | Entity_Recording_2025_09_28_17_46_00 | train   |        17 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| gf        | Entity_Recording_2025_09_26_10_03_00 | train   |         1 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| gf        | Entity_Recording_2025_09_26_10_18_49 | train   |         9 |         1 |                                    1 |                                           1 |                                    5   |                                          0.75     |
| gf        | Entity_Recording_2025_09_26_10_30_12 | train   |         9 |         1 |                                    1 |                                           1 |                                    4   |                                          1        |
| gf        | Entity_Recording_2025_09_26_10_40_59 | train   |         9 |         1 |                                    1 |                                           1 |                                    4   |                                          1        |
| gf        | Entity_Recording_2025_09_26_10_52_57 | train   |         8 |         1 |                                    1 |                                           1 |                                    4   |                                          1        |
| hzh       | Entity_Recording_2025_09_26_20_50_27 | train   |        27 |         1 |                                    1 |                                           1 |                                    5   |                                          0.75     |
| hzh       | Entity_Recording_2025_09_26_21_03_19 | train   |        24 |         1 |                                    1 |                                           1 |                                    5   |                                          0.583333 |
| hzh       | Entity_Recording_2025_09_26_21_17_02 | train   |        10 |         1 |                                    1 |                                           1 |                                    5   |                                          0.583333 |
| hzh       | Entity_Recording_2025_09_27_19_22_27 | train   |        17 |         1 |                                    1 |                                           1 |                                    5   |                                          0.666667 |
| hzh       | Entity_Recording_2025_09_27_19_33_25 | train   |        19 |         1 |                                    1 |                                           1 |                                    5   |                                          0.666667 |
| hzh       | Entity_Recording_2025_09_27_19_44_05 | train   |        21 |         1 |                                    1 |                                           1 |                                    5   |                                          0.666667 |
| jy        | Entity_Recording_2025_09_26_17_17_11 | train   |         8 |         1 |                                    1 |                                           1 |                                    5   |                                          0.708333 |
| jy        | Entity_Recording_2025_09_26_17_29_44 | train   |        11 |         1 |                                    1 |                                           1 |                                    4   |                                          0.666667 |
| jy        | Entity_Recording_2025_09_26_17_40_51 | train   |        10 |         1 |                                    1 |                                           1 |                                    5   |                                          0.5      |
| jy        | Entity_Recording_2025_09_26_17_51_46 | train   |         3 |         1 |                                    1 |                                           1 |                                    5   |                                          0.666667 |
| jy        | Entity_Recording_2025_09_26_18_01_40 | train   |        10 |         1 |                                    1 |                                           1 |                                    4   |                                          0.708333 |
| xst       | Entity_Recording_2025_09_26_11_34_18 | train   |         6 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| yyl       | Entity_Recording_2025_09_28_09_14_23 | train   |        21 |         1 |                                    1 |                                           1 |                                    4   |                                          0.5      |
| yyl       | Entity_Recording_2025_09_28_09_29_01 | train   |        28 |         1 |                                    1 |                                           1 |                                    4   |                                          0.666667 |
| yyl       | Entity_Recording_2025_09_28_09_39_01 | train   |        20 |         1 |                                    1 |                                           1 |                                    4   |                                          0.583333 |
| yyl       | Entity_Recording_2025_09_28_09_49_11 | train   |        18 |         1 |                                    1 |                                           1 |                                    3   |                                          0.666667 |
| yzy       | Entity_Recording_2025_09_27_14_13_03 | train   |        23 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| yzy       | Entity_Recording_2025_09_27_14_26_04 | train   |        21 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| yzy       | Entity_Recording_2025_09_27_14_37_08 | train   |        21 |         1 |                                    1 |                                           1 |                                    2   |                                          1        |
| yzy       | Entity_Recording_2025_09_27_15_04_26 | train   |         1 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| yzy       | Entity_Recording_2025_09_27_15_07_57 | train   |        13 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| zt        | Entity_Recording_2025_09_28_11_20_08 | train   |        15 |         1 |                                    1 |                                           1 |                                    4   |                                          1        |
| zx        | Entity_Recording_2025_09_27_16_32_00 | train   |        24 |         0 |                                  nan |                                         nan |                                  nan   |                                        nan        |
| zx        | Entity_Recording_2025_09_27_16_46_13 | train   |        26 |         0 |                                  nan |                                         nan |                                  nan   |                                        nan        |
| zx        | Entity_Recording_2025_09_27_17_14_07 | train   |        21 |         0 |                                  nan |                                         nan |                                  nan   |                                        nan        |
| zx        | Entity_Recording_2025_09_27_17_25_16 | train   |         3 |         0 |                                  nan |                                         nan |                                  nan   |                                        nan        |
| zx        | Entity_Recording_2025_09_27_17_29_08 | train   |         1 |         0 |                                  nan |                                         nan |                                  nan   |                                        nan        |
| zx        | Entity_Recording_2025_09_27_17_45_11 | train   |        27 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| zx        | Entity_Recording_2025_09_27_17_56_42 | train   |         3 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| zx        | Entity_Recording_2025_09_27_18_07_01 | train   |        23 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| zx        | Entity_Recording_2025_09_27_18_17_48 | train   |        25 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| zxy       | Entity_Recording_2025_09_28_15_57_38 | train   |         4 |         1 |                                    1 |                                           1 |                                    3   |                                          1        |
| zxy       | Entity_Recording_2025_09_28_16_01_55 | train   |         1 |         1 |                                    1 |                                           1 |                                    5   |                                          0.75     |
| zxy       | Entity_Recording_2025_09_28_16_12_11 | train   |        12 |         1 |                                    1 |                                           1 |                                    4   |                                          1        |
| zxy       | Entity_Recording_2025_09_28_16_25_51 | train   |        12 |         1 |                                    1 |                                           1 |                                    4   |                                          1        |
| zxy       | Entity_Recording_2025_09_28_16_35_30 | train   |         7 |         1 |                                    1 |                                           1 |                                    4   |                                          1        |

## 图表

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702\figures\v288_badtop10_val_test_delta.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702\figures\v288_ecg_offset_group_summary.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v288_ecg_source_signal_route_audit_20260702\figures\v288_ecg_feature_screen_summary.png`

## 解释

- route gate 未通过：即使回到 ECG 源信号和 R 峰/RR 层，当前 ECG 仍没有形成可部署候选选择收益。
- 如果 test-best 仍只有弱排序相关，而 validation 选择后的 top1 不赢 latest，则不能把 ECG 解释为已解决差样本问题。
- 本轮使用的是因果同步偏移窗口；若这些窗口都不通过，后续不应继续靠同类 ECG 特征微调。

## guardrail

```json
{
  "pass": true,
  "zip_testzip": true,
  "event_n": 1167,
  "candidate_rows": 46680,
  "ecg_source_feature_n": 518,
  "feature_set_n": 27,
  "uses_post_observation_any": false,
  "ok_rate": 0.919451585261354,
  "baseline_valid_ratio_median": 1.0,
  "dur2_end0_valid_ratio_median": 1.0,
  "fixed_wait_latest_badtop10": 0.6950484153471495,
  "route_viable_now": false,
  "deployable_top1_badtop10_pass": false,
  "deployable_top1_bad_ambiguous_pass": false,
  "test_best_top1_diagnostic_pass": false,
  "best_test_badtop10_top1_delta": 0.0903037010054839,
  "best_test_badtop10_corr": 0.06203064366124616,
  "reused_v260_feature_table": false,
  "test_used_for_current_feature_selection": false,
  "ecg_direction_seeded_by_prior_v287_test_diagnostic": true,
  "v287_source_guardrail_pass": true,
  "v287_source_route_viable_now": false
}
```