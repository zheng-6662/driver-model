# v284 dynamic low-identity physiology route gate

## 本轮目的

- 承接 v283：不再沿旧 bio selector 微调，而是构造新的低身份、动态生理状态特征。
- 用 train-only 行为相关性和 subject/recording 身份惩罚筛特征。
- 在 v278 vehicle top40 候选池中重新计算生理距离排序，先过 route gate，再谈轨迹模型。

## route gate 判定

| check                                             | requirement                                                                    | pass   | evidence                    | deployable   | route_viable_now   |
|:--------------------------------------------------|:-------------------------------------------------------------------------------|:-------|:----------------------------|:-------------|:-------------------|
| deployable_top1_val_chosen_bad_top10              | validation 选出的新生理 top1 在 test bad_top10 上低于 latest                   | False  | 0.16970132369744145         | True         | False              |
| deployable_top1_val_chosen_bad_ambiguous          | validation 选出的新生理 top1 在 test bad_top10_vehicle_ambiguous 上低于 latest | False  | 0.190283058087031           | True         | False              |
| oracle_top3_val_test_same_direction_bad_ambiguous | 非部署 top3 上限在 val/test 歧义差样本上同向改善                               | False  | val=0.136679, test=0.065228 | False        | False              |
| test_bad_top10_any_feature_corr_gt_005            | test bad_top10 至少一个新特征集的生理距离-真实误差排序相关均值 > 0.05          | True   | 0.055257621745776           | False        | False              |
| test_best_top1_diagnostic_beats_latest            | 即使 test-best 诊断，新生理 top1 至少有一个特征集低于 latest                   | False  | 0.15248347034579826         | False        | False              |

## feature set 审计

| feature_set                 |   feature_n |
|:----------------------------|------------:|
| dyn_behavior_identity_top64 |          64 |
| dyn_bad_identity_top48      |          48 |
| dyn_noamp_multi_top48       |          48 |
| low_identity_dyn_top48      |          32 |
| strict_ratio_noamp_top32    |          32 |

## validation 选择后的 test 泛化

| event_group                 | method          | deployable   | val_chosen_feature_set      |   val_delta_vs_latest_mean |   test_delta_vs_latest_mean |   test_corr_mean | test_passes_latest   | val_and_test_same_direction_gain   |
|:----------------------------|:----------------|:-------------|:----------------------------|---------------------------:|----------------------------:|-----------------:|:---------------------|:-----------------------------------|
| all                         | bio_top1        | True         | low_identity_dyn_top48      |                 0.104295   |                 0.0862437   |        0.0146758 | False                | False                              |
| all                         | bio_top3_oracle | False        | low_identity_dyn_top48      |                -0.00318959 |                 0.000801983 |        0.0146758 | False                | False                              |
| all                         | bio_top5_oracle | False        | dyn_behavior_identity_top64 |                -0.0414782  |                -0.0228888   |        0.0199585 | True                 | True                               |
| vehicle_ambiguous           | bio_top1        | True         | low_identity_dyn_top48      |                 0.139576   |                 0.101054    |        0.0201838 | False                | False                              |
| vehicle_ambiguous           | bio_top3_oracle | False        | low_identity_dyn_top48      |                 0.0157617  |                 0.00632876  |        0.0201838 | False                | False                              |
| vehicle_ambiguous           | bio_top5_oracle | False        | dyn_behavior_identity_top64 |                -0.0290761  |                -0.0205368   |        0.0253774 | True                 | True                               |
| bad_top10                   | bio_top1        | True         | dyn_bad_identity_top48      |                 0.471146   |                 0.169701    |        0.0159745 | False                | False                              |
| bad_top10                   | bio_top3_oracle | False        | dyn_bad_identity_top48      |                 0.122857   |                 0.0638187   |        0.0159745 | False                | False                              |
| bad_top10                   | bio_top5_oracle | False        | dyn_behavior_identity_top64 |                 0.0384108  |                 0.00469925  |        0.0200449 | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top1        | True         | dyn_bad_identity_top48      |                 0.507098   |                 0.190283    |        0.0356529 | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top3_oracle | False        | dyn_bad_identity_top48      |                 0.136679   |                 0.0652281   |        0.0356529 | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top5_oracle | False        | dyn_behavior_identity_top64 |                 0.0436007  |                 0.00509686  |        0.0189003 | False                | False                              |

## bad_top10 分层

| feature_set                 | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_best_in_top3_rate |
|:----------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|------------------------:|
| dyn_noamp_multi_top48       | test    |  19 |           0.695048 |             0.847532 |                     0.152483 |                    0.732757 |                   0.0377089  |       0.0045424 |               0.105263  |
| dyn_behavior_identity_top64 | test    |  19 |           0.695048 |             0.851336 |                     0.156287 |                    0.746034 |                   0.0509858  |       0.0200449 |               0         |
| dyn_bad_identity_top48      | test    |  19 |           0.695048 |             0.86475  |                     0.169701 |                    0.758867 |                   0.0638187  |       0.0159745 |               0.0526316 |
| strict_ratio_noamp_top32    | test    |  19 |           0.695048 |             0.892355 |                     0.197307 |                    0.723955 |                   0.0289069  |      -0.010431  |               0.0526316 |
| low_identity_dyn_top48      | test    |  19 |           0.695048 |             0.932223 |                     0.237175 |                    0.703894 |                   0.00884523 |       0.0552576 |               0.157895  |
| dyn_bad_identity_top48      | val     |  31 |           1.07279  |             1.54393  |                     0.471146 |                    1.19564  |                   0.122857   |       0.0643539 |               0.129032  |
| low_identity_dyn_top48      | val     |  31 |           1.07279  |             1.5471   |                     0.474311 |                    1.28692  |                   0.214131   |       0.0412786 |               0.0322581 |
| dyn_behavior_identity_top64 | val     |  31 |           1.07279  |             1.65234  |                     0.579552 |                    1.24198  |                   0.169194   |       0.0731617 |               0.0645161 |
| strict_ratio_noamp_top32    | val     |  31 |           1.07279  |             1.67665  |                     0.603862 |                    1.33854  |                   0.265752   |       0.0333313 |               0.0645161 |
| dyn_noamp_multi_top48       | val     |  31 |           1.07279  |             1.69769  |                     0.624905 |                    1.27925  |                   0.206458   |       0.0487348 |               0.0645161 |

## bad_top10 + vehicle_ambiguous 分层

| feature_set                 | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_best_in_top3_rate |
|:----------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|------------------------:|
| dyn_noamp_multi_top48       | test    |  15 |           0.744423 |             0.922681 |                     0.178258 |                    0.790039 |                    0.0456164 |      -0.0095381 |               0.133333  |
| dyn_behavior_identity_top64 | test    |  15 |           0.744423 |             0.927499 |                     0.183076 |                    0.806857 |                    0.0624338 |       0.0189003 |               0         |
| dyn_bad_identity_top48      | test    |  15 |           0.744423 |             0.934706 |                     0.190283 |                    0.809651 |                    0.0652281 |       0.0356529 |               0.0666667 |
| strict_ratio_noamp_top32    | test    |  15 |           0.744423 |             0.983134 |                     0.238711 |                    0.783368 |                    0.0389449 |      -0.0218572 |               0.0666667 |
| low_identity_dyn_top48      | test    |  15 |           0.744423 |             0.984383 |                     0.23996  |                    0.753479 |                    0.0090557 |       0.0454788 |               0.2       |
| dyn_bad_identity_top48      | val     |  27 |           1.02949  |             1.53659  |                     0.507098 |                    1.16617  |                    0.136679  |       0.0597241 |               0.148148  |
| low_identity_dyn_top48      | val     |  27 |           1.02949  |             1.54393  |                     0.514433 |                    1.2702   |                    0.240704  |       0.0409329 |               0.037037  |
| dyn_behavior_identity_top64 | val     |  27 |           1.02949  |             1.66092  |                     0.631425 |                    1.21924  |                    0.189744  |       0.0692359 |               0.0740741 |
| strict_ratio_noamp_top32    | val     |  27 |           1.02949  |             1.68883  |                     0.659337 |                    1.33202  |                    0.302522  |       0.0325732 |               0.0740741 |
| dyn_noamp_multi_top48       | val     |  27 |           1.02949  |             1.71299  |                     0.683498 |                    1.26202  |                    0.232528  |       0.0484771 |               0.0740741 |

## train-only 特征筛选 top20

| feature                                              |   finite_rate_train |   behavior_eta_max |   bad_top10_eta |   identity_eta_max |   identity_to_behavior_ratio |   behavior_identity_score | is_dynamic   |
|:-----------------------------------------------------|--------------------:|-------------------:|----------------:|-------------------:|-----------------------------:|--------------------------:|:-------------|
| bio260_pre10_pre5_hr_z_range                         |            0.888724 |         0.0297173  |     0.000110501 |         0.0870562  |                      2.92948 |                  0.306187 | True         |
| bio260_pre10_pre5_emg_z_slope                        |            0.888724 |         0.0227142  |     5.28715e-05 |         0.0643974  |                      2.83512 |                  0.305309 | True         |
| bio260_pre10_pre5_hr_z_std                           |            0.888724 |         0.0306659  |     3.70092e-06 |         0.0927597  |                      3.02485 |                  0.298423 | True         |
| bio260_pre5_pre2_scr_burst_longest_s                 |            0.737389 |         0.0152794  |     0.000719208 |         0.0451858  |                      2.95731 |                  0.276871 | False        |
| bio260_pre5_0_resp_z_slope                           |            0.888724 |         0.0188651  |     0.0188651   |         0.0669948  |                      3.55125 |                  0.245018 | True         |
| bio260_pre5_pre2_scr_burst_rate                      |            0.737389 |         0.0116872  |     0.000944391 |         0.037997   |                      3.25115 |                  0.243499 | True         |
| bio260_pre5_0_scr_burst_rate                         |            0.737389 |         0.014004   |     0.000197459 |         0.0508096  |                      3.62821 |                  0.230293 | True         |
| bio260_delta_pre2_0_minus_pre5_pre2_hr_z_mean        |            0.888724 |         0.0158528  |     0.000103229 |         0.0597075  |                      3.76638 |                  0.227418 | True         |
| bio260_pre5_0_scr_burst_longest_s                    |            0.737389 |         0.0177076  |     4.14947e-05 |         0.0681449  |                      3.84834 |                  0.2266   | False        |
| bio260_pre10_pre5_scr_z_slope                        |            0.737389 |         0.00576345 |     2.76888e-07 |         0.0175264  |                      3.04095 |                  0.209379 | True         |
| bio260_pre10_pre5_scr_burst_rate                     |            0.737389 |         0.0151269  |     0.0151269   |         0.070209   |                      4.64134 |                  0.188593 | True         |
| bio260_pre5_pre2_resp_phase_sin_end                  |            0.422849 |         0.020097   |     8.02962e-05 |         0.0999325  |                      4.9725  |                  0.182812 | True         |
| bio260_pre5_0_scr_z_slope                            |            0.737389 |         0.00206038 |     9.89682e-06 |         0.00256096 |                      1.24295 |                  0.16403  | True         |
| bio260_pre20_pre10_hr_z_last_minus_first             |            0.888724 |         0.0142188  |     0.00127407  |         0.0773096  |                      5.43712 |                  0.162855 | True         |
| bio260_pre5_pre2_emg_burst_longest_s                 |            0.888724 |         0.0199707  |     0.000136717 |         0.117633   |                      5.89028 |                  0.15647  | False        |
| bio260_delta_pre2_0_minus_pre20_pre10_scr_z_pos_area |            0.737389 |         0.00819009 |     2.03241e-05 |         0.0486599  |                      5.94131 |                  0.13962  | True         |
| bio260_pre10_pre5_resp_z_slope                       |            0.888724 |         0.0127     |     0.0127      |         0.0813299  |                      6.40391 |                  0.139057 | True         |
| bio260_pre10_pre5_emg_burst_episode_count            |            0.888724 |         0.00965219 |     0.00281675  |         0.0601972  |                      6.23664 |                  0.137501 | True         |
| bio260_pre5_pre2_hr_z_last_minus_first               |            0.888724 |         0.0152536  |     0.00407016  |         0.102185   |                      6.69908 |                  0.135968 | True         |
| bio260_pre20_pre10_scr_z_slope                       |            0.737389 |         0.0122346  |     2.05687e-08 |         0.0805578  |                      6.58445 |                  0.135102 | True         |

## 关键判读

- route gate 未通过：即使重筛动态低身份 biomarker，当前生理状态仍未稳定弥补车辆锚点前信息不足。
- 这说明下一步若还继续生理 goal，需要更底层的信号重处理或明确转为 subject-aware 个体校准任务；不应直接训练更复杂融合模型。

## 关键图

- `figures\v284_badtop10_val_test_delta.png`
- `figures\v284_bad_ambiguous_corr.png`
- `figures\v284_feature_screen_identity_vs_behavior.png`

## guardrail

```json
{
  "pass": true,
  "zip_testzip": true,
  "event_n": 1167,
  "candidate_rows": 46680,
  "feature_set_n": 5,
  "all_candidate_feature_n": 203,
  "dynamic_candidate_feature_n": 148,
  "fixed_wait_latest_badtop10": 0.695048,
  "route_viable_now": false,
  "deployable_top1_badtop10_pass": false,
  "deployable_top1_bad_ambiguous_pass": false,
  "test_best_top1_diagnostic_pass": false,
  "v283_old_route_closed": true
}
```
