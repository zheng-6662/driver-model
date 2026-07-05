# v285 raw-200Hz signal-shape physiology route gate

## 本轮目的

- 承接 v284：不再复用 v260 biomarker 做筛选，而是直接从 cleaned 200Hz 连续信号重算事件前状态。
- 重点特征包括质量、短窗形态、导数/突变、节律/相位、跨信号耦合、个体内 causal past percentile。
- 仍然先在 v278 vehicle top40 候选池中过 route gate，未通过则不进入复杂融合轨迹模型。

## route gate 判定

| check                                             | requirement                                                                    | pass   | evidence                    | deployable   | route_viable_now   |
|:--------------------------------------------------|:-------------------------------------------------------------------------------|:-------|:----------------------------|:-------------|:-------------------|
| deployable_top1_val_chosen_bad_top10              | validation 选出的新生理 top1 在 test bad_top10 上低于 latest                   | False  | 0.19575335791236476         | True         | False              |
| deployable_top1_val_chosen_bad_ambiguous          | validation 选出的新生理 top1 在 test bad_top10_vehicle_ambiguous 上低于 latest | False  | 0.18260557651519777         | True         | False              |
| oracle_top3_val_test_same_direction_bad_ambiguous | 非部署 top3 上限在 val/test 歧义差样本上同向改善                               | False  | val=0.192130, test=0.021161 | False        | False              |
| test_bad_top10_any_feature_corr_gt_005            | test bad_top10 至少一个新特征集的生理距离-真实误差排序相关均值 > 0.05          | False  | 0.049829102344494856        | False        | False              |
| test_best_top1_diagnostic_beats_latest            | 即使 test-best 诊断，新生理 top1 至少有一个特征集低于 latest                   | False  | 0.157776603573247           | False        | False              |

## feature set 审计

| feature_set              |   feature_n |
|:-------------------------|------------:|
| raw_shape_behavior_top64 |          64 |
| raw_shape_bad_top64      |          64 |
| raw_low_identity_top64   |          64 |
| raw_quality_shape_top64  |          64 |
| raw_coupling_top48       |          24 |
| raw_causal_past_top48    |          39 |

## validation 选择后的 test 泛化

| event_group                 | method          | deployable   | val_chosen_feature_set   |   val_delta_vs_latest_mean |   test_delta_vs_latest_mean |   test_corr_mean | test_passes_latest   | val_and_test_same_direction_gain   |
|:----------------------------|:----------------|:-------------|:-------------------------|---------------------------:|----------------------------:|-----------------:|:---------------------|:-----------------------------------|
| all                         | bio_top1        | True         | raw_coupling_top48       |                 0.11271    |                  0.0657677  |      0.0284068   | False                | False                              |
| all                         | bio_top3_oracle | False        | raw_coupling_top48       |                -0.00444714 |                 -0.00855346 |      0.0284068   | True                 | True                               |
| all                         | bio_top5_oracle | False        | raw_coupling_top48       |                -0.0461459  |                 -0.0339662  |      0.0284068   | True                 | True                               |
| vehicle_ambiguous           | bio_top1        | True         | raw_coupling_top48       |                 0.153509   |                  0.067377   |      0.0389119   | False                | False                              |
| vehicle_ambiguous           | bio_top3_oracle | False        | raw_coupling_top48       |                 0.0168563  |                 -0.0100153  |      0.0389119   | True                 | False                              |
| vehicle_ambiguous           | bio_top5_oracle | False        | raw_coupling_top48       |                -0.0314257  |                 -0.0349871  |      0.0389119   | True                 | True                               |
| bad_top10                   | bio_top1        | True         | raw_shape_bad_top64      |                 0.445967   |                  0.195753   |     -0.000140311 | False                | False                              |
| bad_top10                   | bio_top3_oracle | False        | raw_shape_bad_top64      |                 0.168891   |                  0.0410847  |     -0.000140311 | False                | False                              |
| bad_top10                   | bio_top5_oracle | False        | raw_shape_bad_top64      |                 0.0231503  |                 -0.0154119  |     -0.000140311 | True                 | False                              |
| bad_top10_vehicle_ambiguous | bio_top1        | True         | raw_shape_bad_top64      |                 0.498317   |                  0.182606   |      0.0247136   | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top3_oracle | False        | raw_shape_bad_top64      |                 0.19213    |                  0.0211607  |      0.0247136   | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top5_oracle | False        | raw_shape_bad_top64      |                 0.0247973  |                 -0.0179065  |      0.0247136   | True                 | False                              |

## bad_top10 分层

| feature_set              | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_best_in_top3_rate |
|:-------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|------------------------:|
| raw_coupling_top48       | test    |  19 |           0.695048 |             0.852825 |                     0.157777 |                    0.748147 |                    0.0530985 |     0.0498291   |               0.0526316 |
| raw_quality_shape_top64  | test    |  19 |           0.695048 |             0.864441 |                     0.169393 |                    0.738122 |                    0.0430738 |     0.00796495  |               0         |
| raw_shape_behavior_top64 | test    |  19 |           0.695048 |             0.867737 |                     0.172688 |                    0.732028 |                    0.0369793 |     0.00562573  |               0         |
| raw_shape_bad_top64      | test    |  19 |           0.695048 |             0.890802 |                     0.195753 |                    0.736133 |                    0.0410847 |    -0.000140311 |               0.0526316 |
| raw_low_identity_top64   | test    |  19 |           0.695048 |             0.947417 |                     0.252368 |                    0.755061 |                    0.0600121 |     0.00581896  |               0         |
| raw_causal_past_top48    | test    |  19 |           0.695048 |             0.997027 |                     0.301978 |                    0.796409 |                    0.101361  |    -0.0617901   |               0.0526316 |
| raw_shape_bad_top64      | val     |  31 |           1.07279  |             1.51875  |                     0.445967 |                    1.24168  |                    0.168891  |     0.0233469   |               0.16129   |
| raw_shape_behavior_top64 | val     |  31 |           1.07279  |             1.52679  |                     0.454007 |                    1.26757  |                    0.194785  |     0.025437    |               0.0967742 |
| raw_low_identity_top64   | val     |  31 |           1.07279  |             1.53312  |                     0.460336 |                    1.26088  |                    0.188095  |     0.0715525   |               0.129032  |
| raw_quality_shape_top64  | val     |  31 |           1.07279  |             1.58787  |                     0.515078 |                    1.27338  |                    0.200588  |     0.0383777   |               0.0645161 |
| raw_coupling_top48       | val     |  31 |           1.07279  |             1.63331  |                     0.560524 |                    1.28203  |                    0.209245  |     0.0911935   |               0.0645161 |
| raw_causal_past_top48    | val     |  31 |           1.07279  |             1.73221  |                     0.659419 |                    1.30982  |                    0.237029  |     0.0625654   |               0.0645161 |

## bad_top10 + vehicle_ambiguous 分层

| feature_set              | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_best_in_top3_rate |
|:-------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|------------------------:|
| raw_coupling_top48       | test    |  15 |           0.744423 |             0.905449 |                     0.161026 |                    0.781713 |                    0.0372898 |       0.0526483 |               0.0666667 |
| raw_shape_behavior_top64 | test    |  15 |           0.744423 |             0.915889 |                     0.171466 |                    0.776937 |                    0.0325141 |       0.038867  |               0         |
| raw_shape_bad_top64      | test    |  15 |           0.744423 |             0.927028 |                     0.182606 |                    0.765584 |                    0.0211607 |       0.0247136 |               0.0666667 |
| raw_quality_shape_top64  | test    |  15 |           0.744423 |             0.92727  |                     0.182847 |                    0.784657 |                    0.0402339 |       0.0390683 |               0         |
| raw_low_identity_top64   | test    |  15 |           0.744423 |             0.953815 |                     0.209392 |                    0.799852 |                    0.0554293 |       0.0182463 |               0         |
| raw_causal_past_top48    | test    |  15 |           0.744423 |             1.08005  |                     0.335627 |                    0.870571 |                    0.126148  |      -0.0656062 |               0.0666667 |
| raw_shape_bad_top64      | val     |  27 |           1.02949  |             1.52781  |                     0.498317 |                    1.22162  |                    0.19213   |       0.0201371 |               0.185185  |
| raw_low_identity_top64   | val     |  27 |           1.02949  |             1.53436  |                     0.504862 |                    1.24399  |                    0.214497  |       0.0763069 |               0.111111  |
| raw_shape_behavior_top64 | val     |  27 |           1.02949  |             1.53528  |                     0.505782 |                    1.24261  |                    0.213116  |       0.0300592 |               0.0740741 |
| raw_quality_shape_top64  | val     |  27 |           1.02949  |             1.58618  |                     0.556688 |                    1.25833  |                    0.228841  |       0.0440144 |               0.037037  |
| raw_coupling_top48       | val     |  27 |           1.02949  |             1.65768  |                     0.628187 |                    1.26817  |                    0.238675  |       0.0975891 |               0.0740741 |
| raw_causal_past_top48    | val     |  27 |           1.02949  |             1.76717  |                     0.737678 |                    1.2907   |                    0.261204  |       0.0651264 |               0.0740741 |

## train-only 特征类型筛选摘要

| feature_category   | signal_family   |   feature_n |   finite_rate_train_median |   behavior_eta_max |   bad_eta_max |   identity_eta_median |   behavior_identity_score_max |
|:-------------------|:----------------|------------:|---------------------------:|-------------------:|--------------:|----------------------:|------------------------------:|
| shape_dynamic      | eda             |         108 |                   0.737389 |         0.0400816  |    0.0400816  |             0.0627362 |                     0.531335  |
| level_dynamic      | eda             |          56 |                   0.737389 |         0.0452955  |    0.0452955  |             0.0619992 |                     0.487915  |
| shape_dynamic      | ecg             |         108 |                   0.888724 |         0.043357   |    0.043357   |             0.258801  |                     0.326665  |
| level_dynamic      | resp            |          56 |                   0.888724 |         0.0210765  |    0.0118055  |             0.162101  |                     0.314181  |
| shape_dynamic      | hr              |         108 |                   0.888724 |         0.0361572  |    0.0149863  |             0.102245  |                     0.300231  |
| level_dynamic      | hr              |          56 |                   0.888724 |         0.0306659  |    0.00620624 |             0.120401  |                     0.298423  |
| causal_past        | hr              |           9 |                   0.888724 |         0.0310685  |    0.0101163  |             0.131226  |                     0.278475  |
| causal_past        | eda             |           9 |                   0.737389 |         0.0223931  |    0.0073078  |             0.078888  |                     0.238208  |
| quality            | resp            |          25 |                   0.888724 |         0.0208779  |    0.0208779  |             0.181291  |                     0.217886  |
| shape_dynamic      | resp            |         108 |                   0.888724 |         0.0165949  |    0.0165949  |             0.116462  |                     0.211201  |
| rhythm             | resp            |          40 |                   0.888724 |         0.0132769  |    0.00582356 |             0.263745  |                     0.18975   |
| rhythm             | eda             |          24 |                   0.737389 |         0.021527   |    0.0056603  |             0.143999  |                     0.146455  |
| coupling           | emg             |          18 |                   0.888724 |         0.0149168  |    0.0116306  |             0.0851533 |                     0.145074  |
| rhythm             | emg             |          24 |                   0.888724 |         0.0172847  |    0.0172847  |             0.146044  |                     0.142509  |
| quality            | eda             |          25 |                   0.737389 |         0.046282   |    0.0185591  |             0.74489   |                     0.140718  |
| shape_dynamic      | emg             |         108 |                   0.888724 |         0.0318017  |    0.0145786  |             0.113813  |                     0.134152  |
| quality            | hr              |          25 |                   0.888724 |         0.0133402  |    0.0117888  |             0.127036  |                     0.132806  |
| quality            | emg             |          25 |                   0.888724 |         0.0178396  |    0.0178396  |             0.169022  |                     0.132728  |
| level_dynamic      | emg             |          56 |                   0.888724 |         0.0182575  |    0.0169339  |             0.154317  |                     0.128948  |
| coupling           | resp            |           6 |                   0.888724 |         0.00956482 |    0.00644329 |             0.104543  |                     0.118509  |
| level_dynamic      | ecg             |          56 |                   0.888724 |         0.0457388  |    0.0377538  |             0.364525  |                     0.0973467 |
| quality            | ecg             |          25 |                   0.888724 |         0.041691   |    0.020323   |             0.344612  |                     0.0922448 |
| coupling           | eda             |           6 |                   0.737389 |         0.0224064  |    0.00422688 |             0.163852  |                     0.0882265 |
| causal_past        | resp            |           9 |                   0.888724 |         0.01556    |    0.00500042 |             0.174161  |                     0.0808027 |
| coupling           | ecg             |           6 |                   0.888724 |         0.00605826 |    0.00439505 |             0.187365  |                     0.0606037 |
| rhythm             | ecg             |          32 |                   0.888724 |         0.0144027  |    0.0108287  |             0.688129  |                     0.0577274 |
| causal_past        | emg             |           9 |                   0.888724 |         0.0124932  |    0.0124932  |             0.27674   |                     0.0531235 |
| causal_past        | ecg             |           9 |                   0.888724 |         0.0152708  |    0.00619822 |             0.706927  |                     0.0172708 |

## train-only top20 特征

| feature                                              | feature_category   | signal_family   |   finite_rate_train |   behavior_eta_max |   bad_eta_max |   identity_eta_max |   identity_to_behavior_ratio |   behavior_identity_score |
|:-----------------------------------------------------|:-------------------|:----------------|--------------------:|-------------------:|--------------:|-------------------:|-----------------------------:|--------------------------:|
| bio285_delta_pre2_0_minus_pre20_pre10_eda_z_range    | shape_dynamic      | eda             |            0.737389 |         0.00720768 |   0.00672101  |         0.00356521 |                     0.494641 |                  0.531335 |
| bio285_pre20_pre10_eda_bin_mean_last_minus_first     | shape_dynamic      | eda             |            0.737389 |         0.0400816  |   0.0400816   |         0.0697346  |                     1.73981  |                  0.502688 |
| bio285_pre30_pre20_eda_z_pos_area_per_s              | level_dynamic      | eda             |            0.737389 |         0.0452955  |   0.0452955   |         0.0828348  |                     1.82877  |                  0.487915 |
| bio285_delta_pre2_0_minus_pre20_pre10_eda_z_std      | shape_dynamic      | eda             |            0.737389 |         0.00942068 |   0.00942068  |         0.0136384  |                     1.4477   |                  0.398534 |
| bio285_pre20_pre10_eda_z_slope                       | shape_dynamic      | eda             |            0.737389 |         0.0354185  |   0.0354185   |         0.0810619  |                     2.28869  |                  0.38895  |
| bio285_pre20_pre10_eda_z_last_minus_first            | shape_dynamic      | eda             |            0.737389 |         0.0354185  |   0.0354185   |         0.0810619  |                     2.28869  |                  0.38895  |
| bio285_delta_pre2_0_minus_pre5_pre2_eda_z_abs_mean   | shape_dynamic      | eda             |            0.737389 |         0.0214302  |   0.000291891 |         0.0497971  |                     2.3237   |                  0.358381 |
| bio285_pre2_0_eda_z_abs_mean                         | level_dynamic      | eda             |            0.737389 |         0.0210326  |   0.0002985   |         0.0488774  |                     2.32389  |                  0.357227 |
| bio285_pre2_0_eda_z_abs_area_per_s                   | level_dynamic      | eda             |            0.737389 |         0.0210326  |   0.0002985   |         0.0488774  |                     2.32389  |                  0.357227 |
| bio285_pre5_0_eda_z_abs_mean                         | level_dynamic      | eda             |            0.737389 |         0.0206001  |   0.000298844 |         0.048006   |                     2.33037  |                  0.355138 |
| bio285_pre5_0_eda_z_abs_area_per_s                   | level_dynamic      | eda             |            0.737389 |         0.0206001  |   0.000298844 |         0.048006   |                     2.33037  |                  0.355138 |
| bio285_pre1_0_eda_z_abs_mean                         | level_dynamic      | eda             |            0.737389 |         0.0207656  |   0.000319256 |         0.0491958  |                     2.3691   |                  0.350795 |
| bio285_pre1_0_eda_z_abs_area_per_s                   | level_dynamic      | eda             |            0.737389 |         0.0207656  |   0.000319256 |         0.0491958  |                     2.3691   |                  0.350795 |
| bio285_pre5_pre2_eda_z_abs_mean                      | level_dynamic      | eda             |            0.737389 |         0.019653   |   0.000294899 |         0.0461812  |                     2.34982  |                  0.349815 |
| bio285_pre5_pre2_eda_z_abs_area_per_s                | level_dynamic      | eda             |            0.737389 |         0.019653   |   0.000294899 |         0.0461812  |                     2.34982  |                  0.349815 |
| bio285_pre5_0_eda_z_range                            | level_dynamic      | eda             |            0.737389 |         0.0205949  |   0.000262136 |         0.0492332  |                     2.39055  |                  0.347692 |
| bio285_delta_pre2_0_minus_pre20_pre10_eda_z_abs_mean | shape_dynamic      | eda             |            0.737389 |         0.0172129  |   0.000297326 |         0.0400677  |                     2.32777  |                  0.343793 |
| bio285_delta_pre2_0_minus_pre5_pre2_eda_z_mean       | shape_dynamic      | eda             |            0.737389 |         0.0206032  |   0.000281651 |         0.0504169  |                     2.44705  |                  0.341017 |
| bio285_pre5_pre2_eda_bin_absmax                      | shape_dynamic      | eda             |            0.737389 |         0.0194961  |   0.000275193 |         0.0488064  |                     2.5034   |                  0.33153  |
| bio285_pre1_0_eda_bin_absmax                         | shape_dynamic      | eda             |            0.737389 |         0.0197585  |   0.000347196 |         0.0496409  |                     2.51238  |                  0.331291 |

## 关键判读

- route gate 未通过：即使回到底层 200Hz 信号形态，当前生理状态仍未形成可部署候选选择收益。
- deployable 结论只看 validation 选择后的 top1；top3/top5 oracle 只作为上限诊断。
- 如果本轮仍未通过，继续做更复杂融合模型的收益很低，应考虑更底层信号清洗/事件定义，或转为 subject-aware 个体校准任务。

## 关键图

- `figures\v285_badtop10_val_test_delta.png`
- `figures\v285_feature_screen_by_family.png`
- `figures\v285_bad_ambiguous_corr.png`

## guardrail

```json
{
  "pass": true,
  "zip_testzip": true,
  "event_n": 1167,
  "candidate_rows": 46680,
  "raw200_feature_n": 1146,
  "feature_set_n": 6,
  "uses_post_observation_any": false,
  "ok_rate": 0.919451585261354,
  "fixed_wait_latest_badtop10": 0.6950484153471495,
  "route_viable_now": false,
  "deployable_top1_badtop10_pass": false,
  "deployable_top1_bad_ambiguous_pass": false,
  "test_best_top1_diagnostic_pass": false,
  "reused_v260_feature_table": false,
  "test_used_for_feature_selection": false,
  "v283_old_route_closed": true
}
```