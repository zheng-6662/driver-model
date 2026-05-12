# 阶段 3：keypoint+residual vs RBF 坏样本差异复盘 v0.1

生成时间：2026-05-13

## 为什么做

keypoint+residual 在 B 轨道 test 上 RMSE 仍略差于 RBF KRR，但错侧率和大幅响应召回更好。因此需要逐样本检查它到底修复了哪些物理错误、又在哪些样本上退化。

## 输入

- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_per_sample_metrics.csv`
- 范围：B 轨道 `response3s_strict_core_candidate` 的 session-level test 40 个样本。
- 对照：`keypoint_residual_vehicle_transformer_no_subject` vs `rbf_kernel_ridge_context_no_subject`。
- 本轮只读已有车辆-only 评估表，不训练模型，不使用生理、脑电、连续风格、服务器或服务器密码文件。

## 总结

- test 样本数：40
- keypoint - RBF 样本 RMSE 平均差：0.025325
- RMSE 明显改善样本数：11
- RMSE 明显退化样本数：20
- 修复错侧样本数：5
- 新增错侧样本数：1
- 修复大幅响应召回样本数：1
- 丢失大幅响应召回样本数：0
- 修复尾段漂移样本数：0
- 新增尾段漂移样本数：1

## 变化计数

```text
        change_type              label  n_samples  rate
        rmse_change           degraded         20 0.500
        rmse_change           improved         11 0.275
        rmse_change            similar          9 0.225
   direction_change   fixed_wrong_side          5 0.125
   direction_change     new_wrong_side          1 0.025
   direction_change          unchanged         34 0.850
large_recall_change fixed_large_recall          1 0.025
large_recall_change          unchanged         39 0.975
  tail_drift_change     new_tail_drift          1 0.025
  tail_drift_change          unchanged         39 0.975
   amp_under_change    fixed_under_amp          3 0.075
   amp_under_change      new_under_amp          6 0.150
   amp_under_change          unchanged         31 0.775
   peak_time_change degraded_peak_time         19 0.475
   peak_time_change improved_peak_time         15 0.375
   peak_time_change            similar          6 0.150
       onset_change     degraded_onset         30 0.750
       onset_change     improved_onset          6 0.150
       onset_change            similar          4 0.100
```

## 分被试摘要

```text
subject  n_samples  rmse_delta_mean  rmse_improved_rate  wrong_side_fixed  new_wrong_side  large_recall_fixed  large_recall_lost
     zx          4        -0.174087            0.500000                 0               1                   0                  0
    byx          2        -0.025550            0.000000                 0               0                   0                  0
    zxy          2         0.009965            0.000000                 0               0                   0                  0
    gzj          9         0.015275            0.333333                 0               0                   0                  0
    tyy          2         0.057498            0.000000                 0               0                   0                  0
    hzh         14         0.062531            0.285714                 3               0                   0                  0
     gf          4         0.065051            0.500000                 1               0                   1                  0
    yyl          3         0.117471            0.000000                 1               0                   0                  0
```

## RMSE 改善最大的样本

```text
                                                                                     sample_id subject  sample_rmse__delta_keypoint_minus_rbf direction_change large_recall_change tail_drift_change   peak_time_change   onset_change
 vehicle_instability_allraw__zx__2025_09_27_18_00_08__000173505__pre3_label3_response_coverage      zx                              -0.380288        unchanged           unchanged         unchanged degraded_peak_time degraded_onset
 vehicle_instability_allraw__zx__2025_09_27_18_00_08__000353905__pre3_label3_response_coverage      zx                              -0.377715   new_wrong_side           unchanged         unchanged            similar degraded_onset
vehicle_instability_allraw__hzh__2025_09_26_20_50_27__000034605__pre3_label3_response_coverage     hzh                              -0.190748        unchanged           unchanged         unchanged improved_peak_time degraded_onset
 vehicle_instability_allraw__gf__2025_09_26_10_30_12__000425785__pre3_label3_response_coverage      gf                              -0.154870        unchanged  fixed_large_recall         unchanged degraded_peak_time improved_onset
vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000081700__pre3_label3_response_coverage     gzj                              -0.153219        unchanged           unchanged         unchanged improved_peak_time degraded_onset
vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000504660__pre3_label3_response_coverage     hzh                              -0.152621        unchanged           unchanged         unchanged improved_peak_time degraded_onset
vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000518180__pre3_label3_response_coverage     gzj                              -0.150863        unchanged           unchanged         unchanged degraded_peak_time degraded_onset
vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000362460__pre3_label3_response_coverage     gzj                              -0.138182        unchanged           unchanged         unchanged degraded_peak_time        similar
```

## RMSE 退化最大的样本

```text
                                                                                     sample_id subject  sample_rmse__delta_keypoint_minus_rbf direction_change large_recall_change tail_drift_change   peak_time_change   onset_change
 vehicle_instability_allraw__gf__2025_09_26_10_52_57__000300870__pre3_label3_response_coverage      gf                               0.444123 fixed_wrong_side           unchanged         unchanged improved_peak_time        similar
vehicle_instability_allraw__hzh__2025_09_26_20_50_27__000681250__pre3_label3_response_coverage     hzh                               0.276423 fixed_wrong_side           unchanged         unchanged degraded_peak_time improved_onset
vehicle_instability_allraw__hzh__2025_09_27_19_44_05__000081885__pre3_label3_response_coverage     hzh                               0.238882        unchanged           unchanged         unchanged improved_peak_time degraded_onset
vehicle_instability_allraw__yyl__2025_09_28_09_49_11__000327680__pre3_label3_response_coverage     yyl                               0.225455        unchanged           unchanged         unchanged            similar degraded_onset
vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000221890__pre3_label3_response_coverage     hzh                               0.206840        unchanged           unchanged         unchanged improved_peak_time degraded_onset
vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000585940__pre3_label3_response_coverage     gzj                               0.162569        unchanged           unchanged         unchanged degraded_peak_time improved_onset
vehicle_instability_allraw__yyl__2025_09_28_09_49_11__000062885__pre3_label3_response_coverage     yyl                               0.157127 fixed_wrong_side           unchanged         unchanged improved_peak_time degraded_onset
vehicle_instability_allraw__hzh__2025_09_27_19_44_05__000326390__pre3_label3_response_coverage     hzh                               0.146547        unchanged           unchanged         unchanged degraded_peak_time degraded_onset
```

## 图

- RMSE 差异 Top 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/figures/keypoint_vs_rbf_rmse_delta_top_samples.png`
- 错误变化计数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_vs_rbf_bad_sample_review_v0_1/figures/keypoint_vs_rbf_error_change_counts.png`

## 结论边界

这个复盘只说明 keypoint+residual 与 RBF 在 B 轨道 test 样本上的错误转移关系。它不能证明连续风格、生理或 EEG 有效，也不能替代后续多 seed/多切分验证。
