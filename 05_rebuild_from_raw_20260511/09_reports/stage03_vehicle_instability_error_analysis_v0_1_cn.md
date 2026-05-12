# 阶段 3 v0.1 坏样本物理错误分型分析

生成时间：2026-05-12

## 目的

阶段 3 v0.1 的最优浅层车辆基线 `ridge_vehicle_context_no_subject` 虽然是当前新流程车辆-only 起点，但固定图显示它仍有明显物理错误。本分析把 test 样本逐条打上错误标签，判断下一步应优先强化车辆模型哪一部分。

## 输入

- 正式车辆基线逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_formal_baselines_v0_1/tables/formal_baseline_per_sample_metrics.csv`
- 正式样本清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 旧 `vehicle_direct` clean 对照逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_per_sample_metrics.csv`

## 样本范围

- 窗口：`pre2_label2_old_main`
- split：`session_level_split` / `test`
- 模型：`ridge_vehicle_context_no_subject`
- test 样本数：139

## 错误标签计数

                  error_flag  n_samples     rate  mean_rmse  median_gt_peak_abs  old_deep_mean_rmse
      reversal_mismatch_flag        126 0.906475   0.518772            0.829643            0.544419
             tail_drift_flag         87 0.625899   0.605790            0.920487            0.578756
       severe_amp_under_flag         81 0.582734   0.617606            1.086118            0.673040
onset_delay_large_error_flag         75 0.539568   0.600795            1.015260            0.639875
 multi_segment_mismatch_flag         46 0.330935   0.483546            0.807826            0.495272
  peak_time_large_error_flag         43 0.309353   0.490863            0.731118            0.490251
 multi_segment_overpred_flag         42 0.302158   0.492287            0.807826            0.496489
             wrong_side_flag         32 0.230216   0.646427            0.704240            0.539724
          high_rmse_top20pct         28 0.201439   1.098625            1.736691            0.980742
 zero_crossing_mismatch_flag         24 0.172662   0.704208            0.565224            0.510065
  large_response_missed_flag         23 0.165468   0.955515            1.769938            0.982881
   multi_segment_missed_flag          4 0.028777   0.391756            0.959059            0.482489

## 主错误类型计数

       primary_error_type  n_samples
      03_severe_amp_under         42
            01_wrong_side         32
     06_reversal_mismatch         25
 02_large_response_missed         20
05_multi_segment_overpred         15
         10_no_major_flag          2
  04_multi_segment_missed          2
            07_tail_drift          1

## 分响应类型错误

eval_label_morphology  n_samples  mean_rmse  wrong_side_rate  large_response_missed_rate  severe_amp_under_rate  multi_segment_missed_rate  multi_segment_overpred_rate  multi_segment_mismatch_rate  reversal_mismatch_rate  tail_drift_rate  high_rmse_top20pct_rate
     multi_correction         94   0.533247         0.202128                    0.159574               0.627660                   0.042553                     0.000000                     0.042553                0.893617         0.563830                 0.191489
   reverse_correction         36   0.483778         0.277778                    0.194444               0.583333                   0.000000                     0.916667                     0.916667                0.916667         0.722222                 0.222222
          single_lobe          9   0.438468         0.333333                    0.111111               0.111111                   0.000000                     1.000000                     1.000000                1.000000         0.888889                 0.222222

## 分被试错误，按 mean RMSE 前 12

subject  n_samples  mean_rmse  wrong_side_rate  large_response_missed_rate  severe_amp_under_rate  multi_segment_missed_rate  multi_segment_overpred_rate  multi_segment_mismatch_rate  reversal_mismatch_rate  tail_drift_rate  high_rmse_top20pct_rate
    zxy         11   0.804746         0.272727                    0.090909               0.545455                   0.000000                     0.181818                     0.181818                0.909091         0.545455                 0.272727
    tyy         10   0.704148         0.500000                    0.200000               0.500000                   0.000000                     0.400000                     0.400000                0.900000         0.800000                 0.500000
    gzj         17   0.600713         0.000000                    0.235294               0.647059                   0.058824                     0.058824                     0.117647                0.941176         0.588235                 0.235294
     gf         12   0.521253         0.250000                    0.333333               0.750000                   0.000000                     0.166667                     0.166667                0.916667         0.583333                 0.250000
    hzh         53   0.479915         0.169811                    0.188679               0.603774                   0.018868                     0.452830                     0.471698                0.905660         0.679245                 0.188679
     zx         11   0.472151         0.454545                    0.000000               0.636364                   0.000000                     0.090909                     0.090909                1.000000         0.636364                 0.090909
    yyl         12   0.381628         0.333333                    0.083333               0.500000                   0.083333                     0.250000                     0.333333                0.833333         0.583333                 0.083333
    byx         13   0.301379         0.230769                    0.076923               0.384615                   0.076923                     0.384615                     0.461538                0.846154         0.461538                 0.076923

## 与旧 deep vehicle_direct 对照

                                       comparison  n_samples  formal_aggregate_rmse  old_deep_aggregate_rmse  formal_mean_rmse  old_deep_mean_rmse  formal_better_n  formal_better_rate  shared_bad_top20pct_n  formal_bad_top20pct_n  old_deep_bad_top20pct_n
formal_ridge_context_vs_old_vehicle_direct_active        139               0.649341                 0.637366          0.514298            0.545169               92            0.661871                     21                     28                       28

说明：`formal_aggregate_rmse` 和 `old_deep_aggregate_rmse` 是主指标表中的整体 RMSE；`formal_mean_rmse` 和 `old_deep_mean_rmse` 是逐样本 RMSE 的算术平均。两者口径不同，不能混用。旧 deep 的整体 RMSE 仍略低于 formal ridge，但 formal ridge 在更多单个样本上逐样本 RMSE 更小，说明 formal ridge 的错误更集中在少数高幅/复杂响应样本上。

## Top 12 坏样本

                                                                            sample_id subject  sample_rmse  gt_peak_abs  pred_peak_abs        primary_error_type  wrong_side_flag  large_response_missed_flag  severe_amp_under_flag  multi_segment_missed_flag  multi_segment_overpred_flag  multi_segment_mismatch_flag  reversal_mismatch_flag  tail_drift_flag  old_deep_sample_rmse
vehicle_instability_allraw__zxy__2025_09_28_16_35_30__000185120__pre2_label2_old_main     zxy     3.064340     2.708229       1.676651             01_wrong_side             True                       False                  False                      False                        False                        False                    True             True              2.248861
vehicle_instability_allraw__zxy__2025_09_28_16_35_30__000460020__pre2_label2_old_main     zxy     1.745926     0.450290       2.412558             01_wrong_side             True                       False                  False                      False                        False                        False                    True             True              0.150945
vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000392590__pre2_label2_old_main     gzj     1.515566     2.594604       0.227807  02_large_response_missed            False                        True                   True                      False                        False                        False                    True             True              1.477252
vehicle_instability_allraw__zxy__2025_09_28_16_35_30__000471085__pre2_label2_old_main     zxy     1.506541     2.604385       0.386149             01_wrong_side             True                        True                   True                      False                         True                         True                    True             True              1.199644
vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000592750__pre2_label2_old_main     gzj     1.464037     2.132270       0.230836  02_large_response_missed            False                        True                   True                      False                        False                        False                    True             True              1.376794
vehicle_instability_allraw__tyy__2025_09_28_14_44_09__000554145__pre2_label2_old_main     tyy     1.448888     0.050440       1.847090 05_multi_segment_overpred            False                       False                  False                      False                         True                         True                    True             True              0.562181
vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000221890__pre2_label2_old_main     hzh     1.121195     1.678482       0.211415  02_large_response_missed            False                        True                   True                      False                         True                         True                    True             True              1.225015
vehicle_instability_allraw__tyy__2025_09_28_14_44_09__000058890__pre2_label2_old_main     tyy     1.104195     2.333853       0.470537  02_large_response_missed            False                        True                   True                      False                        False                        False                    True             True              1.262975
 vehicle_instability_allraw__gf__2025_09_26_10_30_12__000138490__pre2_label2_old_main      gf     1.102031     1.925098       0.289957  02_large_response_missed            False                        True                   True                      False                         True                         True                    True             True              0.490095
 vehicle_instability_allraw__gf__2025_09_26_10_30_12__000425785__pre2_label2_old_main      gf     1.078778     2.145881       0.704391  02_large_response_missed            False                        True                   True                      False                         True                         True                    True            False              0.913515
vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000053185__pre2_label2_old_main     gzj     1.040882     2.043077       0.551441  02_large_response_missed            False                        True                   True                      False                        False                        False                    True             True              1.242462
vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000326200__pre2_label2_old_main     hzh     1.021964     1.307079       0.283550             01_wrong_side             True                       False                   True                      False                        False                        False                    True             True              0.877424

## 图

- 错误标签柱状图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/formal_error_flag_counts.png`
- 与旧 deep RMSE 散点图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/formal_vs_old_deep_rmse_scatter.png`
- Top bad 错误矩阵：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/top_bad_sample_error_matrix.png`
- 分被试错误热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/subject_error_rate_heatmap.png`

## 当前判断

车辆-only 浅层基线的主要问题不是单一 RMSE，而是高比例的复杂响应结构错误：反向修正计数不匹配、多段修正过度预测或漏检、尾段漂移、严重幅值不足和错侧同时存在。下一步更适合先增强车辆时序/结构化响应基线，而不是直接声称连续风格或生理提供增量。
