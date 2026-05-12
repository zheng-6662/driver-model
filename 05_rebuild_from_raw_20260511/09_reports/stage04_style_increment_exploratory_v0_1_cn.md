# 阶段 4：连续驾驶风格探索性增量对照 v0.1

## 输入与边界

- 样本：B 轨道 `response3s_strict_core_candidate`，窗口 `pre3_label3_response_coverage`。
- 主参照：`rbf_kernel_ridge_context_no_subject`。
- 风格表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_feature_candidate_wide_trainz_session_split.csv`。
- 标准化：沿用阶段 4 协议的 session-level train-only z-score。
- 模型：只做 RBF 残差 Ridge，不训练 Transformer，不使用生理、EEG、EMG、RESP 或驾驶员生理状态。

## 特征数量

```text
                                         model_name  n_features  selected_alpha                           feature_source
        rbf_plus_style_last60_guard3_residual_ridge          94         10000.0 train-only standardized continuous style
          rbf_plus_style_all_windows_residual_ridge         376          1000.0 train-only standardized continuous style
                  rbf_plus_driver_id_residual_ridge          18           100.0                          control one-hot
                rbf_plus_road_module_residual_ridge           7         10000.0                          control one-hot
rbf_plus_style_last60_with_driver_id_residual_ridge         112         10000.0 train-only standardized continuous style
```

## test 指标摘录

```text
                                               model_name  n_samples  rmse_steer  wrong_side_rate  large_response_recall  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse
                      rbf_kernel_ridge_context_no_subject       40.0    0.533667            0.225                   0.75                          1.109729                  0.125             0.181751                              0.0              0.678907
                        rbf_plus_driver_id_residual_ridge       40.0    0.533661            0.225                   0.75                          1.108888                  0.125             0.181435                              0.0              0.679859
                      rbf_plus_road_module_residual_ridge       40.0    0.533750            0.225                   0.75                          1.109755                  0.125             0.181751                              0.0              0.679023
                rbf_plus_style_all_windows_residual_ridge       40.0    0.564153            0.175                   0.75                          1.486933                  0.125             0.185400                              0.0              0.702178
        rbf_plus_style_last60_global_shuffle_seed20260513       40.0    0.534817            0.225                   0.75                          1.112312                  0.125             0.182778                              0.0              0.679413
        rbf_plus_style_last60_global_shuffle_seed20260514       40.0    0.532707            0.225                   0.75                          1.115730                  0.100             0.179523                              0.0              0.681154
        rbf_plus_style_last60_global_shuffle_seed20260515       40.0    0.537590            0.200                   0.75                          1.117044                  0.125             0.183094                              0.0              0.688425
        rbf_plus_style_last60_global_shuffle_seed20260516       40.0    0.533028            0.225                   0.75                          1.108536                  0.125             0.180035                              0.0              0.677506
        rbf_plus_style_last60_global_shuffle_seed20260517       40.0    0.533316            0.225                   0.75                          1.114022                  0.125             0.182058                              0.0              0.680297
              rbf_plus_style_last60_guard3_residual_ridge       40.0    0.534559            0.225                   0.75                          1.135936                  0.125             0.181904                              0.0              0.680891
 rbf_plus_style_last60_road_balanced_shuffle_seed20260513       40.0    0.535230            0.225                   0.75                          1.112060                  0.125             0.181586                              0.0              0.681650
 rbf_plus_style_last60_road_balanced_shuffle_seed20260514       40.0    0.545441            0.225                   0.75                          1.157388                  0.125             0.187954                              0.0              0.715393
 rbf_plus_style_last60_road_balanced_shuffle_seed20260515       40.0    0.536073            0.225                   0.75                          1.106220                  0.125             0.180412                              0.0              0.677653
 rbf_plus_style_last60_road_balanced_shuffle_seed20260516       40.0    0.538395            0.225                   0.75                          1.116591                  0.125             0.178393                              0.0              0.683649
 rbf_plus_style_last60_road_balanced_shuffle_seed20260517       40.0    0.532771            0.225                   0.75                          1.108060                  0.125             0.178864                              0.0              0.677369
      rbf_plus_style_last60_with_driver_id_residual_ridge       40.0    0.534558            0.225                   0.75                          1.135930                  0.125             0.181897                              0.0              0.680895
rbf_plus_style_last60_within_subject_shuffle_seed20260513       40.0    0.532148            0.200                   0.75                          1.063211                  0.125             0.181379                              0.0              0.678938
rbf_plus_style_last60_within_subject_shuffle_seed20260514       40.0    0.547281            0.200                   0.75                          1.109989                  0.125             0.175797                              0.0              0.683567
rbf_plus_style_last60_within_subject_shuffle_seed20260515       40.0    0.533355            0.225                   0.75                          1.149095                  0.125             0.181426                              0.0              0.678039
rbf_plus_style_last60_within_subject_shuffle_seed20260516       40.0    0.535619            0.225                   0.75                          1.121413                  0.125             0.183007                              0.0              0.679387
rbf_plus_style_last60_within_subject_shuffle_seed20260517       40.0    0.534267            0.225                   0.75                          1.114440                  0.125             0.184071                              0.0              0.681539
        rbf_plus_style_last60_within_subject_shuffle_mean       40.0    0.536534            0.215                   0.75                          1.111629                  0.125             0.181136                              0.0              0.680294
                rbf_plus_style_last60_global_shuffle_mean       40.0    0.534292            0.220                   0.75                          1.113529                  0.120             0.181498                              0.0              0.681359
         rbf_plus_style_last60_road_balanced_shuffle_mean       40.0    0.537582            0.225                   0.75                          1.120064                  0.125             0.181442                              0.0              0.687143
```

## gate

```text
                           gate_item                    status                                                                                                                              evidence                                             decision_cn
             fixed_vehicle_reference            pass_reference                                                                          rbf_kernel_ridge_context_no_subject test RMSE=0.533667; n=40                      固定 RBF/KRR 车辆-only 作为本轮连续风格增量对照底线。
         style_last60_beats_rbf_rmse                      fail                                                                                    style60 test RMSE=0.534559; RBF test RMSE=0.533667                                  只作为探索性迹象；不能单独证明连续风格有效。
            style_not_only_driver_id fail_or_driver_proxy_risk                                                                style60 RMSE=0.534559; driver ID RMSE=0.533661; style+ID RMSE=0.534558 需继续用 subject-level 或留一被试验证；本轮 session-level 不能排除身份代理风险。
                shuffle_control_drop          pass_exploratory                                                                 style60 true RMSE=0.534559; within-subject shuffle mean RMSE=0.536534               置乱后若收益下降，说明存在样本-风格对应信号；仍需更多 split 和 seed。
physical_metric_improvement_required              needs_review wrong_side RBF=0.225000, style60=0.225000; large_recall RBF=0.750000, style60=0.750000; difficult RMSE RBF=0.678907, style60=0.680891                             若只改善 RMSE 而不改善物理错误，不能升级为主线。
   style_effectiveness_claim_allowed                   blocked                                                                       当前只完成 session-level 探索性残差对照；subject-level/跨被试验证、更多置乱和固定图复核尚未完成。                            不能宣称连续风格有效，只能说形成或未形成下一步验证候选。
                  physio_eeg_allowed                   blocked                                                                                                           车辆+连续风格公平参照还没有完成多 split 验证。                                       生理/EEG 仍不进入有效性验证。
```

## 输出

- 指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_metrics.csv`
- 逐样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_per_sample_metrics.csv`
- alpha 选择：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_validation_selection.csv`
- 置乱汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_permutation_summary.csv`
- gate 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_gate_table.csv`
- 固定图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_fixed_predictions_test.png`
- 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/figures/style_increment_bad_samples_test.png`

## 解释限制

本轮如果出现收益，只能说明“事件前连续风格候选值得继续做更严格验证”。它还不满足阶段 4 对有效性的完成标准：至少两类切分成立、置乱收益稳定下降、不是驾驶员 ID 替代品、物理错误或困难样本稳定改善。
