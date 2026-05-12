# 阶段 3 技术报告：RBF 主参照冻结审计 v0.1

## 决策

固定 `rbf_kernel_ridge_context_no_subject` 为 `B_response3s_strict_core` 后续增量实验的保守车辆-only 主参照。冻结类型为 `limited_reference_freeze`，不是 `vehicle_problem_solved`。

## RBF profile

| metric | value | interpretation_cn |
| --- | --- | --- |
| n_test_samples | 40.000000 | B 轨道严格响应覆盖 test 样本数。 |
| rmse_steer | 0.533667 | 主参照整体误差。 |
| wrong_side_rate | 0.225000 | 主峰方向错侧率，仍偏高。 |
| large_response_recall | 0.750000 | 大幅响应召回，仍有漏召回。 |
| severe_amp_under_rate | 0.125000 | 严重幅值不足率。 |
| tail_drift_risk_rate | 0.050000 | 尾段漂移风险。 |
| reversal_count_exact_match_rate | 0.000000 | 反向修正计数完全匹配率，是当前最大物理缺陷。 |
| difficult_top20_rmse | 0.678907 | 困难峰值样本 RMSE。 |

## Failure profile

| flag | flag_cn | overall_count | overall_rate | high_rmse_top20_count | high_rmse_top20_rate |
| --- | --- | --- | --- | --- | --- |
| high_rmse_top20_flag | RMSE最高20% | 8 | 0.200000 | 8 | 1.000000 |
| wrong_side_flag | 主峰错侧 | 9 | 0.225000 | 3 | 0.375000 |
| severe_amp_under_flag | 严重幅值不足 | 5 | 0.125000 | 3 | 0.375000 |
| large_response_missed_flag | 大幅响应漏召回 | 2 | 0.050000 | 2 | 0.250000 |
| tail_drift_flag | 尾段漂移/未回正 | 2 | 0.050000 | 0 | 0.000000 |
| zero_crossing_mismatch_flag | 零线穿越错误 | 3 | 0.075000 | 2 | 0.250000 |
| reversal_mismatch_flag | 反向修正计数不匹配 | 40 | 1.000000 | 8 | 1.000000 |
| multi_segment_mismatch_flag | 多段修正结构不匹配 | 1 | 0.025000 | 0 | 0.000000 |
| peak_time_large_error_flag | 峰值时间误差大 | 9 | 0.225000 | 4 | 0.500000 |
| onset_delay_large_error_flag | 启动延迟误差大 | 7 | 0.175000 | 2 | 0.250000 |
| amplitude_large_error_flag | 峰值幅值误差大 | 10 | 0.250000 | 4 | 0.500000 |

## Freeze gates

| gate_item | status | evidence | decision_cn |
| --- | --- | --- | --- |
| reference_identity_fixed | pass_limited | rbf_kernel_ridge_context_no_subject; B test RMSE=0.533667 | 固定为 B 轨道后续增量实验的保守车辆-only 主参照。 |
| reference_is_deployable_vehicle_only | pass | 输入只包含事件前车辆历史和因果可得道路/事件上下文；不使用 subject ID、生理、脑电或连续风格。 | 可作为公平车辆-only 对照。 |
| physical_errors_explained | pass_limited | wrong_side=0.225; reversal_exact=0.000; large_recall=0.750; failure summary 已覆盖错侧、幅值、尾段、反向修正、启动延迟等。 | 错误类型已被列出并可追溯，但还没有被车辆-only 模型解决。 |
| vehicle_only_problem_solved | fail | 反向修正计数完全匹配率为 0；错侧率仍为 0.225；top-K fallback 未超过 RBF。 | 不能宣称车辆-only 已解决方向盘物理响应预测。 |
| oracle_used_as_performance | fail_if_used | best-of-RBF+topK 仅作事后上限。 | 后续所有主表必须区分可部署模型与 oracle 上限。 |
| stage04_style_protocol_allowed | conditional_pass | 主参照身份已固定，但必须携带 RBF 物理缺陷，并用置乱、分被试和物理指标验证风格增量。 | 允许进入阶段 4 的协议设计/探索性实验；不得直接宣称连续风格有效。 |
| stage05_physio_eeg_allowed | blocked | 连续风格验证尚未完成；生理/EEG 仍需在更强车辆+风格参照后验证。 | 生理、脑电仍阻塞。 |

## Robustness snapshot

| robustness_config_id | window_config_id | split_strategy | val_selected_model | val_selected_test_rmse | rbf_test_rmse | knn_test_rmse | knn_train_rmse | knn_memory_risk | interpretation_cn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_main | pre2_label2_old_main | random_event_split | rbf_kernel_ridge_context_no_subject | 0.613049 | 0.613049 | 0.614998 | 0.000001 | True | KNN 训练集近零误差，仍按模板记忆风险处理；val 选择候选相对 formal 有提升。 |
| session_pre1 | pre1_label2_event_trigger | session_level_split | peak_scaled_template_context_no_subject | 0.539372 | 0.520104 | 0.525429 | 0.000003 | True | KNN 训练集近零误差，仍按模板记忆风险处理；val 选择候选相对 formal 有提升。 |
| session_pre3 | pre3_label3_response_coverage | session_level_split | direction_gated_knn_template_no_subject | 0.658432 | 0.607706 | 0.590207 | 0.000002 | True | KNN 训练集近零误差，仍按模板记忆风险处理；val 选择候选相对 formal 有提升。 |
| subject_main | pre2_label2_old_main | subject_level_split | rbf_kernel_ridge_context_no_subject | 0.609792 | 0.609792 | 0.597936 | 0.000001 | True | KNN 训练集近零误差，仍按模板记忆风险处理；val 选择候选相对 formal 有提升。 |

## Top bad samples

| sample_id | subject | road_design_module_name | sample_rmse | primary_failure_type | failure_tags |
| --- | --- | --- | --- | --- | --- |
| vehicle_instability_allraw__zx__2025_09_27_18_00_08__000173505__pre3_label3_response_coverage | zx | curve1 | 1.155950 | wrong_side | high_rmse_top20_flag;wrong_side_flag;zero_crossing_mismatch_flag;reversal_mismatch_flag;peak_time_large_error_flag |
| vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000392590__pre3_label3_response_coverage | gzj | middle_section | 1.071305 | large_response_missed | high_rmse_top20_flag;severe_amp_under_flag;large_response_missed_flag;reversal_mismatch_flag;onset_delay_large_error_flag;amplitude_large_error_flag |
| vehicle_instability_allraw__gf__2025_09_26_10_30_12__000425785__pre3_label3_response_coverage | gf | middle_section | 0.993488 | large_response_missed | high_rmse_top20_flag;severe_amp_under_flag;large_response_missed_flag;reversal_mismatch_flag;onset_delay_large_error_flag;amplitude_large_error_flag |
| vehicle_instability_allraw__zx__2025_09_27_18_00_08__000149680__pre3_label3_response_coverage | zx | curve1 | 0.959970 | wrong_side | high_rmse_top20_flag;wrong_side_flag;zero_crossing_mismatch_flag;reversal_mismatch_flag;peak_time_large_error_flag |
| vehicle_instability_allraw__zx__2025_09_27_18_00_08__000353905__pre3_label3_response_coverage | zx | curve1 | 0.757357 | reversal_structure_mismatch | high_rmse_top20_flag;reversal_mismatch_flag;peak_time_large_error_flag |
| vehicle_instability_allraw__tyy__2025_09_28_14_44_09__000058890__pre3_label3_response_coverage | tyy | longstraight | 0.674694 | reversal_structure_mismatch | high_rmse_top20_flag;reversal_mismatch_flag;amplitude_large_error_flag |
| vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000221890__pre3_label3_response_coverage | hzh | fix_road | 0.660698 | reversal_structure_mismatch | high_rmse_top20_flag;reversal_mismatch_flag |
| vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000326200__pre3_label3_response_coverage | hzh | curve2 | 0.657525 | wrong_side | high_rmse_top20_flag;wrong_side_flag;severe_amp_under_flag;reversal_mismatch_flag;peak_time_large_error_flag;amplitude_large_error_flag |

## 后续约束

阶段 4 可以开始“连续风格协议设计/探索性实验”，但不能直接宣称风格有效。所有风格增量必须对比固定 RBF 主参照，并做置乱、分被试、物理指标、困难样本和坏样本图。生理/EEG 仍阻塞。
