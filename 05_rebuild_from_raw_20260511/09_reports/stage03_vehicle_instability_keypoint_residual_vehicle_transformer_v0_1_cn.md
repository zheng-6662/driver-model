# 阶段 3：B 轨道车辆-only 关键点 + 残差 Transformer v0.1

生成时间：2026-05-13

## 为什么做

上一轮响应分解分类辅助头没有超过 RBF KRR，也没有改善大幅响应和尾段。因此本轮换成更直接的物理结构：模型先预测启动点、主峰时间、主峰幅值和尾段值，生成一条关键点折线，再学习残差轨迹。目标是检查关键点约束是否比分类辅助头更适合车辆-only 预测。

## 输入和无泄漏边界

- 样本 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`
- B 轨道：`pre3_label3_response_coverage` + `response3s_strict_core_candidate`，270 个样本。
- 输入：事件前车辆时序 9 个车辆特征 + 可因果获得的事件/道路上下文。
- 关键点标签：只作为训练目标和评估目标，不作为推理输入。
- 不使用：生理、脑电、连续风格、驾驶员 ID、真实关键点标签输入、`eval_label_*` 未来标签输入。
- 标准化：车辆时序和数值上下文只在 train split 拟合。
- 模型选择：早停只看 val 轨迹 RMSE；test 只用于最终评估。
- 物理边界：模型内部扣除自己的 t=0 输出，使方向盘增量轨迹从 0 开始。
- 本轮未连接服务器，未读取服务器指令与密码文件。

## 模型信息

```text
                track_id  best_epoch  best_val_rmse  epochs_ran                                       model_name              window_config_id                 task_sample_role  train_n  val_n  test_n  label_scale_train_std  context_feature_count  vehicle_input_tokens  vehicle_input_downsample_step uses_keypoint_labels_as_targets uses_keypoint_labels_as_input trajectory_uses_keypoint_base_plus_residual prediction_zero_origin_constraint device                                                                                                                                                                                                                                 checkpoint_path
B_response3s_strict_core         5.0       0.598301        20.0 keypoint_residual_vehicle_transformer_no_subject pre3_label3_response_coverage response3s_strict_core_candidate      188     42      40               0.780019                   51.0                 101.0                            6.0                            True                         False                                        True                              True   cuda F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_keypoint_residual_vehicle_transformer_no_subject_best.pt
```

## test 指标对照

说明：`direct tx` 和 `structured` 是前两轮 Transformer 结果；`knn` 只是模板参照，不是本轮主模型。

```text
                track_id                                       model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  severe_amp_under_rate  peak_time_mae_s  onset_delay_mae_s  tail_abs_error_mean  tail_drift_risk_rate  reversal_count_exact_match_rate  difficult_top20_rmse
B_response3s_strict_core              rbf_kernel_ridge_context_no_subject         40    0.533667                    0.775            0.225                  0.750      0.409504                  0.125         0.434750           0.257625             0.181751                 0.050                            0.000              0.678907
B_response3s_strict_core keypoint_residual_vehicle_transformer_no_subject         40    0.548994                    0.875            0.125                  0.875      0.445834                  0.200         0.476500           0.844750             0.195331                 0.075                            0.025              0.728866
B_response3s_strict_core           vehicle_transformer_context_no_subject         40    0.566011                    0.775            0.225                  0.625      0.550825                  0.300         0.537625           0.852125             0.298802                 0.175                            0.050              0.770506
B_response3s_strict_core    structured_vehicle_transformer_aux_no_subject         40    0.602174                    0.775            0.225                  0.500      0.553695                  0.350         0.532250           0.643625             0.330727                 0.250                            0.075              0.802289
B_response3s_strict_core                  knn_template_context_no_subject         40    0.625829                    0.825            0.175                  0.750      0.445666                  0.175         0.413500           0.436000             0.171038                 0.050                            0.000              0.710014
```

## 关键点回归误差

```text
split  n_samples  peak_signed_mae  tail_signed_mae  onset_frac_mae  peak_frac_mae                 track_id                                       model_name
train        188         0.590768         0.176760        0.140085       0.143282 B_response3s_strict_core keypoint_residual_vehicle_transformer_no_subject
  val         42         0.618813         0.208077        0.171224       0.167170 B_response3s_strict_core keypoint_residual_vehicle_transformer_no_subject
 test         40         0.516174         0.194605        0.170120       0.134467 B_response3s_strict_core keypoint_residual_vehicle_transformer_no_subject
```

## 按 val 选择的当前结果

- B_response3s_strict_core: val 选择 `rbf_kernel_ridge_context_no_subject`；test RMSE=0.533667，错侧率=0.225000，大幅响应召回=0.750000。

## keypoint+residual 单独结果

- B_response3s_strict_core: keypoint+residual test RMSE=0.548994，错侧率=0.125000，大幅响应召回=0.875000，尾段漂移风险=0.075000，反向修正完全匹配率=0.025000。

## B 轨道判断

B 轨道 keypoint+residual 的 test RMSE=0.548994；RBF KRR=0.533667；direct Transformer=0.566011；本模型 wrong-side=0.125000，large recall=0.875000，tail drift=0.075000。 是否继续必须看 RMSE、幅值、峰时、尾段和坏样本图，不能只看关键点头误差。

## 图

- 指标概览：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/keypoint_residual_vehicle_transformer_metric_summary_test.png`
- B 轨道固定图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/figures/B_response3s_strict_core_keypoint_residual_bad_samples_test.png`

## 当前结论边界

这一步只回答车辆历史和事件/道路上下文下的关键点 + 残差车辆模型表现，不能说明连续风格、生理或 EEG 有效。如果关键点误差下降但轨迹物理指标没有改善，也不能升级为主线。
