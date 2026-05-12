# 阶段 3：B 轨道车辆-only 响应分解 Transformer v0.1

生成时间：2026-05-13

## 为什么做

直接车辆-only Transformer 没有超过 B 轨道 RBF KRR，且坏样本仍集中在反向修正、多段修正、峰值时间、启动延迟和尾段问题。因此本轮在 B 轨道上做响应分解辅助头：先让模型预测方向、幅值桶、峰值时间桶、启动时间桶、响应形态和尾段状态，再用模型自己的结构表征辅助轨迹头。

## 输入和无泄漏边界

- 样本 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`
- B 轨道：`pre3_label3_response_coverage` + `response3s_strict_core_candidate`，270 个样本。
- 输入：事件前车辆时序 9 个车辆特征 + 可因果获得的事件/道路上下文。
- 响应分解标签：只作为训练目标和评估目标，不作为推理输入。
- 不使用：生理、脑电、连续风格、驾驶员 ID、真实响应分解标签输入、`eval_label_*` 未来标签输入。
- 标准化：车辆时序和数值上下文只在各轨道 train split 拟合。
- 模型选择：结构化 Transformer 早停只看 val 轨迹 RMSE；test 只用于最终评估。
- 物理边界：模型内部扣除自己的 t=0 输出，使方向盘增量轨迹从 0 开始。
- 本轮未连接服务器，未读取服务器指令与密码文件。

## 模型信息

```text
                track_id  best_epoch  best_val_rmse  epochs_ran                                    model_name              window_config_id                 task_sample_role  train_n  val_n  test_n  label_scale_train_std  context_feature_count  vehicle_input_tokens  vehicle_input_downsample_step uses_response_decomposition_labels_as_targets trajectory_uses_predicted_structure_features prediction_zero_origin_constraint device                                                                                                                                                                                                                       checkpoint_path
B_response3s_strict_core        42.0       0.621421        57.0 structured_vehicle_transformer_aux_no_subject pre3_label3_response_coverage response3s_strict_core_candidate      188     42      40               0.780019                   51.0                 101.0                            6.0                                          True                                         True                              True   cuda F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_structured_vehicle_transformer_aux_no_subject_best.pt
```

## test 指标对照

说明：`direct tx` 是上一轮真正 vehicle-only Transformer 的同一 B 轨道结果；`knn` 只是模板参照，不是本轮主模型。

```text
                track_id                                    model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  severe_amp_under_rate  peak_time_mae_s  onset_delay_mae_s  tail_abs_error_mean  tail_drift_risk_rate  reversal_count_exact_match_rate  difficult_top20_rmse
B_response3s_strict_core           rbf_kernel_ridge_context_no_subject         40    0.533667                    0.775            0.225                  0.750      0.409504                  0.125         0.434750           0.257625             0.181751                 0.050                            0.000              0.678907
B_response3s_strict_core        vehicle_transformer_context_no_subject         40    0.566011                    0.775            0.225                  0.625      0.550825                  0.300         0.537625           0.852125             0.298802                 0.175                            0.050              0.770506
B_response3s_strict_core structured_vehicle_transformer_aux_no_subject         40    0.602174                    0.775            0.225                  0.500      0.553695                  0.350         0.532250           0.643625             0.330727                 0.250                            0.075              0.802289
B_response3s_strict_core               knn_template_context_no_subject         40    0.625829                    0.825            0.175                  0.750      0.445666                  0.175         0.413500           0.436000             0.171038                 0.050                            0.000              0.710014
```

## 响应分解辅助头准确率

```text
split  n_samples  peak_direction_accuracy  amplitude_bucket_accuracy  peak_time_bucket_accuracy  onset_bucket_accuracy  computed_morphology_accuracy  tail_state_accuracy  is_large_response_target_accuracy                 track_id                                    model_name
train        188                 1.000000                   0.941489                   0.904255                0.81383                      0.984043             0.851064                           0.952128 B_response3s_strict_core structured_vehicle_transformer_aux_no_subject
  val         42                 0.857143                   0.523810                   0.595238                0.50000                      0.904762             0.547619                           0.809524 B_response3s_strict_core structured_vehicle_transformer_aux_no_subject
 test         40                 0.825000                   0.550000                   0.475000                0.67500                      0.950000             0.525000                           0.750000 B_response3s_strict_core structured_vehicle_transformer_aux_no_subject
```

## 按 val 选择的当前结果

- B_response3s_strict_core: val 选择 `rbf_kernel_ridge_context_no_subject`；test RMSE=0.533667，错侧率=0.225000，大幅响应召回=0.750000。

## 结构化 Transformer 单独结果

- B_response3s_strict_core: 结构化 Transformer test RMSE=0.602174，错侧率=0.225000，大幅响应召回=0.500000，反向修正完全匹配率=0.075000。

## B 轨道判断

B 轨道结构化 Transformer 的 test RMSE=0.602174，RBF KRR 为 0.533667；wrong-side 二者同为 0.225000/0.225000，large recall 为 0.500000 vs 0.750000。结构化辅助头是否值得继续，要同时看 RMSE、方向、幅值、峰时、尾段和坏样本图。

## 图

- 指标概览：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/structured_vehicle_transformer_metric_summary_test.png`
- B 轨道固定图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/B_response3s_strict_core_structured_bad_samples_test.png`

## 当前结论边界

这一步只回答车辆历史和事件/道路上下文下的结构化车辆模型表现，不能说明连续风格、生理或 EEG 有效。如果响应分解辅助头带来物理指标改善但 RMSE 没有明显改善，只能把它作为结构路线候选，不能升级为最终主线。
