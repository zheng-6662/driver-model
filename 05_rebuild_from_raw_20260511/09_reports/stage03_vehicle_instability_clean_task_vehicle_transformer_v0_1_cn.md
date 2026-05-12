# 阶段 3：干净响应任务车辆-only Transformer v0.1

生成时间：2026-05-13

## 为什么做

用户指出上一轮干净响应任务的最优对照仍然是 KNN/RBF/KRR 等非 Transformer 模型。因此本轮在同一 A/B 干净轨道上补跑真正的车辆-only Transformer，对齐固定图、坏样本图和物理指标。

## 输入和无泄漏边界

- 样本 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`
- A 轨道：`pre2_label2_old_main` + `instant2s_core_candidate`，84 个样本。
- B 轨道：`pre3_label3_response_coverage` + `response3s_strict_core_candidate`，270 个样本。
- 输入：事件前车辆时序 9 个车辆特征 + 可因果获得的事件/道路上下文。
- 不使用：生理、脑电、连续风格、驾驶员 ID、响应分解标签、`eval_label_*` 未来标签。
- 标准化：车辆时序和数值上下文只在各轨道 train split 拟合。
- 模型选择：Transformer 早停只看 val RMSE；test 只用于最终评估。
- 物理边界：模型内部扣除自己的 t=0 输出，使方向盘增量轨迹从 0 开始。
- 本轮未连接服务器，未读取服务器指令与密码文件。

## 模型信息

```text
                track_id  best_epoch  best_val_rmse  epochs_ran                             model_name              window_config_id                 task_sample_role  train_n  val_n  test_n  label_scale_train_std  context_feature_count  vehicle_input_tokens  vehicle_input_downsample_step prediction_zero_origin_constraint device                                                                                                                                                                                                                checkpoint_path
        A_instant2s_core         1.0       0.484257        16.0 vehicle_transformer_context_no_subject          pre2_label2_old_main         instant2s_core_candidate       62     10      12               0.427783                   50.0                 101.0                            4.0                              True   cuda         F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/checkpoints/A_instant2s_core_vehicle_transformer_context_no_subject_best.pt
B_response3s_strict_core        45.0       0.602414        60.0 vehicle_transformer_context_no_subject pre3_label3_response_coverage response3s_strict_core_candidate      188     42      40               0.780019                   51.0                 101.0                            6.0                              True   cuda F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_vehicle_transformer_context_no_subject_best.pt
```

## test 指标对照

```text
                track_id                              model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  severe_amp_under_rate  peak_time_mae_s  onset_delay_mae_s  tail_abs_error_mean  tail_drift_risk_rate  reversal_count_exact_match_rate  difficult_top20_rmse
        A_instant2s_core  vehicle_transformer_context_no_subject         12    0.336483                 0.416667         0.583333                  0.000      0.444799               1.000000         1.005417           1.667500             0.099739              0.333333                         0.333333              0.696410
        A_instant2s_core     rbf_kernel_ridge_context_no_subject         12    0.338846                 0.750000         0.250000                  0.400      0.345994               0.416667         0.386667           0.492083             0.071555              0.166667                         0.083333              0.499163
        A_instant2s_core formal_ridge_vehicle_context_no_subject         12    0.363449                 0.583333         0.416667                  0.400      0.300308               0.250000         0.346667           0.455833             0.082073              0.250000                         0.000000              0.734370
        A_instant2s_core         knn_template_context_no_subject         12    0.428130                 0.666667         0.333333                  0.600      0.392557               0.333333         0.430417           0.413750             0.090362              0.416667                         0.250000              0.529858
B_response3s_strict_core     rbf_kernel_ridge_context_no_subject         40    0.533667                 0.775000         0.225000                  0.750      0.409504               0.125000         0.434750           0.257625             0.181751              0.050000                         0.000000              0.678907
B_response3s_strict_core  vehicle_transformer_context_no_subject         40    0.566011                 0.775000         0.225000                  0.625      0.550825               0.300000         0.537625           0.852125             0.298802              0.175000                         0.050000              0.770506
B_response3s_strict_core         knn_template_context_no_subject         40    0.625829                 0.825000         0.175000                  0.750      0.445666               0.175000         0.413500           0.436000             0.171038              0.050000                         0.000000              0.710014
B_response3s_strict_core formal_ridge_vehicle_context_no_subject         40    0.652392                 0.850000         0.150000                  0.125      0.826006               0.750000         0.426500           1.327125             0.183897              0.050000                         0.050000              0.975715
```

## 按 val 选择的当前结果

- A_instant2s_core: val 选择 `knn_template_context_no_subject`；test RMSE=0.428130，错侧率=0.333333，大幅响应召回=0.600000。
- B_response3s_strict_core: val 选择 `rbf_kernel_ridge_context_no_subject`；test RMSE=0.533667，错侧率=0.225000，大幅响应召回=0.750000。

## Transformer 单独结果

- A_instant2s_core: Transformer test RMSE=0.336483，错侧率=0.583333，大幅响应召回=0.000000，反向修正完全匹配率=0.333333。
- B_response3s_strict_core: Transformer test RMSE=0.566011，错侧率=0.225000，大幅响应召回=0.625000，反向修正完全匹配率=0.050000。

## B 轨道判断

B 轨道 Transformer 的 test RMSE=0.566011，高于 RBF KRR 的 0.533667；wrong-side 二者同为 0.225000/0.225000，large recall 为 0.625000 vs 0.750000。因此这次补跑确认了 Transformer 对照，但当前不能把直接 Transformer 升级为主车辆基线。

## 图

- 指标概览：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/clean_task_vehicle_transformer_metric_summary_test.png`
- A 轨道固定图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/A_instant2s_core_fixed_predictions_test.png`
- A 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/A_instant2s_core_transformer_bad_samples_test.png`
- B 轨道固定图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
- B 轨道坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/B_response3s_strict_core_transformer_bad_samples_test.png`

## 当前结论边界

这一步只回答车辆历史和事件/道路上下文下的 Transformer 表现，不能说明连续风格、生理或 EEG 有效。若 Transformer 仍无法改善反向修正、多段修正和尾段错误，下一步应使用刚生成的响应分解标签做辅助目标或结构化轨迹模型，而不是直接引入生理解释。
