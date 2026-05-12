# 阶段 3 用户查看版：B 轨道车辆-only 响应分解 Transformer v0.1

## 为什么做

上一轮已经补跑了真正的车辆-only Transformer，但它没有超过 B 轨道 RBF KRR。这个阶段进一步测试“结构化预测”是否更合适：模型先预测方向、幅值、峰值时间、启动时间、响应形态和尾段状态，再预测轨迹。

## 这次检查了什么

- B 轨道：3 秒响应覆盖严格核心样本，270 条，是当前更重要的主线。
- 输入只用事件前车辆历史和道路/事件上下文。
- 不用生理、脑电、连续风格、驾驶员 ID，也不把未来响应分解标签当输入；响应分解标签只作为训练目标。
- 模型内部加了一个物理约束：方向盘增量在 t=0 从 0 开始。

## 目前发现

一句话判断：这是车辆-only 结构化路线的第一版，不是生理/风格实验；是否继续要看它相对 RBF KRR 是否改善物理错误和坏样本。
表里的 `direct tx` 是上一轮真正 vehicle-only Transformer；`knn` 只是模板参照，不是本轮要验证的 Transformer。

```text
                track_id                                    model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  severe_amp_under_rate  peak_time_mae_s  onset_delay_mae_s  tail_abs_error_mean  tail_drift_risk_rate  reversal_count_exact_match_rate  difficult_top20_rmse
B_response3s_strict_core           rbf_kernel_ridge_context_no_subject         40    0.533667                    0.775            0.225                  0.750      0.409504                  0.125         0.434750           0.257625             0.181751                 0.050                            0.000              0.678907
B_response3s_strict_core        vehicle_transformer_context_no_subject         40    0.566011                    0.775            0.225                  0.625      0.550825                  0.300         0.537625           0.852125             0.298802                 0.175                            0.050              0.770506
B_response3s_strict_core structured_vehicle_transformer_aux_no_subject         40    0.602174                    0.775            0.225                  0.500      0.553695                  0.350         0.532250           0.643625             0.330727                 0.250                            0.075              0.802289
B_response3s_strict_core               knn_template_context_no_subject         40    0.625829                    0.825            0.175                  0.750      0.445666                  0.175         0.413500           0.436000             0.171038                 0.050                            0.000              0.710014
```

## 哪些结果可信

可信的是：这次只在 B 轨道干净响应样本上跑了车辆-only 结构化 Transformer，训练标准化和早停都只看 train/val，没有把 test 信息用于训练。

## 哪些还不能下结论

还不能说生理、脑电或连续风格有效；也不能只因为加了响应分解辅助头就默认更强，必须看 B 轨道 test 指标、辅助头准确率和坏样本图。

## 下一步是否可以继续

可以继续。若这版结构化模型仍然解决不了反向修正、多段修正、峰值时间和尾段错误，下一步应尝试关键点+残差或多假设车辆模型，而不是跳到生理结论。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/B_response3s_strict_core_structured_bad_samples_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/figures/structured_vehicle_transformer_metric_summary_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_structured_vehicle_transformer_v0_1/tables/structured_vehicle_transformer_metrics.csv`
