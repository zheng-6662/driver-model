# 阶段 3 用户查看版：干净响应任务车辆-only Transformer v0.1

## 为什么做

你指出现在跑出来的对照主要还是 KNN/RBF/KRR，不是 Transformer。本轮就把 Transformer 补到同一批干净样本上，避免拿旧的混合 906 样本 Transformer 和现在的 A/B 干净轨道混着比较。

## 这次检查了什么

- A 轨道：2 秒即时响应核心样本，84 条，只作诊断。
- B 轨道：3 秒响应覆盖严格核心样本，270 条，是当前更重要的主线。
- 输入只用事件前车辆历史和道路/事件上下文。
- 不用生理、脑电、连续风格、驾驶员 ID，也不把未来响应分解标签当输入。
- 模型内部加了一个物理约束：方向盘增量在 t=0 从 0 开始。

## 目前发现

一句话判断：B 轨道已经补跑真正的车辆-only Transformer，但直接 Transformer 当前没有超过 RBF KRR 主参照。

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

## 哪些结果可信

可信的是：这次确实是在 A/B 干净响应轨道上跑了车辆-only Transformer，训练标准化和早停都只看 train/val，没有把 test 信息用于训练。

## 哪些还不能下结论

还不能说生理、脑电或连续风格有效；也不能只因为模型叫 Transformer 就默认比 RBF/KRR 更强，必须看 B 轨道 test 指标和坏样本图。

## 下一步是否可以继续

可以继续。若 Transformer 仍然解决不了反向修正、多段修正、峰值时间和尾段错误，下一步应进入响应分解/关键点+残差车辆模型，而不是跳到生理结论。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/B_response3s_strict_core_fixed_predictions_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/B_response3s_strict_core_transformer_bad_samples_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/figures/clean_task_vehicle_transformer_metric_summary_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_clean_task_vehicle_transformer_v0_1/tables/clean_task_vehicle_transformer_metrics.csv`
