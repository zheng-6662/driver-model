# 阶段 3：RBF vs keypoint train/val 选择器 v0.1

生成时间：2026-05-13

## 为什么做

RBF KRR 的整体 RMSE 稳定，但 keypoint+residual 在错侧率和大幅响应召回上更好。本轮测试一个只用 train 训练、只用 val 定阈值的选择器，判断是否能在 test 前决定每个样本选 RBF 还是 keypoint。

## 输入和无泄漏边界

- 逐样本指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/tables/keypoint_residual_vehicle_transformer_per_sample_metrics.csv`
- 样本 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_response_task_decision_v0_1/tables/sample_response_task_manifest.csv`
- 训练：selector 只用 train split 拟合。
- 阈值：只用 val split 按 selector RMSE 选择。
- test：只做最终评价，不参与训练或阈值选择。
- 特征：事件前车辆模型已经可输出的候选预测特征 + 事件/道路上下文；不使用 GT peak、sample_rmse、wrong_side、large_response、subject ID、生理、脑电、连续风格。
- 本轮未连接服务器，未读取服务器指令与密码文件。

## 阈值选择

- val 选择阈值：0.55
- val selector RMSE：0.561189
- test keypoint 选择率：0.275000

## test 指标对照

```text
                                      model_name  n_samples  rmse_steer  wrong_side_rate  large_response_recall  peak_amp_mae  peak_time_mae_s  onset_delay_mae_s  tail_drift_risk_rate  reversal_count_exact_match_rate  difficult_top20_rmse
         oracle_best_of_rbf_keypoint_upper_bound         40    0.475095            0.200                  0.875      0.418862         0.440250           0.521500                 0.050                            0.000              0.648368
             rbf_kernel_ridge_context_no_subject         40    0.533667            0.225                  0.750      0.409504         0.434750           0.257625                 0.050                            0.000              0.678907
         selector_logreg_rbf_keypoint_no_subject         40    0.533912            0.200                  0.875      0.386642         0.415875           0.339375                 0.075                            0.025              0.648368
keypoint_residual_vehicle_transformer_no_subject         40    0.548994            0.125                  0.875      0.445834         0.476500           0.844750                 0.075                            0.025              0.728866
```

## test 选择计数

```text
                                  selected_model  n_test_samples
             rbf_kernel_ridge_context_no_subject              29
keypoint_residual_vehicle_transformer_no_subject              11
```

## 结论边界

这个选择器是第一版 train/val 可用策略，不是 test oracle。若它不能超过 RBF，说明当前可用特征还不足以稳定判断何时选 keypoint；若它接近 oracle，则可以继续发展为多假设/可靠性模型。无论结果如何，本轮不能说明连续风格、生理或 EEG 有效。

## 图

- test 指标图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_test_metrics.png`
- 阈值扫描图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_threshold_sweep.png`
