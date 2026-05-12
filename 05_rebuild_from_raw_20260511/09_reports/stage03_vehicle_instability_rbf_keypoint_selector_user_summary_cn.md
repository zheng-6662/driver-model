# 阶段 3 用户查看版：RBF 和 keypoint 的自动选择器

## 为什么做

RBF 整体更稳，keypoint 更会修方向和大幅响应。我们不能在 test 上事后挑哪个模型好，所以这一步用 train 训练一个选择器，再用 val 定阈值，最后只在 test 上看结果。

## 这次检查了什么

- 每个样本在预测前应该选 RBF 还是 keypoint。
- 选择器不用生理、脑电、连续风格和驾驶员 ID。
- 选择器不用 test 结果调参。

## 目前发现

- val 选出的阈值：0.55
- test 上 keypoint 被选择比例：0.275000

```text
                                      model_name  n_samples  rmse_steer  wrong_side_rate  large_response_recall  peak_amp_mae  peak_time_mae_s  onset_delay_mae_s  tail_drift_risk_rate  reversal_count_exact_match_rate  difficult_top20_rmse
         oracle_best_of_rbf_keypoint_upper_bound         40    0.475095            0.200                  0.875      0.418862         0.440250           0.521500                 0.050                            0.000              0.648368
             rbf_kernel_ridge_context_no_subject         40    0.533667            0.225                  0.750      0.409504         0.434750           0.257625                 0.050                            0.000              0.678907
         selector_logreg_rbf_keypoint_no_subject         40    0.533912            0.200                  0.875      0.386642         0.415875           0.339375                 0.075                            0.025              0.648368
keypoint_residual_vehicle_transformer_no_subject         40    0.548994            0.125                  0.875      0.445834         0.476500           0.844750                 0.075                            0.025              0.728866
```

## 哪些结果可信

可信的是：这是一个不看 test 调参的初版选择器，可以判断当前可用特征是否足以在 RBF/keypoint 之间做可部署选择。

## 哪些还不能下结论

如果 selector 没超过 RBF，不能说 keypoint 没价值，只能说当前选择特征不够；也不能由此证明生理或风格有效。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_test_metrics.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/figures/rbf_keypoint_selector_threshold_sweep.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_metrics.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_keypoint_selector_v0_1/tables/rbf_keypoint_selector_decisions.csv`
