# 阶段 3 用户查看版：车辆-only Transformer 时序基线 v0.1

生成时间：2026-05-12

## 为什么做

前一轮强车辆结果主要用了 KNN/RBF/模板检索，它可以作为诊断对照，但不是 Transformer。这个阶段补一个真正的车辆-only Transformer，让后续讨论“强车辆基线”时有神经时序模型作为参照。

## 这次检查了什么

- 输入只用事件前 2 秒车辆时序和事件/道路上下文。
- 不用生理、脑电、连续风格，也不用驾驶员 ID。
- 输出事件后 2 秒方向盘增量轨迹。
- 早停只看验证集，测试集只做最后评估。

## 目前发现

                             model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  onset_delay_mae_s  tail_abs_error_mean  tail_drift_risk_rate  reversal_count_exact_match_rate  difficult_top20_rmse
 vehicle_transformer_context_no_subject        139    0.567162                 0.820144         0.179856                   0.44      0.470062                          1.544153               0.266187         0.447230           0.380899             0.526112              0.460432                         0.201439              1.089107
formal_ridge_vehicle_context_no_subject        139    0.649341                 0.769784         0.230216                   0.08      0.654372                          1.401294               0.582734         0.449029           0.716942             0.663662              0.625899                         0.093525              1.265239

## 哪些结果可信

训练、标准化、早停都遵守 train/val/test 边界，没有使用测试集信息训练模型，也没有使用未来标签作为输入。

## 哪些还不能下结论

这仍然是车辆-only 阶段，不能说明生理、脑电或连续风格有效。Transformer 是否作为后续主车辆基线，还要看固定图和坏样本图里的物理错误是否比 RBF/KNN/formal ridge 更合理。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/figures/vehicle_transformer_fixed_predictions_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/figures/vehicle_transformer_bad_samples_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/tables/vehicle_transformer_metrics.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_vehicle_transformer_v0_1_cn.md`
