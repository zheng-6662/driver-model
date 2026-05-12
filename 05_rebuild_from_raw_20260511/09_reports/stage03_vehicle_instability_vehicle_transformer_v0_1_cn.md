# 阶段 3：车辆-only Transformer 时序基线 v0.1

生成时间：2026-05-12

## 为什么做

用户指出上一轮强车辆-only 主要是 KNN/RBF/模板检索，不是真正的 Transformer。这个版本建立一个明确的车辆-only Transformer 时序神经基线，用来回答“只用事件前车辆时序和事件/道路上下文时，Transformer 能做到什么程度”。

## 输入和边界

- 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 处理后车辆窗口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/arrays/pre2_label2_old_main.npz`
- 主窗口：`pre2_label2_old_main`
- split：`session_level_split`
- 输入：事件前 2 秒车辆时序 9 个车辆特征 + 事件/道路上下文。
- 输出：事件后 2 秒方向盘增量轨迹。
- 不使用：生理、脑电、连续风格、驾驶员 ID、`eval_label_*` 训练输入。
- 标准化：车辆时序和数值上下文只在 train split 拟合。
- 模型选择：早停只看 val RMSE；test 只用于最终评估。

## 模型

- Encoder：2 层 TransformerEncoder，`d_model=64`，`nhead=4`。
- Decoder：全局车辆历史表示 + 上下文表示 + label time embedding，逐时间点输出未来方向盘增量。
- 损失：masked trajectory MSE + 0.08 一阶差分 MSE。
- 最佳 epoch：32，val RMSE=0.716799。

## session-level test 指标

                             model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  onset_delay_mae_s  tail_abs_error_mean  tail_drift_risk_rate  reversal_count_exact_match_rate  difficult_top20_rmse
 vehicle_transformer_context_no_subject        139    0.567162                 0.820144         0.179856                   0.44      0.470062                          1.544153               0.266187         0.447230           0.380899             0.526112              0.460432                         0.201439              1.089107
formal_ridge_vehicle_context_no_subject        139    0.649341                 0.769784         0.230216                   0.08      0.654372                          1.401294               0.582734         0.449029           0.716942             0.663662              0.625899                         0.093525              1.265239

## 与上一轮 RBF/KNN 诊断候选的参考

                         model_name  rmse_steer  wrong_side_rate  large_response_recall  severe_amp_under_rate  reversal_count_exact_match_rate
    knn_template_context_no_subject    0.516941         0.172662                   0.52               0.244604                         0.007194
rbf_kernel_ridge_context_no_subject    0.540287         0.215827                   0.60               0.251799                         0.043165

## 图

- 固定样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/figures/vehicle_transformer_fixed_predictions_test.png`
- Transformer 坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_transformer_v0_1/figures/vehicle_transformer_bad_samples_test.png`

## 当前判断

Transformer test RMSE=0.567162，formal ridge test RMSE=0.649341。这一步只说明车辆-only Transformer 在当前设置下的表现，不支持连续风格、生理或 EEG 有效性结论。是否把 Transformer 作为下一版主车辆基线，还需要看固定图/坏样本图，以及它是否改善方向、幅值、尾段、反向修正和多段修正。
