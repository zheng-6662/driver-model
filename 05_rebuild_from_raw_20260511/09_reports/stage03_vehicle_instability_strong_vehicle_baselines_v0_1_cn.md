# 阶段 3：更强车辆-only 时序/结构化基线 v0.1

生成时间：2026-05-12

## 目的

上一轮正式车辆-only ridge 基线说明，主要错误集中在幅值不足、启动延迟、反向修正、多段修正和尾段漂移。本轮继续在阶段 3 内部强化车辆-only 基线，仍然不使用生理、脑电、连续风格或驾驶员 ID。

## 输入和边界

- 样本：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 车辆窗口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/arrays/pre2_label2_old_main.npz`
- 主窗口：`pre2_label2_old_main`
- 切分：`session_level_split`
- 选择规则：候选模型和超参数只使用 train/val；test 只用于最终报告。
- 特征：事件前 2 秒车辆时序统计、下采样车辆历史、事件/道路上下文；不包含 `eval_label_*`、生理、脑电、连续风格、驾驶员 ID。

## 候选模型

1. `formal_ridge_vehicle_context_no_subject`：上一轮正式 ridge 上下文基线，作为本轮内部参考。
2. `ridge_rich_history_no_subject`：更丰富的事件前车辆历史统计 + 下采样时序，不含上下文。
3. `ridge_rich_context_no_subject`：丰富车辆历史 + 事件/道路上下文。
4. `rbf_kernel_ridge_context_no_subject`：RBF kernel ridge 非线性车辆模型，直接预测整条轨迹。
5. `knn_template_context_no_subject`：车辆特征检索训练集响应模板。
6. `direction_gated_knn_template_no_subject`：先预测主峰方向，再在同方向训练模板中检索。
7. `peak_scaled_template_context_no_subject`：先预测方向和峰值幅值，再检索归一化模板并按幅值缩放。

## val 选择结果

本轮预先规定按 val RMSE 选择候选模型。val 排名如下：

                             model_name  n_samples  rmse_steer  wrong_side_rate  severe_amp_under_rate  reversal_count_exact_match_rate
    rbf_kernel_ridge_context_no_subject        156    0.743835         0.198718               0.352564                         0.051282
        knn_template_context_no_subject        156    0.763502         0.288462               0.326923                         0.032051
peak_scaled_template_context_no_subject        156    0.777061         0.211538               0.185897                         0.064103
direction_gated_knn_template_no_subject        156    0.788281         0.211538               0.160256                         0.096154
          ridge_rich_context_no_subject        156    0.816020         0.275641               0.391026                         0.025641
          ridge_rich_history_no_subject        156    0.821956         0.269231               0.455128                         0.044872
formal_ridge_vehicle_context_no_subject        156    0.826239         0.288462               0.583333                         0.115385

val 选择模型：`rbf_kernel_ridge_context_no_subject`。

## session-level test 指标

                             model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_mae  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  onset_delay_mae_s  tail_abs_error_mean  tail_drift_risk_rate  reversal_count_exact_match_rate  difficult_top20_rmse
        knn_template_context_no_subject        139    0.516941                 0.827338         0.172662                   0.52      0.420714                          1.280109               0.244604         0.432374           0.333741             0.501961              0.446043                         0.007194              1.040524
    rbf_kernel_ridge_context_no_subject        139    0.540287                 0.784173         0.215827                   0.60      0.422609                          0.949628               0.251799         0.455432           0.458957             0.530500              0.532374                         0.043165              1.015198
peak_scaled_template_context_no_subject        139    0.555055                 0.791367         0.208633                   0.60      0.395637                          1.798675               0.079137         0.386403           0.306403             0.552014              0.460432                         0.093525              1.066179
direction_gated_knn_template_no_subject        139    0.579581                 0.791367         0.208633                   0.64      0.396333                          2.024186               0.071942         0.397374           0.338345             0.574765              0.482014                         0.129496              1.085050
formal_ridge_vehicle_context_no_subject        139    0.649341                 0.769784         0.230216                   0.08      0.654372                          1.401294               0.582734         0.449029           0.716942             0.663662              0.625899                         0.093525              1.265239
          ridge_rich_context_no_subject        139    0.652941                 0.820144         0.179856                   0.24      0.552534                          2.096012               0.330935         0.432518           0.327734             0.651683              0.553957                         0.035971              1.254761
          ridge_rich_history_no_subject        139    0.680683                 0.755396         0.244604                   0.16      0.598466                          2.021668               0.374101         0.482302           0.373885             0.696528              0.589928                         0.057554              1.307168

旧 `vehicle_direct` clean active checkpoint 只作为历史参照：RMSE=0.637366，错侧率=0.129496，严重幅值不足率=0.683453。

## 关键图

- 固定样本预测图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_fixed_predictions_test.png`
- val 选择模型坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_bad_samples_test.png`
- test 指标柱状图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_model_metric_bars_test.png`
- 与 formal ridge 的逐样本 RMSE 差异：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_baselines_v0_1/figures/strong_vehicle_selected_vs_formal_rmse_delta.png`

## val 选择模型的坏样本错误分型

                     selected_model                   error_flag  n_samples     rate  mean_rmse
rbf_kernel_ridge_context_no_subject       reversal_mismatch_flag        133 0.956835   0.450679
rbf_kernel_ridge_context_no_subject              tail_drift_flag         74 0.532374   0.503029
rbf_kernel_ridge_context_no_subject   peak_time_large_error_flag         49 0.352518   0.436294
rbf_kernel_ridge_context_no_subject onset_delay_large_error_flag         48 0.345324   0.507846
rbf_kernel_ridge_context_no_subject  multi_segment_mismatch_flag         38 0.273381   0.433688
rbf_kernel_ridge_context_no_subject        severe_amp_under_flag         35 0.251799   0.590085
rbf_kernel_ridge_context_no_subject              wrong_side_flag         30 0.215827   0.467744
rbf_kernel_ridge_context_no_subject           high_rmse_top20pct         28 0.201439   0.928449
rbf_kernel_ridge_context_no_subject  zero_crossing_mismatch_flag         20 0.143885   0.555786
rbf_kernel_ridge_context_no_subject   large_response_missed_flag         10 0.071942   1.246666

## 当前判断

本轮是车辆-only 强化，不支持任何风格、生理或 EEG 有效性结论。和上一轮 formal ridge 相比，val 选择模型 test RMSE 从 0.649341 变为 0.540287，错侧率从 0.230216 变为 0.215827，严重幅值不足率从 0.582734 变为 0.251799，反向修正精确匹配率从 0.093525 变为 0.043165。

是否升级为阶段 3 主车辆基线，不能只看一个 RMSE，需要同时看固定图、坏样本图、错侧、幅值、尾段和反向/多段修正。
