# v0.3 全量样本车辆-only 基线（用户查看版）

## 这次为什么做

在 v0.3 全量原始车辆数据重筛后，先不加入连续驾驶风格、生理数据或脑电，只验证车辆历史、道路/工况上下文和早期车辆状态能否预测后续方向盘相对轨迹。

这一步的作用是先确认新筛出来的极限/近极限工况样本本身是否更适合建模。如果车辆-only 都站不住，后面直接解释风格或生理增量会不可靠。

## 数据集

- 样本来源：v0.3 全量原始车辆数据 episode 表。
- 纳入类别：强响应、弱/保守响应、延迟/无明显转向、正常对照。
- 排除类别：待人工复核、已排除样本。
- 可用样本数：482。
- 划分数量：train=280，val=88，test=114。
- 输入窗口：工况锚点前 2 秒，20 Hz。
- 标签窗口：工况锚点后 5 秒方向盘相对变化，20 Hz。
- 输入特征：方向盘、方向盘角速度、车速、制动、油门、纵向/横向加速度、横摆、横滚、横向偏移、路面附着系数、曲率等车辆/道路信息。
- 未使用：连续驾驶风格、生理数据、脑电、驾驶员 ID。

## test 总体指标

| model_name | split | n | rmse_steer | primary_rmse_0_2s | tail_rmse_2_5s | peak_abs_mae | wrong_side_rate_large | severe_amp_under_rate_large | large_response_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_kernel_vehicle_context_alpha0.1_g2 | test | 114 | 0.797252 | 0.600911 | 0.905429 | 0.649123 | 0.0964912 | 0.22807 | 0.428571 |
| zero_delta | test | 114 | 0.811135 | 0.638187 | 0.909279 | 0.982288 | 0.307018 | 0.307018 | 0 |
| knn_vehicle_history_context_k5 | test | 114 | 0.819036 | 0.609355 | 0.933327 | 0.629376 | 0.122807 | 0.22807 | 0.628571 |
| ridge_vehicle_history_context_alpha1000 | test | 114 | 0.821677 | 0.647704 | 0.920137 | 0.656143 | 0.0964912 | 0.245614 | 0.571429 |
| train_global_mean | test | 114 | 0.822071 | 0.636862 | 0.926055 | 0.829358 | 0.166667 | 0.307018 | 0 |
| train_category_mean | test | 114 | 0.825545 | 0.615827 | 0.940586 | 0.766296 | 0.175439 | 0.289474 | 0.257143 |
| train_context_mean | test | 114 | 0.832098 | 0.646957 | 0.936043 | 0.785911 | 0.166667 | 0.307018 | 0 |
| linear_trend_from_last_rate | test | 114 | 4.39105 | 1.79402 | 5.46983 | 3.42279 | 0.254386 | 0.22807 | 0.257143 |

## 当前最好车辆-only 基线

- 最好模型：`rbf_kernel_vehicle_context_alpha0.1_g2`
- test RMSE：0.797252
- 主响应阶段 0-2s RMSE：0.600911
- 尾段 2-5s RMSE：0.905429
- 大响应错侧率：0.096491
- 大响应严重幅值不足率：0.228070
- 大响应召回：0.428571

## 最好模型分样本类型结果

| v0_3_category_cn | n | large_n | rmse_steer_approx | peak_abs_mae | mean_gt_peak_abs | mean_pred_peak_abs | wrong_side_rate_large | severe_amp_under_rate_large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 正常驾驶/普通弯道对照 | 19 | 1 | 0.19208 | 0.154396 | 0.243749 | 0.161511 | 0 | 1 |
| 弱响应/保守响应 | 46 | 11 | 0.665565 | 0.6862 | 0.932864 | 0.361303 | 0.454545 | 1 |
| 延迟或无明显转向响应 | 35 | 14 | 0.808566 | 0.528055 | 1.109 | 0.658116 | 0.214286 | 0.428571 |
| 强响应型极限工况 | 14 | 9 | 1.42677 | 1.50139 | 1.83021 | 0.348994 | 0.333333 | 0.888889 |

## 最好模型分被试结果

| subject | n | large_n | rmse_steer_approx | peak_abs_mae | mean_gt_peak_abs | mean_pred_peak_abs | wrong_side_rate_large | severe_amp_under_rate_large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jy | 2 | 0 | 0.0433604 | 0.062839 | 0 | 0.062839 | NA | NA |
| lx | 13 | 2 | 0.378921 | 0.286202 | 0.700025 | 0.430498 | 0 | 0.5 |
| zdq | 7 | 1 | 0.50251 | 0.195834 | 0.521005 | 0.508843 | 1 | 0 |
| gf | 9 | 4 | 0.56614 | 0.461459 | 0.853185 | 0.449527 | 0.25 | 1 |
| byx | 10 | 2 | 0.601897 | 0.406911 | 0.728423 | 0.557966 | 0 | 0.5 |
| zx | 28 | 11 | 0.784332 | 0.66591 | 1.1123 | 0.525867 | 0.363636 | 0.545455 |
| xst | 14 | 6 | 0.978213 | 0.746189 | 1.01013 | 0.27193 | 0.5 | 0.833333 |
| txj | 31 | 9 | 1.01111 | 1.01511 | 1.25756 | 0.327996 | 0.222222 | 1 |

## 结论边界

- 这轮结果只说明新 v0.3 样本上的车辆-only 可预测性，不能证明连续风格或生理数据有效。
- 目前最好车辆-only 模型相对零响应基线有小幅总体改善，并明显降低了大响应错侧率，但大响应召回仍然不足。
- 如果预测图中仍然存在严重幅值压缩或物理意义不对，下一步应优先检查样本类型、锚点和响应分组，而不是直接加入生理数据。

## 可查看文件

- 固定预测图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_fixed_predictions_test.png`
- 坏样本图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\figures\v03_vehicle_only_bad_samples_test.png`
- 总指标表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_baseline_metrics.csv`
- 分样本类型表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_category_test.csv`
- 分被试表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_subject_test.csv`
- 分工况上下文表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v03_vehicle_only_baselines\tables\v03_vehicle_only_best_model_by_context_test.csv`