# 阶段 3：B 轨道 RBF KRR 坏样本物理复查 v0.1

## 为什么做

clean-task 车辆-only 对照显示，B_response3s_strict_core 上 RBF KRR 是按验证集选出的车辆-only 参考候选，但坏样本图仍有明显多段和反向响应失败。因此这里不训练新模型，只复查它在 test 集上的失败类型。

## 输入和边界

- 目标轨道：`B_response3s_strict_core`。
- 目标模型：`rbf_kernel_ridge_context_no_subject`。
- 目标 split：`test`，共 40 个样本。
- 只使用 clean-task 逐样本指标表和响应任务 manifest；不使用生理、脑电、连续风格、驾驶员 ID 或服务器。

## 核心发现

- high-RMSE top20% 阈值：sample RMSE >= 0.657000，对应 8 个样本。
- 全部 B test 样本 mean RMSE=0.476061，median RMSE=0.432152。
- high-RMSE top20% 样本 mean RMSE=0.866373，mean GT peak=1.623221。
- top 坏样本的主要失败类型计数：{"not_high_rmse_top20": 4, "wrong_side": 3, "reversal_structure_mismatch": 3, "large_response_missed": 2}

## 失败标记汇总

| flag | flag_cn | overall_count | overall_rate | high_rmse_top20_count | high_rmse_top20_rate |
| --- | --- | --- | --- | --- | --- |
| high_rmse_top20_flag | RMSE最高20% | 8 | 0.200 | 8 | 1.000 |
| wrong_side_flag | 主峰错侧 | 9 | 0.225 | 3 | 0.375 |
| severe_amp_under_flag | 严重幅值不足 | 5 | 0.125 | 3 | 0.375 |
| large_response_missed_flag | 大幅响应漏召回 | 2 | 0.050 | 2 | 0.250 |
| tail_drift_flag | 尾段漂移/未回正 | 2 | 0.050 | 0 | 0.000 |
| zero_crossing_mismatch_flag | 零线穿越错误 | 3 | 0.075 | 2 | 0.250 |
| reversal_mismatch_flag | 反向修正计数不匹配 | 40 | 1.000 | 8 | 1.000 |
| multi_segment_mismatch_flag | 多段修正结构不匹配 | 1 | 0.025 | 0 | 0.000 |
| peak_time_large_error_flag | 峰值时间误差大 | 9 | 0.225 | 4 | 0.500 |
| onset_delay_large_error_flag | 启动延迟误差大 | 7 | 0.175 | 2 | 0.250 |
| amplitude_large_error_flag | 峰值幅值误差大 | 10 | 0.250 | 4 | 0.500 |

## 结论

B 轨道 RBF KRR 可以作为当前车辆-only 参考候选继续复查，但它还不能说明车辆历史已经充分解决失稳响应预测。最明显的剩余问题不是单一 RMSE，而是反向修正、多段修正、幅值/方向和尾段回正的组合错误。下一步应优先做结构化车辆-only 响应分解，而不是直接进入连续风格或生理有效性结论。

## 推荐查看

1. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_clean_task_bad_sample_review_v0_1\tables\b_track_rbf_bad_sample_table.csv`
2. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_clean_task_bad_sample_review_v0_1\tables\b_track_rbf_top_bad_samples.csv`
3. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_clean_task_bad_sample_review_v0_1\figures\b_track_rbf_failure_flag_rates.png`
4. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_clean_task_bad_sample_review_v0_1\figures\b_track_rbf_top_bad_rmse.png`
5. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_clean_task_bad_sample_review_v0_1\figures\b_track_rbf_peak_amp_scatter.png`
