# v2.0 训练任务重定义：车辆-only 实验报告

## 这次为什么做

本轮按 `gptpro_answer/goal1.txt` 执行：不再只把任务理解成固定窗口方向盘轨迹预测，而是先把 v2.0 episode 转成可分任务、可掩码、可复核的车辆-only 训练任务。当前阶段仍不加入连续驾驶风格、生理数据或脑电。

## 新版 manifest 和样本利用

- v2.0 episode 总数：1766。
- training_role 分布：`{"main_train": 746, "control": 319, "aux_train": 285, "curve_task": 238, "excluded_slope_or_offroad": 162, "review_need_manual_check": 16}`。
- episode_type 分布：`{"weak_response_control": 409, "noncurve_extreme": 337, "normal_control": 319, "review_candidate": 301, "curve_normal_or_weak": 208, "excluded_slope_or_offroad": 162, "curve_abnormal_roll": 30}`。
- 完整 2s 输入样本数：1734。
- 完整 5s 输出样本数：1749。
- 窗口不完整但满足 1s 输入 + 核心标签条件、可用 mask 训练的样本数：9。

## 实验结果

| experiment | name_cn | train | val | test | window_incomplete_used | steering_rmse | wrong_side_rate | severe_under_amplitude_rate | large_response_recall | response_type_macro_f1 | curve_type_macro_f1 | keypoint_steering_peak_time_mae | keypoint_roll_peak_value_mae | figure_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_fixed_steering_baseline | 旧固定窗口方向盘基线 | 625.0000 | 220.0000 | 115.0000 | 0.0000 | 0.3564 | 0.3750 | 0.3750 | 0.6250 | NA | NA | NA | NA | 100.0000 |
| E1_fixed_multitask_vehicle | 固定窗口多输出车辆基线 | 625.0000 | 220.0000 | 115.0000 | 0.0000 | 0.3359 | 0.3750 | 0.5000 | 0.6250 | 0.7692 | NA | NA | NA | 101.0000 |
| E2_masked_multihorizon_keypoint | 掩码多输出多时域模型 | 803.0000 | 258.0000 | 178.0000 | 9.0000 | 0.3658 | 0.3000 | 0.3000 | 0.7000 | 0.4613 | NA | 0.9579 | 0.0387 | 125.0000 |
| E3_noncurve_response_branch | 非弯道响应类型辅助模型 | 791.0000 | 237.0000 | 190.0000 | 9.0000 | 0.2925 | 0.2500 | 0.7500 | 0.5833 | 0.7392 | NA | 0.8873 | 0.0331 | 100.0000 |
| E4_curve_specialized | 弯道专门模型 | 149.0000 | 60.0000 | 29.0000 | 0.0000 | 0.3429 | 0.3333 | 0.3333 | 0.6667 | NA | 0.4444 | 0.9643 | 0.0344 | 85.0000 |
| E5A_train_candidates_only | 分层纳入 A：只用训练候选 | 625.0000 | 220.0000 | 116.0000 | 1.0000 | 0.3385 | 0.5000 | 0.6250 | 0.6250 | 0.7698 | NA | 0.8490 | 0.0380 | 104.0000 |
| E5B_train_plus_all_review | 分层纳入 B：训练候选 + 全部待复核 | 811.0000 | 258.0000 | 185.0000 | 9.0000 | 0.3440 | 0.4000 | 0.6000 | 0.6000 | 0.3710 | NA | 0.9626 | 0.0384 | 136.0000 |
| E5C_train_plus_stratified_review | 分层纳入 C：训练候选 + 分层干净待复核 | 803.0000 | 258.0000 | 178.0000 | 9.0000 | 0.3546 | 0.7000 | 0.4000 | 0.7000 | 0.5517 | NA | 0.9348 | 0.0426 | 130.0000 |

## 初步判断

- E0 固定窗口只使用 `960` 个样本；E2 掩码多输出使用 `1239` 个样本。这个差异直接说明硬性 2s+5s 窗口会丢掉一批 episode。
- E5 对比中：A 只用训练候选 test steering RMSE=0.3385；B 加全部待复核为 0.3440；C 加分层待复核为 0.3546。是否继续纳入待复核，应同时看 RMSE、错侧率、严重幅值不足率和预测图。
- 弯道 E4 已单独输出，后续不能再只用方向盘 RMSE 判断弯道任务好坏，应重点看 roll/roll_rate/ay/yaw/speed/brake 图。
- 目前这批实验的意义是稳定 vehicle-only 任务定义；它不是连续风格或生理数据有效性的证据。

## 对 goal1 关键问题的回答

1. 固定窗口 steering-only 不适合直接作为唯一主任务：它样本利用率较低，而且只回答方向盘，不回答速度、制动和车辆姿态。
2. masked multi-horizon 可以更充分利用样本，但是否升级为主线不能只看 RMSE，还要看错侧率、严重幅值不足率、关键点误差和预测图。
3. 多输出任务更符合极限工况驾驶员模型，因为它把方向盘、车速、制动、横摆、横滚放在同一响应里看。
4. 非弯道建议保留 response_type 辅助任务，因为 E3 单独训练后整体 steering RMSE 低于混合任务，但仍需看预测图确认物理意义。
5. 弯道必须单独建模；E4 输出了弯道预测图和 curve_type 指标，不能再把正常过弯和非弯道极限事件混成一个方向盘回归。
6. 待复核样本有价值但不能全量无脑加入：E5B 加全部待复核虽然 RMSE 下降，但错侧率和严重幅值不足率明显恶化；E5C 分层纳入更稳。
7. 当前 slope/offroad/高度异常样本只统计和保留，不进入 E0-E5 主训练；后续如要研究路边恢复，应单独开任务。
8. 下一步不应马上加入连续风格和生理数据；应先人工看 E2/E3/E4/E5C 的预测图，确认方向、幅值、速度、制动和姿态曲线是否更合理。

## 产物位置

- 新版 manifest：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\manifests`
- 中间数组：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\record_episode_v2_task_redesign\arrays`
- 实验输出：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\outputs`
- 最终报告：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_goal1_v2_task_redesign\outputs\final_task_redesign_report.md`