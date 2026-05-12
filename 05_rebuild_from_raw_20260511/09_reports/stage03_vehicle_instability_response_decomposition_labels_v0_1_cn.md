# 阶段 3：车辆-only 响应分解标签 v0.1

## 为什么做

B 轨道 RBF KRR 的坏样本复查显示，单纯轨迹回归仍无法处理反向修正和多段修正。下一步结构化车辆-only 模型需要先有稳定的方向、幅值、峰值时间、启动延迟、尾段状态和响应形态目标，所以本轮只从标签轨迹生成响应分解标签，不训练新模型。

## 输入与无泄漏边界

- 输入：`sample_response_task_manifest.csv`、`pre2_label2_old_main.npz`、`pre3_label3_response_coverage.npz`。
- 标签来自事件后方向盘轨迹，只能作为训练目标、辅助任务目标和评估分层，不能作为模型输入、split 条件、标准化条件或风格/生理特征。
- 大幅响应、困难响应和小响应阈值只在每个轨道的 session-level train split 上拟合，然后应用到 val/test。
- 本轮未使用生理、脑电、连续风格、驾驶员 ID、服务器或服务器密码文件。

## 轨道汇总

| track_id | n_samples | subject_n | train_n | val_n | test_n | mean_peak_abs | median_peak_abs | large_response_rate | difficult_response_rate | needs_structure_head_rate | late_peak_rate | unsettled_tail_rate | multi_correction_rate | reverse_or_multi_rate | positive_direction_rate | negative_direction_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_instant2s_core | 84 | 16 | 62 | 10 | 12 | 0.4666 | 0.2545 | 0.2738 | 0.2024 | 1.0000 | 0.3095 | 0.3333 | 0.5595 | 0.9167 | 0.5714 | 0.4286 |
| B_response3s_strict_core | 270 | 18 | 188 | 42 | 40 | 1.1784 | 1.1817 | 0.2407 | 0.1963 | 1.0000 | 0.2185 | 0.0704 | 0.9296 | 0.9889 | 0.4778 | 0.5222 |

## 阈值表

| track_id | n_samples | train_n | val_n | test_n | large_response_threshold_train_p75 | difficult_response_threshold_train_p80 | small_response_threshold_train_max_p25_015 | threshold_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_instant2s_core | 84 | 62 | 10 | 12 | 0.5007 | 0.8782 | 0.1500 | fit on session-level train split within each track |
| B_response3s_strict_core | 270 | 188 | 42 | 40 | 1.6335 | 1.7468 | 0.6605 | fit on session-level train split within each track |

## B 轨道重点结论

- B 轨道共有 270 个样本，train/val/test=188/42/40。
- B 轨道平均主峰幅值=1.1784，大幅响应比例=0.2407，需要结构化 head 的比例=1.0000。
- B 轨道 computed multi-correction 比例=0.9296，reverse/multi 合计比例=0.9889。
- B 轨道正向比例=0.4778，负向比例=0.5222。

## A 轨道处理方式

- A 轨道只有 84 个样本，test 只有 12 个；保留响应分解标签，但只作为即时响应诊断，不作为主线泛化结论。

## 下一步

用这些标签做车辆-only 响应分解模型：先预测方向、幅值桶、峰值时间桶、启动延迟桶、响应形态和尾段状态，再比较关键点+残差轨迹是否能改善 B 轨道坏样本。仍然不能进入连续风格或生理有效性结论。

## 产物

- 样本标签表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_response_decomposition_labels_v0_1\tables\response_decomposition_sample_labels.csv`
- 轨道汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_response_decomposition_labels_v0_1\tables\response_decomposition_track_summary.csv`
- 响应形态汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_response_decomposition_labels_v0_1\tables\response_decomposition_morphology_summary.csv`
- 图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_response_decomposition_labels_v0_1\figures\response_decomposition_morphology_counts.png`
- 图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_response_decomposition_labels_v0_1\figures\response_decomposition_peak_time_amp_scatter.png`
- 图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_response_decomposition_labels_v0_1\figures\b_track_mean_gt_trajectories_by_morphology.png`
