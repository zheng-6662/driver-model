# v232 过晚锚点重锚定候选审核包

## 目的

本包继续 v231 人工审核结论：先处理过晚锚点和目标窗口错位，再讨论模型结构。
本轮只生成重锚定候选和证据，不直接修改训练标签，不训练模型，不改 formal headline。

## 已纳入的人工边界

- `rjy_Entity_Recording_2025_09_28_20_02_20_v108_010` 已由用户人工确认锚点晚了。
- 不重启“先硬判断响应类型，再预测轨迹”的路线；该路线此前已尝试过，且存在分类错误传播。
- 不把“一次性输出多个候选轨迹”作为下一步主线；该路线此前也已尝试过，即使 best candidate 仍有偏差。

## 输出文件

- 全量打分表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\tables\v232_reanchor_candidate_all_scored.csv`
- 人工审核表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\tables\v232_reanchor_candidate_review_table.csv`
- 0.05 秒信号网格：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\tables\v232_reanchor_grid_0p05s.csv`
- 关键时刻表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\tables\v232_reanchor_key_points.csv`
- 图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\figures`
- 候选图拼接总览：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\figures\v232_reanchor_candidates_contact_sheet.png`
- ZIP 包：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v232_late_anchor_reanchor_candidates_20260624\v232_late_anchor_reanchor_candidates_pack.zip`

## 检测方法

每个样本从原始车辆 CSV 读取旧锚点前后 `-10s~+8s` 的信号，并做信号级最近非空采样。
方向盘以旧锚点前 `-8s~-6s` 的平滑中位数作为基线，计算 `steering_delta_from_baseline`。
候选新锚点定义为旧锚点前第一次持续超过阈值的方向盘变化起点，阈值为 `max(0.35, 0.18 * window_peak_abs_delta)`。
候选只作为人工审核入口，不自动生效。

## 审核优先级汇总

|rank|priority|sample_id|old_anchor_s|candidate_anchor_s|shift_s|score|pre3_peak|post03_peak|reason|
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
|1|P0_manual_confirmed_reanchor|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_010`|143.100|138.950|-4.15|9.0|4.243|1.590|用户人工确认锚点晚；旧锚点前3秒已有明显转向；旧锚点处已处于响应进程中；检测到旧锚点前的候选起点；旧锚点后3-8秒仍有更大峰值，需另查窗口/horizon|
|2|P1_high_reanchor_review|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_041`|624.900|617.100|-7.80|4.0|1.926|1.933|旧锚点前3秒已有明显转向；旧锚点处已处于响应进程中；检测到旧锚点前的候选起点|
|3|P1_high_reanchor_review|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_040`|620.900|613.100|-7.80|4.0|2.088|1.816|旧锚点前3秒已有明显转向；旧锚点处已处于响应进程中；检测到旧锚点前的候选起点|
|4|P1_high_reanchor_review|`rjy_Entity_Recording_2025_09_28_19_33_26_v108_032`|598.900|590.900|-8.00|4.0|1.662|1.711|旧锚点前3秒已有明显转向；旧锚点处已处于响应进程中；检测到旧锚点前的候选起点|
|5|P1_high_reanchor_review|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_033`|638.700|634.550|-4.15|4.0|2.585|2.166|旧锚点前3秒已有明显转向；旧锚点处已处于响应进程中；检测到旧锚点前的候选起点|
|6|P2_medium_reanchor_review|`cwh_Entity_Recording_2025_09_26_20_06_19_v108_017`|252.200|248.950|-3.25|3.0|0.694|1.315|旧锚点前3秒存在可见转向；旧锚点处已处于响应进程中；检测到旧锚点前的候选起点|
|7|P2_medium_reanchor_review|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_026`|507.200|500.350|-6.85|3.0|0.900|2.147|旧锚点前3秒存在可见转向；旧锚点处已处于响应进程中；检测到旧锚点前的候选起点|
|8|P2_medium_reanchor_review|`lx_Entity_Recording_2025_09_26_09_17_22_v108_034`|527.600|523.900|-3.70|3.0|1.059|1.436|旧锚点前3秒已有明显转向；检测到旧锚点前的候选起点|
|9|P2_medium_reanchor_review|`rjy_Entity_Recording_2025_09_28_19_33_26_v108_014`|391.200|387.100|-4.10|3.0|1.339|1.285|旧锚点前3秒已有明显转向；检测到旧锚点前的候选起点|
|10|P2_medium_reanchor_review|`cwh_Entity_Recording_2025_09_26_19_56_16_v108_021`|390.700|388.900|-1.80|3.0|0.877|0.895|旧锚点前3秒已有明显转向；检测到旧锚点前的候选起点|
|11|P2_medium_reanchor_review|`rjy_Entity_Recording_2025_09_28_20_15_42_v108_006`|180.800|176.550|-4.25|3.0|0.619|1.086|旧锚点前3秒存在可见转向；旧锚点处已处于响应进程中；检测到旧锚点前的候选起点|

## 人工审核建议

1. 先看 P0/P1：确认候选新锚点是否确实比旧锚点更接近事件起点。
2. 如果确认，填写 `human_decision=accept_reanchor`、`human_corrected_anchor_s` 和 `human_use_for_training`。
3. 如果候选过早或过晚，人工改写 `human_corrected_anchor_s`，不要直接采用算法候选。
4. 只有人工确认后的样本才允许进入下一轮 label window 重建。
5. 锚点确认无误但仍预测差的样本，才进入模型方法提升；不要把锚点晚样本混入模型失败结论。

## 后续方法边界

下一步不是继续加候选轨迹数，也不是硬响应类型分类，而是先完成重锚定候选的人工确认。
重锚定后如果仍有系统偏差，再考虑目标窗口重建、偏差校正、连续相位/延迟参数或对齐鲁棒损失。