# 阶段 3：复发坏样本失败来源归因 v0.1

生成时间：2026-05-12

## 目的

对复发坏样本 Top 12 做规则化归因，区分下一步应该先回到阶段 2 修锚点/窗口/原始信号，还是可以进入阶段 3 的结构化车辆模型。此步骤只使用车辆-only 结果和原始车辆片段，不使用生理、脑电、连续风格或驾驶员 ID 作为模型输入。

## 输入

- 曲线图索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_figure_index.csv`
- 模型逐事件误差表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_model_error_table.csv`
- 正式样本清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 原始车辆 CSV：按样本清单中的路径只读局部片段。

## 规则

- 标签窗口可能偏短：GT 峰值接近标签末端、尾段仍未回正，或事件持续时间超过标签窗口。
- 锚点可能偏晚：锚点前 0.75 秒内方向盘已经有明显响应，或非方向盘车辆动力学在锚点前已经很活跃。
- 原始信号需复核：核心车辆信号有效点比例过低。
- 车辆-only 结构不足：多数候选严重幅值不足，并且 GT 有反向/多段结构但模型反向修正计数基本不匹配。

## 主要结果

- Top 12 中，`sample_rule_or_raw_signal_review` 数量=10。
- Top 12 中，`vehicle_only_model_structure_gap` 数量=1。
- 这说明不能直接跳到风格/生理阶段；下一步仍应先完成车辆-only 错误来源清理和结构化基线设计。

## 归因表

| recurrence_rank | subject | config_id | primary_attribution | gt_peak_abs | gt_peak_time_s | gt_tail_over_peak | severe_amp_under_rate | reversal_exact_rate | reason_cn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hzh | session_pre3 | sample_rule_or_raw_signal_review | 4.153 | 2.990 | 1.000 | 1.000 | 0.000 | 标签窗口/事件持续时间需要复核；锚点附近事件可能已经开始；车辆-only 候选共同漏幅值/反向修正 |
| 2 | gzj | subject_main | sample_rule_or_raw_signal_review | 2.595 | 1.625 | 0.915 | 0.800 | 0.000 | 标签窗口/事件持续时间需要复核；车辆-only 候选共同漏幅值/反向修正 |
| 3 | gzj | subject_main | sample_rule_or_raw_signal_review | 2.132 | 1.180 | 0.896 | 1.000 | 0.000 | 标签窗口/事件持续时间需要复核；车辆-only 候选共同漏幅值/反向修正 |
| 4 | gzj | subject_main | vehicle_only_model_structure_gap | 1.708 | 1.035 | 0.180 | 0.400 | 0.000 | 车辆-only 候选共同漏幅值/反向修正 |
| 5 | gf | random_main | sample_rule_or_raw_signal_review | 0.920 | 0.910 | 0.533 | 0.000 | 0.000 | 标签窗口/事件持续时间需要复核；锚点附近事件可能已经开始 |
| 6 | hzh | session_pre1 | sample_rule_or_raw_signal_review | 1.641 | 0.945 | 0.468 | 1.000 | 0.000 | 标签窗口/事件持续时间需要复核；多个车辆-only 候选错侧 |
| 7 | hzh | session_pre1 | sample_rule_or_raw_signal_review | 2.461 | 1.305 | 0.819 | 1.000 | 0.000 | 标签窗口/事件持续时间需要复核；锚点附近事件可能已经开始；车辆-only 候选共同漏幅值/反向修正 |
| 8 | tyy | random_main | sample_rule_or_raw_signal_review | 2.334 | 1.915 | 0.991 | 0.200 | 0.200 | 标签窗口/事件持续时间需要复核；车辆-only 候选共同漏幅值/反向修正 |
| 9 | hzh | session_pre3 | hard_vehicle_only_case | 1.678 | 1.745 | 0.198 | 0.200 | 0.000 | 未发现明显锚点/窗口/质量问题，但仍是高误差样本 |
| 10 | zxy | session_pre3 | sample_rule_or_raw_signal_review | 2.708 | 1.825 | 0.722 | 0.800 | 0.000 | 标签窗口/事件持续时间需要复核；车辆-only 候选共同漏幅值/反向修正；多个车辆-only 候选错侧 |
| 11 | zxy | session_pre3 | sample_rule_or_raw_signal_review | 3.333 | 2.700 | 0.987 | 1.000 | 0.000 | 标签窗口/事件持续时间需要复核；车辆-only 候选共同漏幅值/反向修正；多个车辆-only 候选错侧 |
| 12 | hzh | random_main | sample_rule_or_raw_signal_review | 2.556 | 1.600 | 0.941 | 1.000 | 0.200 | 标签窗口/事件持续时间需要复核；车辆-only 候选共同漏幅值/反向修正 |

## 产物

- 归因明细表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_failure_attribution_table.csv`
- 归因旗标统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_failure_flag_counts.csv`
- 主归因统计：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_primary_attribution_counts.csv`
- 归因旗标热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/figures/bad_event_failure_attribution_flags.png`
- 主归因计数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/figures/bad_event_primary_attribution_counts.png`

## 下一步

先复核被标记为 `sample_rule_or_raw_signal_review` 的事件。如果主要问题来自窗口太短或锚点偏晚，应回到阶段 2 修样本规则；如果复核后这些事件仍可信，再把 `vehicle_only_model_structure_gap` 作为下一版结构化车辆模型的目标。
