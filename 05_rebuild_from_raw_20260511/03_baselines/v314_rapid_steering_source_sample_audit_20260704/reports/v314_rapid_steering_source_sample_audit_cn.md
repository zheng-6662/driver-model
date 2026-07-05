# 第314版方向盘快转来源抽样排查

## 结论

- 本轮不训练模型，也不做逐个式人工复核；只检查样本是否有方向盘快速转动来源证据。
- 全体事件数：`1167`。
- 当前0到2秒窗口内有快转证据：`1083`；当前窗口快转证据不足或来源错位：`84`。
- 第309版严重错误样本中，当前窗口快转证据不足或来源错位：`4/37`。
- 用户截图样本中，当前窗口快转证据不足或来源错位：`1/5`。
- 本轮固定快转阈值：方向盘转动速度峰值 `>= 0.80` 且当前方向盘变化峰值 `>= 0.35`。

## 主要输出

- 全量排查表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v314_rapid_steering_source_sample_audit_20260704\tables\v314_rapid_steering_source_audit_all_delay0.csv`
- 抽样排查表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v314_rapid_steering_source_sample_audit_20260704\tables\v314_rapid_steering_source_sample_cases.csv`
- 来源分级汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v314_rapid_steering_source_sample_audit_20260704\tables\v314_source_category_summary.csv`
- 粗场景交叉汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v314_rapid_steering_source_sample_audit_20260704\tables\v314_scene_by_source_category_summary.csv`
- 抽样图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v314_rapid_steering_source_sample_audit_20260704\figures\sample_cases`

## 来源分级汇总

| source_category | source_category_cn | suggested_sample_action_cn | event_n | severe_n | screenshot_n |
| --- | --- | --- | --- | --- | --- |
| current_and_late_fast_steer | 当前和后续都有方向盘快转 | 样本来源成立，转入模型幅值/相位诊断 | 1041 | 32 | 4 |
| late_fast_steer_not_current_window | 当前窗口不明显，后续才方向盘快转 | 优先检查锚点或预测窗口 | 71 | 4 | 1 |
| current_window_fast_steer_supported | 当前窗口有方向盘快转证据 | 样本来源成立，转入模型幅值/相位诊断 | 42 | 1 | 0 |
| no_clear_fast_steer_evidence | 全程方向盘快转证据弱 | 候选剔除或重锚定 | 7 | 0 | 0 |
| anchor_after_fast_steer | 锚点前已经方向盘快转 | 优先检查锚点或预测窗口 | 6 | 0 | 0 |

## 转动速度分位数

| quantile | rate_current_peak_abs | rate_near_anchor_peak_abs | rate_pre_peak_abs | rate_late_peak_abs | rate_any_0_6_peak_abs |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 1.63709 | 0.99008 | 0.990089 | 1.25913 | 2.39015 |
| 0.25 | 2.78777 | 1.81674 | 1.75961 | 2.12296 | 3.25582 |
| 0.5 | 4.2237 | 2.93216 | 2.75445 | 3.6398 | 4.99323 |
| 0.75 | 6.40695 | 4.44106 | 3.79375 | 6.43709 | 7.95866 |
| 0.9 | 9.59419 | 6.37775 | 5.59585 | 9.89062 | 11.5566 |
| 0.95 | 11.7864 | 7.82225 | 6.76045 | 12.4313 | 14.0419 |

## 下一步建议

- 对“当前窗口不明显，后续才方向盘快转”和“锚点前已经方向盘快转”两类，不应继续当作普通训练样本硬塞给当前0到2秒预测，应进入锚点或窗口修正。
- 对“全程方向盘快转证据弱”类，应优先考虑候选剔除或重新找触发点，因为这和用户强调的样本定义不一致。
- 对“当前窗口有方向盘快转证据”但仍预测差的严重样本，才进入模型幅值、相位、极端跟随不足的训练修正。
