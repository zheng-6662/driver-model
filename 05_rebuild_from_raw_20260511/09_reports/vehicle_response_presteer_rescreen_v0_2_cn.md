# 车辆响应锚点前方向盘动作重新筛选 v0.2

## 方法

本脚本从 `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_instability_all_raw_rescreen_v0_1\tables\all_raw_vehicle_instability_candidates_v0_1.csv` 的 1991 个原始车辆动态候选重新开始，不直接继承 v0.6 分类。每个候选重新读取原始车辆 CSV，并在候选锚点附近计算：

- 车辆响应锚点：横向加速度、横摆角速度、横滚角速度、横滚角的局部稳健阈值越界起点；
- 多信号车辆响应：至少两个车辆动态信号成立；
- 侧倾/姿态响应：横滚相关证据与横向/横摆证据同时成立；
- 方向盘启动：车辆响应前 2 秒内方向盘角或方向盘角速度显著离开局部基线；
- 后续纠正目标：车辆响应后方向盘存在峰值、回正、反打或持续纠正；
- 坐标连续性：横向偏移步进过大则排除。

## 输出

- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_candidates_v0_2.csv`
- P1：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\primary_roll_presteer_events_P1_v0_2.csv`
- P2：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\secondary_lateral_presteer_events_P2_v0_2.csv`
- 几乎同步：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\near_sync_events_S_v0_2.csv`
- 人工复核：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\manual_review_events_v0_2.csv`
- 排除：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\excluded_events_X_v0_2.csv`
- 汇总：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_summary_v0_2.csv`
- 分场景：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_by_module_v0_2.csv`
- 分位数：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_latency_quantiles_v0_2.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_review_panel_index_v0_2.csv`

## 类别统计

- 车辆姿态证据不足：608 个，占 30.5%
- 车辆响应前未找到明确方向盘启动：510 个，占 25.6%
- 输入窗口开始前已在转向：390 个，占 19.6%
- 方向盘和车辆响应几乎同步：283 个，占 14.2%
- 连续/上下文场景复核：77 个，占 3.9%
- 最干净核心侧倾/姿态响应样本：61 个，占 3.1%
- 窗口或信号不足：23 个，占 1.2%
- 正常弯道/平滑转向候选：16 个，占 0.8%
- 后续纠正目标不足：13 个，占 0.7%
- 坐标连续性异常：9 个，占 0.5%
- 最干净次级横向动态响应样本：1 个，占 0.1%

## 关键数量

- 总候选：1991
- P1 最干净核心样本：61
- P2 最干净次级样本：1
- 近同步样本：283
- 连续/上下文复核样本：77
- 复核/暂缓样本：1614
- 复核图数量：104

## 时间差

- 有效时间差样本：1038
- 方向盘领先 >=0.2 秒比例：55.6%
- 方向盘领先 >=0.5 秒比例：48.6%
