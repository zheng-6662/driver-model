# 车辆响应锚点前方向盘动作重新筛选 v0.2（用户查看版）

## 这次为什么重新筛选

你提出的核心判断是：实际事件大多不是“车辆先侧倾、驾驶员再纠偏”，而是“驾驶员主动打方向盘，车辆随后出现横向动态、横摆、横滚或侧倾增强”。因此，不能再直接沿用旧的 v0.6 样本分类，也不能只用方向盘动作池。

本次重新筛选的目标是：

- 先从原始车辆动态候选里找车辆响应锚点；
- 再检查车辆响应锚点前 2 秒内是否存在明确方向盘启动；
- 再判断车辆响应后是否还有可预测的回正、反打或纠正轨迹；
- 最后把真正适合“侧倾/姿态响应前早期方向盘信息预测后续纠正”的样本筛出来。

## 这次用的输入

- 输入候选表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_instability_all_raw_rescreen_v0_1\tables\all_raw_vehicle_instability_candidates_v0_1.csv`
- 候选数量：1991 个
- 这些候选来自原始车辆 CSV 的横向加速度、横摆、横滚等非方向盘车辆动态扫描，比旧 v0.6 高置信表更宽。

## 核心筛选原则

这次不是“阈值过了就算侧倾失稳”。核心样本至少要同时满足：

1. 车辆动态不是单一信号异常，而是横向加速度、横摆角速度、横滚角速度、横滚角中至少多个信号有证据；
2. 要有横滚相关证据，才进入 P1 侧倾/姿态响应核心样本；
3. 方向盘启动要落在车辆响应锚点前 2 秒内；
4. 车辆响应锚点后还要有可预测的方向盘回正、反打或继续纠正轨迹；
5. 横向偏移存在明显坐标跳变的样本先排除；
6. 正常弯道平滑转向不直接当作侧倾失稳样本。

## 筛选结果

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

其中：

- P1 最干净核心侧倾/姿态响应样本：61 个
- P2 最干净次级横向动态响应样本：1 个
- 几乎同步样本：283 个
- 连续/上下文场景复核样本：77 个
- 需要人工复核或暂缓样本：1614 个

## 时间差结果

在能够计算方向盘启动和车辆响应时间差的样本里：

- 方向盘至少领先车辆响应 0.2 秒的比例：55.6%
- 方向盘至少领先车辆响应 0.5 秒的比例：48.6%

这个数字用于判断“车辆响应锚点前 2 秒是否真的包含早期方向盘动作”。如果 P1/P2 复核图能确认方向盘确实在车辆响应前启动，那么侧倾锚点路线可以继续；如果多数只是同步或检测不清，就不能声称有明显提前量。

## 推荐优先查看

- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_candidates_v0_2.csv`
- P1 最干净核心样本表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\primary_roll_presteer_events_P1_v0_2.csv`
- P2 最干净次级样本表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\secondary_lateral_presteer_events_P2_v0_2.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_summary_v0_2.csv`
- 分场景表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_by_module_v0_2.csv`
- 时间差分位数表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_latency_quantiles_v0_2.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\tables\vehicle_response_presteer_review_panel_index_v0_2.csv`

图：
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\figures\vehicle_response_presteer_category_counts_v0_2.png`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_response_presteer_rescreen_v0_2\figures\vehicle_minus_steer_lead_histogram_v0_2.png`

## 当前结论边界

这一步仍然不是模型训练结果。它只回答：能不能筛出一批“方向盘先动、车辆随后侧倾/横向动态增强、后续仍有纠正轨迹”的样本。

如果人工复核 P1 图后基本认可，下一步才适合基于这些样本构建新预测任务：

> 输入车辆响应锚点前 2 秒的车辆状态和方向盘早期动作，预测车辆响应锚点后的方向盘纠正轨迹。
