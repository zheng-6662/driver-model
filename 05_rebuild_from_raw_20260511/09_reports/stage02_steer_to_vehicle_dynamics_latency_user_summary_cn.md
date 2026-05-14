# 方向盘动作到车辆动态响应时间差审计（用户查看版）

## 这次为什么做

你提出了一个关键判断：如果所有事件本质上都是驾驶员主动打方向盘引起的，那么也许可以把“方向盘刚开始变化、车辆还没有明显侧倾/横摆/横向加速度响应”的这段时间作为输入，再预测后续方向盘轨迹或车辆动态变化。

所以这次没有训练模型，而是先审计时间关系：

- 方向盘什么时候开始明显变化；
- 横向加速度、横摆角速度、侧倾角速度、侧倾角什么时候开始明显变化；
- 二者之间是否有足够时间差；
- 0.2 秒和 0.5 秒早期输入窗口是否现实。

## 审计对象

- 使用 v0.6 事件候选表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\episode_first_event_v0_6\tables\episode_candidates_v0_6.csv`
- 共审计样本：908 个
- 成功得到方向盘与车辆动态时间差的样本：193 个

## 核心结果

- 方向盘到第一类车辆动态响应的时间差中位数：0.000 秒
- 平均时间差：-0.249 秒
- 方向盘至少提前 0.2 秒的比例：9.3%
- 方向盘至少提前 0.5 秒的比例：6.2%
- 方向盘和车辆动态几乎同步（绝对差 < 0.2 秒）的比例：62.7%
- 车辆动态明显早于方向盘（车辆至少早 0.2 秒）的比例：26.9%

## 和 v0.6 旧检测字段交叉验证

v0.6 表里原本也有 `t_steer_onset` 和 `t_dyn_onset` 字段。用旧字段直接计算时：

- 可计算样本：567 个
- 中位时间差：-0.215 秒
- 方向盘至少提前 0.2 秒的比例：7.8%
- 车辆动态至少早于方向盘 0.2 秒的比例：50.8%

也就是说，旧字段和这次重新检测虽然具体起点不完全一致，但方向上是一致的：目前并不支持“多数样本中方向盘先明显动作，然后隔较长时间车辆才侧倾/横摆”的假设。

## 第一版核心干净样本的情况

v0.6 最干净核心训练候选共 19 个，其中这次能计算明确时间差的有 7 个：

- 核心样本中位时间差：-0.255 秒
- 核心样本方向盘至少提前 0.2 秒的比例：0.0%

这说明如果只看第一版最干净样本，也不能直接把任务改成“方向盘先动较长时间后预测车辆侧倾”。

## 分类统计

- 起点都不清楚：705 个，占 77.6%
- 方向盘略早但不足0.2秒：70 个，占 7.7%
- 车辆动态明显早于方向盘：52 个，占 5.7%
- 车辆动态略早，几乎同步：40 个，占 4.4%
- 方向盘起点左截断：13 个，占 1.4%
- 方向盘先动，提前量>=0.5秒：12 个，占 1.3%
- 有车辆动态但无明确方向盘起点：9 个，占 1.0%
- 方向盘先动，提前量0.2-0.5秒：6 个，占 0.7%
- NA：1 个，占 0.1%

## 目前判断

大多数样本没有稳定的 0.2 秒以上提前量，不能直接假设“方向盘先动很久后车辆才侧倾”。

更具体地说，如果目标是“预测车辆什么时候侧倾/横摆/横向动态增强”，那么必须只挑方向盘确实提前的那一类样本；如果目标仍然是“预测后续方向盘轨迹”，那么方向盘已经开始变化后的 0.2 秒输入是可行的，但任务定义就变成了“早期动作后预测剩余轨迹”，不再是“事件发生前预测完整方向盘响应”。

## 对后续事件筛选的影响

建议把样本分成三类，而不是混在一起：

1. 方向盘明显先动：可以尝试“方向盘早期动作 → 后续车辆动态/剩余方向盘轨迹”。
2. 几乎同步：更适合做“动作发生后的短时延续预测”，不适合作为侧倾前预警。
3. 车辆动态先出现：更像“车辆扰动后驾驶员纠偏”，不应和主动打方向样本混训。

## 推荐你优先看

- 总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_summary_v0_1.csv`
- 分组表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_by_bucket_module_v0_1.csv`
- 分位数表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_quantiles_v0_1.csv`
- 直方图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\figures\steer_to_dynamics_latency_histogram_v0_1.png`
- 代表性复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`

## 结论边界

这次是信号时间差审计，不是模型结果。它只能回答“这种任务设定有没有时间基础”，不能直接证明模型一定能预测得更好。下一步如果要训练，也应该按上述三类样本分别训练或分别评估。
