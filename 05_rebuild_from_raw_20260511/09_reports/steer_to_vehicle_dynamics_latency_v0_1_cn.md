# 方向盘到车辆动态时间差审计 v0.1

## 方法

对每个 v0.6 episode，在最终训练锚点附近取 `t0-2s` 到 `t0+4s` 搜索窗口，并用 `t0-4s` 到 `t0-2s` 作为局部基线。重新检测：

- 方向盘起点：方向盘角偏离局部基线，或方向盘角速度显著升高且随后 0.5 秒内幅值确实变大；
- 横向加速度起点；
- 横摆角速度起点；
- 侧倾角速度起点；
- 侧倾角起点。

时间差定义为：

`第一类车辆动态起点 - 方向盘起点`

正值表示方向盘先动；负值表示车辆动态先出现。

## 输出文件

- 明细表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_events_v0_1.csv`
- 汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_summary_v0_1.csv`
- 分组表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_by_bucket_module_v0_1.csv`
- 分位数表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\steer_to_dynamics_latency_quantiles_v0_1.csv`
- 复核图索引：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\steer_to_vehicle_dynamics_latency_v0_1\tables\latency_review_panel_index_v0_1.csv`

## 关键数字

- 总样本：908
- 有有效时间差：193
- 中位时间差：0.0000 秒
- 平均时间差：-0.2491 秒
- `gap >= 0.2s`：9.3%
- `gap >= 0.5s`：6.2%
- `abs(gap) < 0.2s`：62.7%
- `vehicle first <= -0.2s`：26.9%

## 旧字段交叉验证

- 旧字段可计算样本：567
- 旧字段中位时间差：-0.2150 秒
- 旧字段 `gap >= 0.2s`：7.8%
- 旧字段 `vehicle first <= -0.2s`：50.8%

## 核心干净样本

- 核心样本总数：19
- 核心样本有效时间差数：7
- 核心样本中位时间差：-0.2550 秒
- 核心样本 `gap >= 0.2s`：0.0%

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

## 技术注意

1. 该审计没有使用未来方向盘峰值来定义方向盘起点，只用局部偏离和角速度变化检测起点。
2. 车辆动态响应同时看横向加速度、横摆角速度、侧倾角速度和侧倾角。
3. 如果搜索窗口一开始方向盘已经变化，会标记为“方向盘起点左截断”，这种样本不能精确判断提前量。
4. 该结果用于判断任务设定，不直接作为最终训练样本纳入标准。
