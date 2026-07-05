# v231 六个最差样本锚点上下文人工审核包

## 目的

这份包只回答一个问题：这些预测差的样本，错误更像是事件锚点/窗口问题，还是模型对行为响应形态和幅值没有学好。
所有锚点上下文均直接从原始车辆 CSV 按 `anchor_s` 对齐抽取；方向盘字段为原始 `zx|SteeringWheel`，另计算 `steering_delta_from_anchor = 当前方向盘 - 锚点方向盘`。
注意：原始 CSV 同一时间戳行上不同信号会有空值，因此 `v231_anchor_key_points.csv` 和 `v231_anchor_window_sparse_8s.csv` 对每个信号使用目标时刻最近的非空值，并用 `字段名__time_error_ms` 保留该值相对目标时刻的毫秒误差。`v231_anchor_window_dense_pm3s.csv` 保留原始行，不填补空值。

## 输出文件

- 元数据总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_metadata.csv`
- 窗口摘要表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_window_summary.csv`
- 信号对齐关键时刻表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_key_points.csv`，每个样本含 -8/-5/-3/-2/-1/-0.5/0/+0.5/+1/+1.5/+2/+3/+5/+8 秒。
- 信号对齐 0.1 秒稀疏窗口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_window_sparse_8s.csv`，每个样本 `anchor_s ±8s`。
- 原始 200Hz 密集窗口：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_anchor_window_dense_pm3s.csv`，每个样本 `anchor_s ±3s` 原始行。
- 锚点上下文图目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures`
- 六图拼接总览：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures\v231_all_six_anchor_context_contact_sheet.png`
- 用户反馈覆盖表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\tables\v231_user_feedback_overrides.csv`

## 六个样本概览

|rank|sample_id|anchor_s|anchor_abs_time|方向盘锚点误差(ms)|事件类型|展示模型|v225 tail RMSE|观测峰值|预测峰值|锚点上下文初判|
|---:|---|---:|---|---:|---|---|---:|---:|---:|---|
|1|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_010`|143.100|2025-09-28 20:04:43.651000|0.0|下坡弯道事件 / vehicle_strong|strict_main_pool / peak_floor_090|2.012|0.820|1.630|锚点前3秒已明显转向，需查锚点是否落在事件中段/晚锚定；主峰在锚点后3-8秒，需查目标窗口是否太短或锚点偏早|
|2|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_041`|624.900|2025-09-28 20:12:45.451000|0.0|直道事件 / extreme_peak|strict_main_pool / peak_floor_090|1.759|3.314|1.512|锚点前3秒已明显转向，需查锚点是否落在事件中段/晚锚定|
|3|`cwh_Entity_Recording_2025_09_26_20_06_19_v108_017`|252.200|2025-09-26 20:10:31.536000|0.0|下坡弯道事件 / strong_event|strict_main_pool / peak_floor_090|1.755|1.862|3.358|未见明显锚点错位，更像幅值/形态预测问题|
|4|`rjy_Entity_Recording_2025_09_28_19_51_44_v108_023`|357.800|2025-09-28 19:57:42.509000|0.0|下坡弯道事件 / strong_event|loose_main_pool / avg_joint_focus|1.676|1.897|0.976|锚点后有方向反转/多次修正，单峰平滑轨迹会吃亏|
|5|`tyy_Entity_Recording_2025_09_28_14_23_43_v108_026`|507.200|2025-09-28 14:32:10.905000|0.0|下坡弯道事件 / strong_event|loose_main_pool / avg_joint_focus|1.650|2.412|0.499|锚点后有方向反转/多次修正，单峰平滑轨迹会吃亏|
|6|`rjy_Entity_Recording_2025_09_28_20_02_20_v108_031`|549.400|2025-09-28 20:11:29.951000|0.0|下坡弯道事件 / extreme_peak|loose_main_pool / avg_joint_focus|1.641|3.212|1.184|锚点后有方向反转/多次修正，单峰平滑轨迹会吃亏|

## 2026-06-24 人工反馈修正

- `rjy_Entity_Recording_2025_09_28_20_02_20_v108_010`：用户人工确认属于锚点晚了。后续不能把它当作模型形态失败样本；它应进入锚点修正/事件起点定义问题，而不是进入模型训练难例结论。
- 关于“先判断响应类型再预测曲线”：用户指出这条路之前已经尝试过，而且硬响应类型判断本质上会产生错误传播；如果响应类型判断错，后续轨迹预测也会错。因此本报告撤回“硬前置响应类型分类”作为下一步主线。
- 下一步方法表述改为：不做硬分类前置，而是在同一个预测框架里处理多模态响应和时间错位，例如软门控/概率混合、多假设轨迹输出、连续相位或延迟参数、锚点偏移校正、以及能容忍响应延迟的轨迹损失。重点是减少“分类错即全错”的结构性风险。

## 2026-06-24 第二轮人工反馈修正

- 过晚锚点不能只停留在标记层面，需要进一步做重锚定，重新定义事件起点和预测窗口。
- 一次性输出多个候选轨迹也已经尝试过，效果不好；即使选择其中最好的候选轨迹，仍然存在偏差。因此不能再把简单多候选轨迹输出作为下一步主线。
- 方法优先级改为：先做晚锚点重锚定和目标窗口重建；再看锚点无误样本中的系统偏差。模型结构层面可以考虑偏差校正、连续延迟/相位参数或对齐鲁棒损失，但前提是目标锚点先对齐。

## 人工审核时优先看的列

- `nearest_time_error_ms`：`anchor_s` 本身对原始时间戳的误差；这六个样本均为 0ms。
- `steering_anchor_time_error_ms`：方向盘锚点值相对目标锚点的误差；因为原始行有空值，这比只看精确行更可靠。
- `pre_3_0_peak_abs_delta`：锚点前 3 秒是否已经大幅转向；如果很大，优先怀疑锚点落在事件中段或事件起点定义不稳。
- `post_0_3_peak_abs_delta` 与 `post_3_8_peak_abs_delta`：主峰在锚点后 0-3 秒还是 3-8 秒；如果后者更大，可能是锚点偏早或标签窗口太短。
- `post_0_8_sign_changes_delta`：锚点后是否发生方向反转/多次修正；这类样本不适合只用单峰幅值解释。

## 每个样本的图

- #1 `rjy_Entity_Recording_2025_09_28_20_02_20_v108_010`：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures\01_rjy_Entity_Recording_2025_09_28_20_02_20_v108_010_anchor_context.png`
- #2 `rjy_Entity_Recording_2025_09_28_20_02_20_v108_041`：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures\02_rjy_Entity_Recording_2025_09_28_20_02_20_v108_041_anchor_context.png`
- #3 `cwh_Entity_Recording_2025_09_26_20_06_19_v108_017`：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures\03_cwh_Entity_Recording_2025_09_26_20_06_19_v108_017_anchor_context.png`
- #4 `rjy_Entity_Recording_2025_09_28_19_51_44_v108_023`：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures\04_rjy_Entity_Recording_2025_09_28_19_51_44_v108_023_anchor_context.png`
- #5 `tyy_Entity_Recording_2025_09_28_14_23_43_v108_026`：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures\05_tyy_Entity_Recording_2025_09_28_14_23_43_v108_026_anchor_context.png`
- #6 `rjy_Entity_Recording_2025_09_28_20_02_20_v108_031`：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v231_worst_case_anchor_context_20260624\figures\06_rjy_Entity_Recording_2025_09_28_20_02_20_v108_031_anchor_context.png`

## 初步结论

这六个样本更适合当作方法提升的诊断集，而不是写成失败机制论文材料。结合 2026-06-24 人工反馈，应分成三类处理：

1. 锚点前已经动起来的样本：需要改事件起点定义，或给模型显式输入 pre-anchor 动态。
2. 锚点后 3-8 秒才到主峰的样本：需要检查目标窗口长度、预测 horizon，或做可变延迟响应建模。
3. 锚点后存在反转/多次修正的样本：不能简单回到“先判断响应类型”的硬分类路线，也不能把简单多候选轨迹输出作为主线；更优先的是确认锚点和目标窗口无误，然后再做偏差校正、连续延迟/相位参数或对齐鲁棒损失。

人工审核应先确认前两类是否是真锚点/窗口问题。已人工确认锚点晚的样本应从模型难例中剥离；锚点无误但仍预测差的样本，才进入下一轮方法提升的困难样本集。
