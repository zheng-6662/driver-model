# 阶段 2 事件锚点与样本清单重建总结

更新时间：2026-05-12

## 做了什么

1. 读取阶段 1 的原始文件清单、时间连续性、模态重叠和质量报告。
2. 匹配旧流程 `events_v400_context.csv`，只作为历史参考，不直接继承为最终真相。
3. 检索道路设计记录，生成 `road_design_inventory.csv`，确认存在道路中心线、曲率和模块信息。
4. 从原始车辆信号重新生成两类候选：道路曲率进入候选和车辆动态响应 onset 候选。
5. 为每个候选锚点生成 4 套窗口配置，写入 `samples_master.csv/jsonl`。
6. 生成随机、session-level、subject-level 三类 split 表，并明确 train-only 标准化规则。

## 主要数量

- 候选事件总数：11619
- 样本窗口行数：46476
- 道路设计目录文件数：49
- 含曲率信息的道路设计 CSV：8
- source 计数：

anchor_source
old_v400_context_trigger_idx    6247
raw_vehicle_dynamic_onset       5013
raw_road_curvature_onset         359

## 风险判断

- `old_v400_context_trigger_idx`：来自旧处理事件表，能对照历史结果，但不能直接当作新流程真相。
- `raw_vehicle_dynamic_onset`：从方向盘、横摆、横向加速度等响应导出，可能已经接近或进入标签响应，不能用于证明事件触发预测无泄漏。
- `raw_road_curvature_onset`：来自原始道路曲率变化，泄漏风险较低；道路设计文件证明项目中有道路几何记录，但本轮还没有完成逐时间戳投影，所以它仍是候选锚点，不是最终道路真值。
- 任何 `input_end_rel_s > 0` 的窗口都属于早期观察后预测剩余轨迹，不能和事件发生时预测完整未来混淆。

## 是否可以进入阶段 3

可以进入阶段 3 的前置准备，但只能先做无学习/强车辆基线的保守版本：优先使用 `raw_road_curvature_onset` 且 `input_end_rel_s<=0` 的样本。旧 v400 和 raw dynamic 样本必须作为对照或上限分析，不能作为最终无泄漏主线。
