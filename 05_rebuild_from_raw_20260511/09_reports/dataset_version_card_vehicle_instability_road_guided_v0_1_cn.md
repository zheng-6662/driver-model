# 数据版本卡：vehicle_instability_road_guided_v0_1

生成时间：2026-05-12 15:52

## 数据版本定位

`vehicle_instability_road_guided_v0_1` 是阶段 2 的车辆失稳事件候选筛选版本。它不是人工真值版本，也不是模型训练结果；它的作用是替代“让用户逐条看完 1227 个候选”的人工流程，先给出一个可追溯、可解释、可继续处理的失稳事件清单。

## 事件定义

主事件是车辆失稳候选，不是弯道候选。

事件锚点优先来自原始车辆动态中的非方向盘信号：

- `ay`：横向加速度异常；
- `roll_rate`：横滚速率异常。

方向盘相关信号不用于定义失稳开始点。方向盘转角和方向盘变化只作为事件后的响应标签或响应证据，防止把驾驶员已经做出的操作泄漏进事件锚点。

## 输入证据

- 当前失稳候选表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`
- 旧事件上下文：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/candidate_events_master.csv` 中的 `old_v400_context_trigger_idx`
- 道路设计：`F:/data_set_process/data_process/01_datasets/多模态数据/被试数据集合/道路信息/full_centerline_layout.csv`
- 旧日志依据：`F:/data_set_process/data_process/04_project_logs/reports/trigger_response_lag_20260421/TASK_DEFINITION_AND_EVENT_LOGIC.md`
- 人工抽查校准：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_keyboard_player_v0_1/tables/keyboard_instability_event_labels_v0_1.csv`

## 旧道路事件设定如何使用

旧日志记录的事件来源优先级是：优先使用 `*_events_v400_context.csv` 提供 `road_type_anchor`、`curvature_anchor`、`trigger_type`、`phase_type` 和 `trigger_idx`。

本版本只把这些旧信息作为辅助上下文：

- 如果旧 v400 在候选附近有 primary、strong、extreme 或 active 事件，会提高候选可信度；
- 如果旧 v400 只说明是 curve，而车辆动态证据弱，会把它视为“可能是正常过弯”，不会直接升级为失稳；
- 旧 v400 锚点不能替代新流程锚点，不能单独作为最终事件真值。

## 道路设计如何使用

道路中心线提供道路模块顺序和场景先验：

- `curve1/curve2/curve3`：弯道上下文，不等于失稳；
- `mu1/differentmu_road`：高风险路面先验；
- `fix_road/stop/zd`：特殊道路段先验；
- `longstraight` 和 `middle_section*`：普通道路上下文。

车辆坐标到道路中心线的最近点距离被记录为 `road_design_mapping_reliability`。如果可靠度是 `very_low`，模块名只能作为弱参考。

## 当前数量

- 全量失稳候选：1227
- 自动/已确认采用：701
- 需要中间复核：177
- 低证据剔除：349

采用候选中包含：

- `hybrid_accept_high`：326
- `hybrid_accept_medium`：344
- `manual_confirmed_accept`：31

## 输出文件

- 全量判定表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_instability_events_v0_1.csv`
- 自动采用表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_auto_accepted_events_v0_1.csv`
- 中间复核表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_review_queue_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_instability_summary_v0_1.csv`
- 道路模块交叉表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_module_summary_v0_1.csv`
- 人工抽查校准表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_manual_calibration_v0_1.csv`
- 中文说明：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/road_guided_instability_v0_1_cn.md`

## 可用结论

可以说：

- 之前 404 个弯道候选不能作为车辆失稳主样本；
- 当前已有一版道路设定引导的车辆失稳候选清单；
- 该清单减少了人工逐条标注工作量；
- 下一步可以基于 701 个采用候选生成正式样本 manifest 和处理后车辆窗口。

不能说：

- 701 个事件已经是完全人工真值；
- 道路模块本身证明车辆失稳；
- 旧 v400 锚点已经被重新确认正确；
- 生理数据、脑电教师或连续风格已经有效。

## 下一步使用规则

1. 用 `road_guided_auto_accepted_events_v0_1.csv` 构建车辆失稳版 `samples_master`。
2. 所有样本必须保留原始车辆文件、原始时间戳、锚点时间、输入窗口、标签窗口、道路上下文、旧 v400 支持和失稳动态证据。
3. 正式训练前必须重新生成处理后车辆窗口，不能继续使用之前基于 404 个弯道候选的窗口。
4. 进入模型前先做无学习基线和强车辆基线，不能直接进入风格/生理有效性结论。
