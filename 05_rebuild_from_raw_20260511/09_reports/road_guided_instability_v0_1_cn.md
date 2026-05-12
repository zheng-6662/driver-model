# 道路设定引导的车辆失稳事件自动判定 v0.1

生成时间：2026-05-12 15:52:16

## 为什么做

用户指出，上一版 404 个样本主要是弯道/道路曲率样本，不是项目真正需要的车辆失稳样本。逐个手工标注 1227 个失稳候选也不现实，所以本版改成自动综合判定：用原始车辆动态作为主证据，用旧项目日志和道路设定作为辅助先验。

## 用了哪些证据

1. 主证据：`ay` 和 `roll_rate` 触发的非方向盘车辆动态失稳候选。
2. 车辆响应证据：横摆角速度、横向偏移、事件后 3 秒方向盘修正幅值、车速、片段持续时间。
3. 旧流程上下文：`*_events_v400_context.csv` 在项目日志中被记录为优先事件来源，提供 `road_type_anchor`、`phase_type`、`event_level`、`trigger_idx` 等旧事件上下文。
4. 道路设定先验：从 `full_centerline_layout.csv` 读取道路模块顺序，识别 `curve1/curve2/curve3`、`fix_road`、`stop`、`mu1/differentmu_road`、`zd` 等道路场景。
5. 已有人工抽查：当前 31 条键盘标注只作为校准/确认，不要求继续人工标注全量样本。

## 关键原则

- 弯道不等于失稳。弯道只作为道路上下文，如果车辆动态证据弱，会被降权。
- 方向盘动作不用于定义失稳开始点。方向盘只作为事件之后的响应证据，避免把驾驶员操作结果泄漏进事件锚点。
- 旧 v400 事件不是新真值。它只作为旧道路事件设定和旧锚点上下文，不能替代原始车辆动态证据。
- 道路模块映射不是绝对真值。车辆坐标到道路中心线的最近距离会记录可靠度，高距离映射只作为弱参考。

## 当前结果

候选总数：1227

自动/已确认采用数：701

需要复核但不要求用户逐条手工标注的中间候选：177

31 条已有人工抽查命中的候选数：31

按最终建议统计：

```text
road_guided_recommended_decision
hybrid_reject_low_evidence          349
hybrid_accept_medium                344
hybrid_accept_high                  326
hybrid_review_conflict_or_medium    177
manual_confirmed_accept              31
```

旧 v400 道路类型支持统计：

```text
old_v400_road_type_mode
straight    695
none        311
curve       221
```

道路设计风险类别统计：

```text
road_design_risk_class
design_regular_road             478
design_high_risk_surface        326
design_curve_context            293
design_special_event_segment    130
```

道路中心线映射可靠度：

```text
road_design_mapping_reliability
very_low    526
low         378
high        265
medium       58
```

## 产物

- 全量判定表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_instability_events_v0_1.csv`
- 自动采用表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_auto_accepted_events_v0_1.csv`
- 中间复核队列表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_review_queue_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_instability_summary_v0_1.csv`
- 道路模块交叉表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_module_summary_v0_1.csv`
- 人工抽查校准表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/road_guided_instability_v0_1/tables/road_guided_manual_calibration_v0_1.csv`

## 目前可以怎么用

本版可以替代“全人工标注”的第一轮失稳样本筛选。下一步不应该回到 404 个弯道样本，也不应该马上训练模型，而是用自动采用表生成 `vehicle_instability` 样本 manifest，并在样本卡里记录每个事件的道路上下文、旧 v400 支持和失稳动态证据。

## 还不能下的结论

- 不能说这些事件已经是完全人工真值。
- 不能说道路模块本身导致失稳，只能说它提供场景先验。
- 不能说生理或连续风格有效，因为这里还没有进入风格/生理建模。
- 不能把旧 v400 锚点继续当作新流程唯一锚点。

## 质量风险

道路映射状态：`ok`。

如果某些候选的 `road_design_mapping_reliability` 是 `very_low`，说明车辆坐标到中心线距离过大，模块名称只能作为弱参考。最终样本构建时应优先依赖非方向盘车辆动态证据和旧 v400 近邻上下文，而不是单独依赖该模块名。
