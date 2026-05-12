# 阶段 2 补充：人工事件标注审查包 v0.1

生成时间：2026-05-12

## 为什么做

当前 `raw_road_curvature_onset` 仍只是低泄漏候选锚点，不是最终事件真值。用户提出可以人工打标签，因此本包把原始车辆行驶过程重现成多通道时间线，让人工决定哪里到哪里算事件。

## 本包内容

- 审查页面：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/review_index.html`
- 时间线图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/figures`
- 人工标签模板：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/manual_event_labels_template_v0_1.csv`
- 会话清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/manual_event_label_review_v0_1/tables/session_review_manifest_v0_1.csv`

## 当前原型范围

- 本次最多选择 12 个记录作为原型。
- 实际生成记录数：12
- 人工标签模板行数：1878
- 选择逻辑：优先选择 `raw_road_curvature_onset` 数量较多的记录。

## 图中信号

每张图按原始车辆时间重现以下参数：

1. 道路曲率 `lanecurvatureXY`
2. 方向盘转角 `SteeringWheel`
3. 车速 `v_km/h`
4. 横向位置 `lateraldistance`
5. 横摆角速度 `vyaw`
6. 横向加速度 `ay`
7. 横滚角 `roll`

图中颜色：

- 蓝色：`raw_road_curvature_onset`，道路曲率候选。
- 橙色：`old_v400_context_trigger_idx`，旧流程参考。
- 红色：`raw_vehicle_dynamic_onset`，车辆动态响应候选，不能作为无泄漏事件触发真值。

## 人工标注建议

在模板中填写：

- `manual_include_for_dataset`：是否纳入后续数据集，建议填 `yes/no/unsure`。
- `manual_event_start_rel_s` / `manual_event_end_rel_s`：你人工确认的事件起止时间，单位为图中相对秒。
- `manual_anchor_rel_s`：如果要定义事件触发预测锚点，填你认为模型在此刻应该开始预测未来。
- `manual_event_type` / `manual_direction`：事件类型和方向。
- `manual_confidence_1_5`：置信度，1 表示很不确定，5 表示很确定。
- `manual_reason_or_notes`：为什么这么标，或有什么疑问。

## 重要边界

本包不会修改原始 CSV，不会训练模型，也不会把事件锚点定稿。它的目的就是把候选事件可视化给人工确认。只有人工标签回填并通过一致性审查后，才能生成下一版 `manual_verified` 样本清单。
