# 道路事件位置与锚点重建审计 v0.1

生成时间：2026-05-12 16:37:03

## 这一步为什么做

当前旧流程模型改来改去提升有限，一个核心怀疑是：样本锚点可能不是“道路事件真正发生的时刻”，而是车辆或驾驶员已经开始响应之后的时刻。这个审计不训练模型，只检查三件事：

1. 道路设计文件里有哪些道路模块，它们在道路中心线上的位置范围是什么；
2. 每条原始车辆记录能否可靠映射到这些道路模块；
3. 旧 v400 锚点与道路曲率候选、非方向盘车身动态候选、道路模块边界之间是否对齐。

## 当前最重要结论

- 已经可以从道路中心线整理出 16 个道路模块/实例。
- 原始车辆记录共 91 条，其中可完成道路投影的记录数为 91 条。
- 旧 v400 锚点共 6247 个。
- 旧锚点 1 秒内贴近非方向盘车身动态候选的数量为 736。
- 旧锚点 1 秒内贴近道路曲率候选的数量为 169。
- 旧锚点 1 秒内贴近道路模块进入/离开边界的数量为 321。

这些数字说明：旧锚点不能直接当作“道路事件位置真值”。它有一部分和车身动态很近，但和道路曲率或道路模块边界的直接贴合并不充分。后续如果重新构建样本，应该优先采用“道路位置先验 + 非方向盘车身姿态确认”的锚点，而不是直接继承旧 trigger_idx。

## 道路映射质量

每条记录道路映射状态：

```text
mapping_status
ok    91
```

道路模块片段映射可靠性：

```text
segment_mapping_reliability
low         325
very_low    279
high        259
medium       27
```

解释：

- `high` 表示车辆坐标到道路中心线距离较小，道路模块名称较可信；
- `medium` 可以作为参考，但最好结合车身姿态；
- `low` / `very_low` 说明车辆坐标和道路中心线相距较远，模块名称不能单独作为锚点依据。

## 旧锚点对齐情况

旧锚点分类：

```text
old_anchor_audit_bucket
old_after_body_onset           2593
old_before_body_onset          2486
old_close_to_body_only          679
old_close_to_road_only          407
old_close_to_road_and_body       57
old_unaligned_or_unverified      25
```

分类含义：

- `old_close_to_road_and_body`：旧锚点同时接近道路曲率候选和车身动态候选，可信度相对更高；
- `old_close_to_body_only`：旧锚点更像是贴近车辆已经出现动态响应的时刻；
- `old_close_to_road_only`：旧锚点更像贴近道路位置变化，但缺少车身动态支持；
- `old_after_body_onset`：车身动态候选在旧锚点之前出现，说明旧锚点可能偏晚；
- `old_before_body_onset`：旧锚点早于车身动态候选，可能是道路事件提前量，也可能是旧锚点和响应未对齐；
- `old_unaligned_or_unverified`：暂时无法用当前候选解释。

## 道路引导候选情况

道路引导候选采用建议：

```text
recommended_decision
hybrid_reject_low_evidence          349
hybrid_accept_medium                344
hybrid_accept_high                  326
hybrid_review_conflict_or_medium    177
manual_confirmed_accept              31
```

道路引导候选的道路映射可靠性：

```text
road_design_mapping_reliability
very_low    526
low         378
high        265
medium       58
```

这部分说明：上一版道路引导候选可以作为下一步样本候选，但其中低可靠道路映射比例仍然不能忽略。正式训练前需要把候选分成自动采用、人工复核、只诊断不用训练三类。

## 这一步不能下的结论

- 不能说道路文件已经给出了每条记录的绝对真值锚点。
- 不能说旧 v400 锚点全部错误；只能说旧锚点需要按道路位置和车身姿态重新分级。
- 不能说当前道路引导候选已经是人工真值。
- 不能继续用方向盘未来变化来定义事件锚点；方向盘只能作为事件后的响应标签或后验验证。

## 建议下一步

1. 用 `old_new_anchor_alignment_v0_1.csv` 找出旧锚点明显偏晚、明显偏早、无法对齐的样本。
2. 优先抽查 `old_after_body_onset` 和 `old_unaligned_or_unverified`，确认旧流程坏样本是否集中在这些锚点风险组。
3. 对 `high/medium` 道路映射可靠性的道路引导候选，生成新的样本清单。
4. 对 `low/very_low` 道路映射样本，不直接进入正式训练，先做复核或诊断。
5. 如果用户认可，再进入“新锚点样本 manifest + 强车辆基线”阶段。

## 主要产物

- 道路模块位置表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\tables\road_event_position_map_v0_1.csv`
- 每条记录道路映射摘要：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\tables\session_road_mapping_summary_v0_1.csv`
- 每条记录经过道路模块的时间段：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\tables\session_module_entry_exit_v0_1.csv`
- 旧锚点对齐表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\tables\old_new_anchor_alignment_v0_1.csv`
- 道路引导候选对齐表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\tables\road_guided_anchor_alignment_v0_1.csv`
- 审计汇总表：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\tables\road_event_anchor_audit_summary_v0_1.csv`
- 道路模块位置图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\figures\road_event_position_map_v0_1.png`
- 锚点审计概览图：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\figures\road_anchor_audit_overview_v0_1.png`
- 代表样本面板目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\road_event_anchor_audit_v0_1\figures\representative_panels`

代表样本图数量：7
