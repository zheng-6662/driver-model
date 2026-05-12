# 阶段 2 补充：Codex 自动事件审阅 v0.1

生成时间：2026-05-12

## 2026-05-12 重要修正

用户已确认本项目真正需要的是“车辆失稳样本”，不是“弯道样本”。因此本报告中的 404 个候选只能解释为弯道/道路曲率候选，现已降级为道路上下文参考材料，不能作为主事件样本继续训练正式模型。

新的主线已改为 `vehicle_instability_onset_codex_v0_1`，输出见：

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/instability_event_review_v0_1_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`

后续不得把本报告中的 404 个弯道候选称为车辆失稳样本。

## 为什么做

用户认为逐个播放和人工标注事件仍然太耗时，因此本阶段先由 Codex 对低泄漏道路曲率候选进行规则化自动审阅。输出不是最终真值，而是带证据、分数和置信度的候选标签，用来减少人工复核范围。

## 输入

- 候选事件表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/candidate_events_master.csv`
- 原始车辆 CSV：只读取 `原始车辆数据/<被试名>/*.csv`
- 主锚点来源：`raw_road_curvature_onset`
- 辅助证据：`old_v400_context_trigger_idx` 和 `raw_vehicle_dynamic_onset` 只作附近支持计数，不作为无泄漏真值。

## 输出

- 自动审阅标签：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_reviewed_event_labels_v0_1.csv`
- 自动采用标签：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_auto_accepted_event_labels_v0_1.csv`
- 需要人工复核队列：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_needs_human_review_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/tables/codex_event_review_summary_v0_1.csv`
- 分数图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/codex_event_review_v0_1/figures/codex_event_review_score_overview_v0_1.png`

## 审阅规则

1. 只把道路曲率候选作为主事件来源。
2. 长道路曲率段拆成 `curve_entry` 和 `curve_exit_or_return`，避免把一整段弯道当成一个事件。
3. 短道路曲率段保留为 `curve_short` 或 `curve_brief`。
4. 每个候选计算道路曲率强度、方向盘响应幅值、横向加速度、旧流程邻近点、车辆动态邻近点、车速和采样点数量。
5. 得到 0-100 分的 `codex_review_score`，并分为 `auto_accept_high`、`auto_accept_medium`、`needs_human_review`、`reject_low_evidence`。

## 当前数量

总自动审阅标签数：404

按决策统计：

```text
codex_recommended_decision
auto_accept_high       224
auto_accept_medium     136
needs_human_review      43
reject_low_evidence      1
```

按事件角色统计：

```text
auto_event_role
curve_short             314
curve_entry              45
curve_exit_or_return     45
```

需要人工复核或剔除的数量：44

## 重要边界

这不是人工真值，也不能直接证明事件锚点最终正确。它的用途是先由 Codex 做第一轮筛选：高/中置信标签可以进入下一步候选 `codex_auto_accepted` 数据版本，低置信和冲突样本再由用户少量复核。
