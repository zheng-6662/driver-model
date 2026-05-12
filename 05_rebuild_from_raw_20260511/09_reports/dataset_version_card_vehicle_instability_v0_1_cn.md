# 数据版本卡：vehicle_instability_onset_codex_v0_1

生成时间：2026-05-12

## 版本目的

这个版本用于重建“车辆失稳事件触发后的方向盘响应预测”样本，而不是弯道样本。

之前的 404 个 `codex_event_review_v0_1` 候选来自道路曲率，适合表示弯道上下文，但不能回答“车辆失稳后驾驶员如何打方向盘”。因此本版本单独定义车辆失稳锚点。

## 事件锚点如何确定

主锚点只允许来自非方向盘车辆动态异常：

- `ay`：横向加速度异常。
- `roll_rate`：横滚速率异常。

不允许作为主锚点：

- `raw_road_curvature_onset`：道路开始变弯不等于车辆失稳。
- `steer_rate`：方向盘转动已经是驾驶员动作结果，用它定义事件会把响应混入锚点。
- 旧流程 `old_v400_context_trigger_idx`：旧流程锚点只能做历史对照，不作为新流程真值。

相邻 `ay` / `roll_rate` 动态种子在 2.5 秒内会被合并成同一个候选失稳片段，片段起点作为失稳候选锚点。

## 当前数量

- 非方向盘动态种子：1833 个。
- 合并后车辆失稳候选：1227 个。
- 自动高置信采用：134 个。
- 自动中置信采用：224 个。
- 自动采用合计：358 个。
- 需要人工复核：462 个。
- 低失稳证据建议剔除：407 个。
- 覆盖被试：18 个。
- 覆盖记录：85 个。

## 评分证据

每个候选片段计算以下证据：

- 窗口内最大绝对横向加速度。
- 窗口内最大绝对横滚速率。
- 窗口内最大绝对横摆角速度。
- 窗口内横向位置变化范围。
- 窗口内中位车速。
- 事件后 3 秒方向盘相对事件前 1 秒基线的最大变化。
- 附近是否存在道路曲率候选。
- 附近是否存在旧流程候选。

注意：方向盘证据只用于判断“事件后是否存在驾驶员修正响应”，不能反过来定义失稳开始。

## 当前因果设定

本版本支持的任务是：

`检测到车辆失稳动态开始后，预测未来方向盘响应轨迹`

它暂时不支持强结论：

`在车辆失稳发生前，提前预测驾驶员未来方向盘响应`

如果后续要做失稳前预警，必须重新定义更早的锚点或输入窗口，不能直接沿用当前锚点。

## 泄漏边界

当前版本的低泄漏约束：

- 锚点不使用方向盘转动率。
- 方向盘只在锚点后作为标签和响应证据。
- 原始 CSV 不修改。
- 自动评分不使用测试集统计学习参数。

仍需注意的风险：

- 横向加速度高可能只是正常过弯，不一定是失稳。
- 当前自动规则不是人工真值，不能命名为 `manual_verified`。
- 后续训练必须重新生成 split 和标准化流程，不能复用旧道路曲率阶段 3 模型结果。

## 输出文件

- 全量候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`
- 自动采用：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_auto_accepted_events_v0_1.csv`
- 需要复核：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_needs_human_review_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_event_review_summary_v0_1.csv`
- 概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/instability_event_score_overview_v0_1.png`
- 说明报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/instability_event_review_v0_1_cn.md`

## 是否可以进入下一步

可以进入“失稳版本处理窗口生成”和“少量样本抽查”，但还不能直接训练正式模型。

进入正式阶段 3 前必须完成：

1. 抽查高置信和需复核候选，确认正常过弯误判率。
2. 冻结 `vehicle_instability_onset_codex_v0_1` manifest。
3. 重新生成处理后车辆窗口。
4. 重新定义 split 和无泄漏标准化规则。
