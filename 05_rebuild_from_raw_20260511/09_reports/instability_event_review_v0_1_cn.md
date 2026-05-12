# 阶段 2 修正：车辆失稳事件自动审阅 v0.1

生成时间：2026-05-12

## 为什么重做

用户指出，上一版 `codex_event_review_v0_1` 的 404 个样本本质上是弯道/道路曲率候选，不是本项目真正要找的车辆失稳样本。这个判断是正确的。

因此本版把道路曲率样本降级为道路上下文参考，重新用车辆动态异常来建立候选事件。主锚点不再来自弯道开始/结束，而来自原始车辆信号中的非方向盘动态异常。

## 本版事件定义

- 主事件：车辆失稳候选。
- 主锚点来源：`raw_vehicle_dynamic_onset` 中的 `ay` 和 `roll_rate`。
- 不作为主锚点：`raw_road_curvature_onset`，因为它只说明道路是弯的，不等于车辆失稳。
- 不作为主锚点：`steer_rate`，因为它已经是驾驶员方向盘动作，直接用它找事件会把响应结果混入事件定义。
- 方向盘信号只用于事件后响应证据：例如失稳后 3 秒内方向盘是否出现明显修正。

## 因果设定

本版对应的是：

`检测到车辆失稳动态开始后，预测未来方向盘响应轨迹`

这和“进入弯道前预测未来方向盘”不是同一个任务。后续建模时必须把这个数据版本单独命名，不能和弯道事件混在一起。

## 输入

- 候选事件总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/tables/candidate_events_master.csv`
- 原始车辆 CSV：`原始车辆数据/<被试名>/Entity_Recording_*_vehicle.csv`
- 动态种子：`ay`、`roll_rate`
- 辅助上下文：附近弯道候选、旧流程候选、事件后方向盘响应

## 输出

- 全量失稳审阅表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_reviewed_events_v0_1.csv`
- 自动采用失稳候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_auto_accepted_events_v0_1.csv`
- 需要人工复核候选：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_needs_human_review_v0_1.csv`
- 汇总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/tables/instability_event_review_summary_v0_1.csv`
- 概览图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/instability_event_score_overview_v0_1.png`

## 当前数量

合并后的车辆失稳候选总数：1227

自动采用候选数：358

需要人工复核候选数：462

按决策统计：

```text
codex_recommended_decision
needs_human_review                 462
reject_low_instability_evidence    407
auto_accept_instability_medium     224
auto_accept_instability_high       134
```

按失稳证据类型统计：

```text
instability_role
instability_ay_only      1150
instability_roll_only      65
instability_ay_roll        12
```

按被试统计：

```text
subject
byx    107
cwh     72
gf      52
gzj     47
hzh    131
jy      67
lx      38
lxy     69
rjy     42
txj     71
tyy     41
xst     19
yyl     80
yzy     98
zdq     58
zt      17
zx     154
zxy     64
```

动态种子使用量：

```text
ay_seed_count           1736
roll_rate_seed_count      97
merged_seed_count       1833
```

## 当前解释

这版比 404 个弯道样本更接近用户目标，因为它把“车辆有没有出现动态异常”放在主位，而不是把“道路是不是弯道”当成事件。

但它仍然不是最终真值标签。它是 Codex 根据规则做的第一轮失稳候选筛选，后面还要做三件事：

1. 检查高分样本是否真的表现为车辆失稳，而不是正常高速过弯。
2. 检查失稳锚点之后的方向盘响应窗口是否完整覆盖了修正过程。
3. 明确后续模型任务是“失稳检测后预测响应”，还是“失稳发生前预警并预测响应”。两者输入窗口不能混用。

## 和上一版 404 个样本的关系

上一版 `codex_event_review_v0_1` 不再作为主事件样本。它只能作为道路曲率上下文、弯道背景或对照材料保存。

## 推荐优先查看

- 先看全量表：`instability_reviewed_events_v0_1.csv`
- 再看自动采用表：`instability_auto_accepted_events_v0_1.csv`
- 再看概览图：`instability_event_score_overview_v0_1.png`
- 抽查示例图目录：`instability_event_review_v0_1/figures/`

## 示例图

```text
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__byx__2025_09_28_17_35_43__000165905.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__lxy__2025_09_28_18_06_16__000389495.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__zx__2025_09_27_16_32_00__000080680.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__hzh__2025_09_27_19_33_25__000133960.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__jy__2025_09_26_17_40_51__000139535.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__gf__2025_09_26_10_30_12__000138445.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__zx__2025_09_27_18_17_48__000307245.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__jy__2025_09_26_17_51_46__000128830.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__yyl__2025_09_28_09_14_23__000748255.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__hzh__2025_09_26_20_50_27__000148635.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__zx__2025_09_27_17_29_08__000082590.png
F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/instability_event_review_v0_1/figures/vehicle_instability_onset__hzh__2025_09_26_21_17_02__000164995.png
```
