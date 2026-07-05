# v283 生理路线 lineage / gap 审计

## 本轮目的

- v282 之后继续推进 goal，但不再盲目换模型。
- 把 v254b-v282 的生理证据链合并，明确旧路线是否关闭、下一步还剩什么真正可尝试。

## 决策摘要

| item                                     | value   | evidence                                                                   |
|:-----------------------------------------|:--------|:---------------------------------------------------------------------------|
| current_goal_achieved                    | False   | 没有任何 deployable 生理路线让 test bad_top10 稳定超过 fixed wait-latest。 |
| old_feature_selector_route_closed        | True    | v269/v271/v282 对旧特征筛选、校准和歧义消解均未通过。                      |
| physio_source_alignment_ready            | True    | v254b/v260/v268 均显示 200Hz 时间轴与事件窗口覆盖基本可用。                |
| next_route_requires_feature_redefinition | True    | 主要瓶颈是有效信号和身份混淆，不是训练脚本或简单模型容量。                 |

## 生理数据与质量结论

| aspect                      | status   | evidence                                           | implication                                                                                          |
|:----------------------------|:---------|:---------------------------------------------------|:-----------------------------------------------------------------------------------------------------|
| 200Hz_event_alignment       | pass     | v254b ok_rate_mean=0.928; v260 ok_rate_mean=0.928  | 事件窗口覆盖基本够用，失败不能简单归因于大面积对齐缺失。                                             |
| source_timing_integrity     | pass     | median_hz=200.000, gaps=0, duplicates=0            | 200Hz 连续层时序本身稳定；失败不能简单归咎于采样断裂。                                               |
| derived_signal_availability | warn     | HRV_RMSSD:0/82; RESP_Amplitude:0/82; RESP_BPM:0/82 | 部分派生生理列不可用或近常数，尤其 HRV_RMSSD、RESP_BPM/Amplitude、部分 EDA；这会削弱高层 biomarker。 |
| event_window_coverage       | pass     | min split-delay ok_rate=0.889                      | 事件窗口覆盖整体尚可，问题更可能是特征有效性和泛化，而不是大面积缺失。                               |
| identity_vs_behavior_signal | warn     | median family identity/behavior eta ratio=68.74    | bio260 更容易区分 subject/recording，而不是行为目标；这解释了 subject-disjoint 下增量不稳定。        |
| identity_behavior_ratio     | warn     | family_median_ratio=68.74; max_ratio=187.66        | 生理特征更容易识别驾驶员/记录而不是行为，subject-disjoint 泛化受限。                                 |

## 路线 lineage

| version   | route                                             | status                           | best_badtop10_signal                                                 | trajectory_badtop10_outcome                          | reason                                                                                           |
|:----------|:--------------------------------------------------|:---------------------------------|:---------------------------------------------------------------------|:-----------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| v254b     | 200Hz event-window statistics                     | failed_for_main_goal             | physio200_all delta_macro_f1=0.0308                                  | not tested as direct selector in v254b               | 分类上只出现很小 bad_top10 诊断增量，未来行为/回归和 vehicle+bio 主结果未超过 vehicle-only。     |
| v260      | ECG/EDA/RESP/EMG event biomarkers                 | failed_for_main_goal             | vehicle_plus_physio200_curated_ref delta_macro_f1=0.0135             | not sufficient; passed to v261-v269 and still failed | bio260 对 bad_top10 有很小诊断信号，但 subject-disjoint 行为预测和后续候选选择没有转成轨迹收益。 |
| v268      | quality / identity / rerank identifiability audit | diagnosed_bottleneck             | source timing pass; signal availability warn; identity/behavior warn | candidate rerank identifiability warn                | 不是对齐大面积失败，而是派生列不可用、身份混淆强、行为可辨识性弱。                               |
| v269      | reliable / low-identity bio feature screening     | failed_for_main_goal             | pair_test_best_deployable delta=0.0831                               | best deployable RMSE=0.7781                          | 可部署策略仍高于 fixed wait-latest，且最好 wait gate 退化成全 wait-latest。                      |
| v271      | subject/recording calibrated raw physiology state | failed_for_main_goal             | pair_test_best_deployable delta=0.0903                               | best deployable RMSE=0.7853                          | 即使给 subject/recording 无标签基线，差样本候选选择仍明显差于 fixed wait-latest。                |
| v282      | ambiguity route gate                              | failed_for_current_feature_layer | deployable top1 bad_top10 evidence=0.1989; ambiguous=0.2347          | route_viable_now=False                               | 生理 top1 可部署选择和 top3 上限都不能稳定通过 val/test gate，排序相关接近 0。                   |

## 下一步硬要求

| requirement_id   | requirement                                                              | current_evidence                                                        | status           | next_action                                                                               |
|:-----------------|:-------------------------------------------------------------------------|:------------------------------------------------------------------------|:-----------------|:------------------------------------------------------------------------------------------|
| R1               | 不能复用旧的 bio selector/reranker/reliability filter 微调作为下一步主线 | v269/v271/v282 均失败，且 v282 route_viable_now=false                   | closed_old_route | 只保留旧结果作为反例和 guardrail。                                                        |
| R2               | 若继续生理目标，必须先改善生理状态表征而不是直接加模型                   | v268 derived_signal_availability=warn，identity_vs_behavior_signal=warn | required         | 重做可用信号族筛选、质量 mask、个体内变化和低身份行为特征。                               |
| R3               | 新生理特征必须先通过车辆歧义样本 route gate                              | v282 top1 bad_top10 +0.1989，ambiguous +0.2347，corr max 0.00985        | required         | 先在 vehicle top40 内验证生理排序相关和 val/test 同向，再进预测模型。                     |
| R4               | 要明确 subject-disjoint 与 subject-aware/校准任务边界                    | v254b/v260/v271 均显示个体/记录信号强，subject-disjoint 不稳定          | required         | 如果接受 subject-aware，单独建个体校准任务；如果坚持 subject-disjoint，必须做去身份约束。 |
| R5               | 不能把 bio top3/top5 oracle 当成可部署效果                               | v281/v282 的 top3/top5 都依赖真实误差选择候选                           | guardrail        | 所有正式结论只看 val-chosen deployable policy。                                           |

## 关键判断

- 当前 goal 仍未完成：没有可部署生理路线稳定改善差样本。
- 旧路线已经足够清楚：200Hz 统计、事件型 biomarker、低身份筛选、个体/记录校准、候选消歧 gate 都未形成正式增量。
- 如果继续生理 goal，下一步必须是新定义：先做低身份但行为相关的生理状态表示，并先通过 v282 类 route gate，再进入轨迹模型。
- 如果下一版仍无法让生理距离在车辆歧义样本中产生正相关和 val/test 同向收益，就应把生理降级为 subject-aware 个体校准或边界证据。

## 关键图

- `figures\v283_physio_route_lineage_status.png`
- `figures\v283_signal_quality_by_family.png`
- `figures\v283_badtop10_macro_f1_delta.png`

## guardrail

```json
{
  "pass": true,
  "zip_testzip": true,
  "lineage_rows": 6,
  "quality_rows": 6,
  "requirements_rows": 5,
  "current_goal_achieved": false,
  "old_feature_selector_route_closed": true,
  "physio_source_alignment_ready": true,
  "next_route_requires_feature_redefinition": true
}
```
