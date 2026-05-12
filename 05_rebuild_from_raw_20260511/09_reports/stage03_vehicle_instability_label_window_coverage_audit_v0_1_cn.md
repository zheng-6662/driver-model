# 阶段 3：标签窗口覆盖审计 v0.1

## 目的

复发坏样本归因显示，当前失败不一定都来自车辆-only 模型结构，也可能来自 2 秒标签窗口覆盖不足、连续事件未拆分或锚点附近已发生响应。本审计把该问题扩展到正式高置信失稳样本全集。

## 输入

- 样本清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 处理后数组：`pre1_label2_event_trigger.npz`、`pre2_label2_old_main.npz`、`pre3_label3_response_coverage.npz`
- 复发坏样本归因表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_failure_attribution_v0_1/tables/bad_event_failure_attribution_table.csv`

## 窗口级结果

```text
             window_config_id  n_samples  label_end_rel_s  peak_near_end_rate  tail_unsettled_rate  response_unsettled_rate  event_duration_exceeds_label_rate  median_peak_abs  median_peak_time_s  median_tail_over_peak  median_reversal_count  mean_label_valid_ratio
    pre1_label2_event_trigger        906              2.0            0.397351             0.739514                 0.786976                                1.0         0.890118                1.59               0.876206                    2.0                     1.0
         pre2_label2_old_main        906              2.0            0.397351             0.739514                 0.786976                                1.0         0.890118                1.59               0.876206                    2.0                     1.0
pre3_label3_response_coverage        906              3.0            0.271523             0.650110                 0.675497                                1.0         1.078526                1.98               0.615278                    3.0                     1.0
```

## 事件级窗口策略计数

```text
                  recommended_window_policy  n_events     rate
  use_3s_label_or_split_continuing_response       418 0.461369
use_3s_or_longer_label_for_late_peak_review       247 0.272627
  review_continuous_event_or_longer_than_3s       157 0.173289
               two_second_label_probably_ok        60 0.066225
                   review_2s_tail_or_anchor        24 0.026490
```

## 关键数字

- 事件数：906
- 2 秒后出现更大峰值：247/906 (27.26%)
- 2 秒后仍有明显变化：635/906 (70.09%)
- 2 秒标签需复核：822/906 (90.73%)
- 3 秒标签仍需复核：612/906 (67.55%)
- 主 2 秒窗口 response_unsettled_rate：78.70%
- 3 秒诊断窗口 response_unsettled_rate：67.55%

## 图表

- 窗口旗标率：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_coverage_rates_by_window.png`
- 推荐窗口策略计数：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_policy_counts.png`
- 3 秒峰值时间和尾段散点：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_label_window_coverage_audit_v0_1/figures/label_window_peak_tail_scatter_pre3.png`

## 解释边界

本审计不训练模型，不评估连续风格、生理或 EEG 有效性。`label_window_2s_needs_review` 和 `label3_still_needs_review` 是规则旗标，不是人工最终判定。长事件、保持转向和真实连续控制会让尾段不回零，因此下一步需要把任务定义拆清楚。

## 建议

1. 如果目标是“事件触发后即时响应”，保留 2 秒标签，但需要单独处理持续失稳和尾段未稳定样本。
2. 如果目标是“覆盖完整方向盘响应”，应把 3 秒或更长窗口作为正式候选，并重新跑车辆-only 基线。
3. 对 3 秒仍未稳定的样本，优先考虑事件拆分或长事件标签，而不是直接把这些样本丢给生理模型解释。
