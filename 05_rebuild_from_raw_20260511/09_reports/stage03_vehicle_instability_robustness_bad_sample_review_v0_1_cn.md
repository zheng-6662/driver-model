# 阶段 3 稳健性坏样本复盘 v0.1

生成时间：2026-05-12

## 目的

本轮不训练新模型，而是从强车辆稳健性逐样本指标中找出跨配置、跨模型反复失败的事件。目标是区分：当前问题更像事件锚点/样本质量问题、车辆历史信息不足，还是模型结构无法表达反向修正和多段修正。

## 方法

- 输入：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_per_sample_metrics.csv`。
- 只使用 test split 的逐样本指标。
- 对每个 `robustness_config_id::model_name` 单独取 sample RMSE top20% 作为高误差事件。
- 按 `event_uid` 聚合跨模型、跨窗口、跨 split 的复发次数。
- 统计错侧、严重幅值不足、尾段漂移、零线穿越错误、反向修正不匹配、多段修正不匹配、大幅响应漏召回、峰值时间大误差、启动延迟大误差。

## 主要发现

- 复发最高的事件是 `vehicle_instability_allraw__hzh__2025_09_26_20_50_27__000337435`，subject=`hzh`，进入 top20 高误差的 config-model 次数为 15/15。
- 该事件最差的一次出现在 `session_pre3` + `peak_scaled_template_context_no_subject`，sample RMSE=1.889703。
- 高频坏样本不是只由单一模型造成；需要优先画这些事件的原始车辆轨迹、锚点、方向盘标签和预测曲线。

## 复发坏样本 Top10

| recurrence_rank | event_uid | subject | high_rmse_top20_count | worst_config | worst_model | worst_sample_rmse | wrong_side_any | severe_amp_under_any | reversal_mismatch_any | multi_segment_mismatch_any |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vehicle_instability_allraw__hzh__2025_09_26_20_50_27__000337435 | hzh | 15 | session_pre3 | peak_scaled_template_context_no_subject | 1.889703 | 1 | 1 | 1 | 1 |
| 2 | vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000392590 | gzj | 15 | subject_main | formal_ridge_vehicle_context_no_subject | 1.616231 | 0 | 1 | 1 | 0 |
| 3 | vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000592750 | gzj | 15 | subject_main | formal_ridge_vehicle_context_no_subject | 1.513877 | 0 | 1 | 1 | 0 |
| 4 | vehicle_instability_allraw__gzj__2025_09_27_11_38_49__000051595 | gzj | 15 | subject_main | knn_template_context_no_subject | 0.979521 | 0 | 1 | 1 | 0 |
| 5 | vehicle_instability_allraw__gf__2025_09_26_10_52_57__000066795 | gf | 14 | random_main | knn_template_context_no_subject | 1.070809 | 0 | 1 | 1 | 0 |
| 6 | vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000060335 | hzh | 14 | session_pre1 | direction_gated_knn_template_no_subject | 0.933242 | 1 | 1 | 1 | 1 |
| 7 | vehicle_instability_allraw__hzh__2025_09_27_19_44_05__000407670 | hzh | 13 | session_pre1 | rbf_kernel_ridge_context_no_subject | 1.445969 | 0 | 1 | 1 | 1 |
| 8 | vehicle_instability_allraw__tyy__2025_09_28_14_44_09__000058890 | tyy | 12 | random_main | formal_ridge_vehicle_context_no_subject | 1.136105 | 0 | 1 | 1 | 0 |
| 9 | vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000221890 | hzh | 12 | session_pre3 | formal_ridge_vehicle_context_no_subject | 1.127500 | 0 | 1 | 1 | 1 |
| 10 | vehicle_instability_allraw__zxy__2025_09_28_16_35_30__000185120 | zxy | 10 | session_pre3 | formal_ridge_vehicle_context_no_subject | 3.190237 | 1 | 1 | 1 | 1 |

## 分被试坏样本率 Top10

| subject | n_events | high_rmse_top20_rate | mean_sample_rmse | wrong_side_rate | severe_amp_under_rate | reversal_mismatch_rate | multi_segment_mismatch_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xst | 1 | 0.600000 | 0.988742 | 0.400000 | 0.800000 | 1.000000 | 0.000000 |
| txj | 13 | 0.507692 | 0.861238 | 0.384615 | 0.538462 | 0.907692 | 0.200000 |
| lx | 2 | 0.400000 | 0.695153 | 0.500000 | 0.500000 | 1.000000 | 0.000000 |
| gf | 14 | 0.377778 | 0.580297 | 0.237037 | 0.288889 | 0.985185 | 0.074074 |
| tyy | 14 | 0.325926 | 0.568829 | 0.422222 | 0.237037 | 0.962963 | 0.274074 |
| gzj | 78 | 0.253226 | 0.565077 | 0.135484 | 0.267742 | 0.969355 | 0.135484 |
| zxy | 16 | 0.214286 | 0.654892 | 0.264286 | 0.357143 | 0.978571 | 0.121429 |
| zx | 34 | 0.208889 | 0.545010 | 0.244444 | 0.333333 | 0.973333 | 0.195556 |
| zdq | 2 | 0.200000 | 0.489263 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| hzh | 88 | 0.170048 | 0.454834 | 0.154589 | 0.266667 | 0.960386 | 0.349758 |

## 产物

- 复发坏样本总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_bad_event_recurrence.csv`
- 代表坏样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_representative_bad_events.csv`
- 物理错误汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_error_flag_summary_by_config_model.csv`
- 分被试汇总：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_subject_bad_summary.csv`
- 坏样本矩阵：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_bad_event_matrix.csv`
- 复发事件图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_recurrent_bad_events.png`
- 物理错误热图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_error_flag_heatmap.png`
- 分被试坏样本率图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_subject_bad_rate.png`
- 坏样本矩阵图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/figures/robustness_bad_event_matrix.png`

## 下一步

下一步应对代表坏样本表中的前 10-20 个事件画原始车辆时序、锚点、GT 方向盘响应和主要候选预测曲线。只有确认失败不是锚点错误或样本质量问题后，才应进入结构化车辆模型设计。
