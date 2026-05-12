# 阶段 3：复发坏样本详细曲线复盘 v0.1

生成时间：2026-05-12

## 目的

上一轮已经知道 RBF/KNN/template 在若干配置下会反复失败。本轮不训练新路线，不引入生理、脑电或连续风格，只把复发最高的坏事件画成可复核曲线，检查失败更像锚点/窗口问题、原始车辆局部异常，还是车辆-only 模型表达不足。

## 输入

- 代表坏样本表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_robustness_bad_sample_review_v0_1/tables/robustness_representative_bad_events.csv`
- 正式样本清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 处理后车辆窗口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/arrays`
- 原始车辆 CSV：只按 `samples_master.csv` 中每个事件的 `vehicle_raw_absolute_path` 读取片段；未修改原始文件。

## 方法

- 选取复发坏样本 Top 12。
- 对每个事件使用其 `worst_config` 对应的窗口和 split。
- 复用已提交的 formal ridge、RBF KRR、KNN template、direction-gated KNN、peak-scaled template 逻辑，仅为绘图重建预测曲线。
- 图中同时画事件锚点、输入窗口、标签窗口、事件结束线、原始方向盘相对锚点变化、GT 方向盘增量、候选模型预测、原始车辆动力学与道路上下文波形。

## 主要发现

- 复发最高事件：`vehicle_instability_allraw__hzh__2025_09_26_20_50_27__000337435`，subject=`hzh`，config=`session_pre3`。
- Top 12 事件 * 5 个车辆-only 候选模型的逐样本曲线中，严重幅值不足率=0.700，错侧率=0.233，反向修正计数完全匹配率=0.033。
- 这些图仍不能单独证明“生理有效”或“Transformer 更好”；它们的用途是把车辆-only 当前失败类型具体化，为下一版结构化车辆模型提供目标。

## 图表索引 Top12

| recurrence_rank | event_uid | subject | config_id | worst_sample_rmse | figure_png |
| --- | --- | --- | --- | --- | --- |
| 1 | vehicle_instability_allraw__hzh__2025_09_26_20_50_27__000337435 | hzh | session_pre3 | 1.889703 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank01_session_pre3_vehicle_instability_allraw__hzh__2025_09_26_20_50_27__000337435.png |
| 2 | vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000392590 | gzj | subject_main | 1.616231 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank02_subject_main_vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000392590.png |
| 3 | vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000592750 | gzj | subject_main | 1.513877 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank03_subject_main_vehicle_instability_allraw__gzj__2025_09_27_12_17_12__000592750.png |
| 4 | vehicle_instability_allraw__gzj__2025_09_27_11_38_49__000051595 | gzj | subject_main | 0.979521 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank04_subject_main_vehicle_instability_allraw__gzj__2025_09_27_11_38_49__000051595.png |
| 5 | vehicle_instability_allraw__gf__2025_09_26_10_52_57__000066795 | gf | random_main | 1.070809 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank05_random_main_vehicle_instability_allraw__gf__2025_09_26_10_52_57__000066795.png |
| 6 | vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000060335 | hzh | session_pre1 | 0.933242 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank06_session_pre1_vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000060335.png |
| 7 | vehicle_instability_allraw__hzh__2025_09_27_19_44_05__000407670 | hzh | session_pre1 | 1.445969 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank07_session_pre1_vehicle_instability_allraw__hzh__2025_09_27_19_44_05__000407670.png |
| 8 | vehicle_instability_allraw__tyy__2025_09_28_14_44_09__000058890 | tyy | random_main | 1.136105 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank08_random_main_vehicle_instability_allraw__tyy__2025_09_28_14_44_09__000058890.png |
| 9 | vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000221890 | hzh | session_pre3 | 1.127500 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank09_session_pre3_vehicle_instability_allraw__hzh__2025_09_26_21_03_19__000221890.png |
| 10 | vehicle_instability_allraw__zxy__2025_09_28_16_35_30__000185120 | zxy | session_pre3 | 3.190237 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank10_session_pre3_vehicle_instability_allraw__zxy__2025_09_28_16_35_30__000185120.png |
| 11 | vehicle_instability_allraw__zxy__2025_09_28_16_35_30__000471085 | zxy | session_pre3 | 2.686779 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank11_session_pre3_vehicle_instability_allraw__zxy__2025_09_28_16_35_30__000471085.png |
| 12 | vehicle_instability_allraw__hzh__2025_09_27_19_33_25__000325285 | hzh | random_main | 1.494551 | F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event/rank12_random_main_vehicle_instability_allraw__hzh__2025_09_27_19_33_25__000325285.png |

## 产物

- 图索引：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_figure_index.csv`
- 模型逐事件误差表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/tables/bad_event_curve_model_error_table.csv`
- 总览拼图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/bad_event_curve_contact_sheet.png`
- 单事件图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_bad_event_curve_review_v0_1/figures/per_event`

## 下一步

优先人工抽看 Top 12 曲线中是否存在明显锚点偏早/偏晚、标签窗口没有覆盖完整响应、原始 `ay/roll/vyaw` 局部异常或道路上下文突变。如果这些问题不能解释大部分失败，再进入结构化车辆响应模型：方向/幅值/峰值时间/反向修正/多段修正分解，或关键点 + 残差轨迹模型。
