# 阶段 2 数据版本卡：R2E raw candidate manifest v0.2

生成时间：2026-05-12

## 版本定位

本版本只用于事件锚点和样本清单重建，不用于直接训练模型。它把旧事件表、原始道路上下文候选和原始车辆动态候选放在同一张清单里，并用泄漏风险字段区分它们。

## 输入来源

- 原始车辆/生理/脑电清单：`01_audit/tables/raw_file_inventory.csv`
- 时间连续性和模态重叠：`01_audit/tables/timestamp_continuity_report.csv`、`01_audit/tables/modality_overlap_report.csv`
- 旧流程事件参考：`01_datasets/多模态数据/被试数据集合/<subject>/event/*events_v400_context.csv`
- 原始车辆信号：`01_datasets/数据预处理/原始车辆数据/<subject>/*.csv`
- 道路设计记录：`01_datasets/多模态数据/被试数据集合/道路信息`

## 道路设计记录审计

- 道路信息目录文件数：49
- 其中 CSV 文件数：8
- 含 `curvature/kappa/curvature_1pm` 的道路设计 CSV：8
- 道路设计清单：`02_samples/tables/road_design_inventory.csv`
- 当前只把道路设计作为锚点来源证据和后续精确对齐依据；本版低泄漏道路候选仍来自原始车辆 `lanecurvatureXY` 的时间序列，未把道路设计文件强行投影到每个原始时间戳。

## 候选锚点来源

anchor_source
old_v400_context_trigger_idx    6247
raw_vehicle_dynamic_onset       5013
raw_road_curvature_onset         359

## 窗口配置

                window_config_id  input_start_rel_s  input_end_rel_s  label_start_rel_s  label_end_rel_s                                   causal_setting                                                                            window_note
       pre1_label2_event_trigger               -1.0              0.0                0.0              2.0                event_trigger_predict_full_future                                                    1s event-pre input, 2s future label
            pre2_label2_old_main               -2.0              0.0                0.0              2.0                event_trigger_predict_full_future                                2s event-pre input, old-main comparable 2s future label
   pre3_label3_response_coverage               -3.0              0.0                0.0              3.0 event_trigger_predict_full_future_longer_context                           3s pre input and 3s future label for response-coverage audit
pre2_obs0p5_label2_early_observe               -2.0              0.5                0.5              2.5     early_observation_predict_remaining_response contains 0.5s post-anchor observation; not comparable to pure event-trigger prediction

## 样本数量

- 候选事件总数：11619
- 旧处理事件参考候选：6247
- 原始信号重建候选：5372
- old v400 primary 候选：1461
- `samples_master.csv` 行数：46476
- 车辆输入和标签窗口均可覆盖的样本行：46284
- 当前可作为较低泄漏道路上下文候选的 stage3 车辆基线行：1077

## 窗口可用性

window_config_id
pre1_label2_event_trigger           11584
pre2_label2_old_main                11574
pre2_obs0p5_label2_early_observe    11571
pre3_label3_response_coverage       11555

## 旧锚点与原始候选的近邻关系

- 旧参考锚点 1 秒内可找到 raw dynamic onset 的数量：2817
- 旧参考锚点 1 秒内可找到 raw road curvature onset 的数量：169

## 切分协议

- `random_event_split`：按 `event_uid` 哈希切分，避免同一事件的不同窗口落入不同 split。
- `session_level_split`：按 `subject + session_stamp` 哈希切分，同一记录内所有事件同 split。
- `subject_level_split`：按 `subject` 哈希切分，评估跨被试泛化可行性。
- 任何标准化、特征学习、风格聚类、质量阈值学习都必须只在 train split 上拟合，再应用到 val/test。

## 当前结论

本版本已经能把候选样本追溯到原始文件、原始时间戳、锚点来源、窗口和模态可用性。但 old v400 和 raw dynamic 锚点都不能直接证明无泄漏；进入正式阶段 3 前，只能优先使用 `raw_road_curvature_onset` 的低泄漏候选做保守车辆基线预研，或者先人工/GPTPro 审查锚点规则。
