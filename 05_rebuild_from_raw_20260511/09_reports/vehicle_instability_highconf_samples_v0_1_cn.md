# 车辆失稳高置信正式样本清单 v0.1

生成时间：2026-05-12

## 这一步做了什么

这一步把全原始车辆 CSV 重筛得到的高置信失稳事件，整理成新流程正式 `samples_master`。它不是新模型训练，也不是生理/风格有效性验证。

## 输入

- 高置信事件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_primary_high_confidence_v0_1.csv`
- 处理后车辆窗口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1`
- 模态完整性矩阵：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/subject_session_modality_matrix.csv`

## 输出

- 样本清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.csv`
- 样本 JSONL：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/samples_master.jsonl`
- 事件锚点表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/event_anchor_table.csv`
- split 表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/split_table.csv`
- split 可行性：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/split_feasibility_report.csv`
- 排除原因：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1/tables/sample_exclusion_reasons.csv`
- 数据版本卡：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/dataset_version_card_vehicle_instability_highconf_v0_1_cn.md`

## 当前判断

906 个高置信失稳事件已经具备可追溯样本记录，可以进入新流程车辆基线准备。需要注意，本版本只是车辆失稳样本清单和窗口索引，不证明旧模型有效，也不证明连续风格、生理或脑电有效。

## 下一步

使用 `pre2_label2_old_main` + `session_level_split` 建立新流程无学习基线和强车辆基线；训练和标准化必须只用 train split。
