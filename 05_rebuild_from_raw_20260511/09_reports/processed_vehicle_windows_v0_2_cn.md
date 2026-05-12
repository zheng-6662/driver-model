# 处理后车辆窗口数据 v0.2 说明

生成时间：2026-05-12

## 处理目标

把阶段 2 中低泄漏的 `raw_road_curvature_onset` 候选样本，处理成阶段 3 可用的车辆输入窗口和方向盘未来标签窗口。本版本只处理车辆数据，不处理生理或脑电，不训练模型。

## 输入

- 样本清单：`02_samples/tables/samples_master.csv`
- 原始车辆 CSV：`01_datasets/数据预处理/原始车辆数据/<被试名>/*.csv`
- 选择规则：`recommended_for_stage3_vehicle_baseline=True`
- 窗口：`pre1_label2_event_trigger`、`pre2_label2_old_main`、`pre3_label3_response_coverage`

## 处理规则

1. 原始 CSV 只读，不覆盖。
2. `StorageTime` 强制转为 `datetime64[ns]` 后换算为秒，避免微秒/纳秒单位错误。
3. 同一时间戳的同一信号先按均值折叠，再插值到 200 Hz 车辆时间网格。
4. 输入特征保持原始物理量，不做标准化、不做 train/test 统计拟合、不做基线校正。
5. 标签为 `zx|SteeringWheel` 相对锚点时刻方向盘值的未来增量。
6. 每个数组同时保存 valid mask，后续模型或基线必须显式使用 mask。

## 输出概要

             window_config_id                                                                                                                                                 npz_path                                                                                                                                                            index_path  sample_count   input_shape label_shape  mean_input_valid_ratio  mean_label_valid_ratio
    pre1_label2_event_trigger     F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/arrays/pre1_label2_event_trigger.npz     F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/tables/sample_index_pre1_label2_event_trigger.csv           359 [359, 201, 9]  [359, 401]                     1.0                     1.0
         pre2_label2_old_main          F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/arrays/pre2_label2_old_main.npz          F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/tables/sample_index_pre2_label2_old_main.csv           359 [359, 401, 9]  [359, 401]                     1.0                     1.0
pre3_label3_response_coverage F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/arrays/pre3_label3_response_coverage.npz F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_road_curvature_v0_2/tables/sample_index_pre3_label3_response_coverage.csv           359 [359, 601, 9]  [359, 601]                     1.0                     1.0

## 无泄漏边界

本处理版本只包含 `raw_road_curvature_onset` 且 `input_end_rel_s<=0` 的样本，适合作为阶段 3 低泄漏车辆基线的起点。早期观察窗口、旧 v400 参考锚点和 raw dynamic 响应锚点没有被处理进本版本，避免把响应结果混入事件触发预测主线。
