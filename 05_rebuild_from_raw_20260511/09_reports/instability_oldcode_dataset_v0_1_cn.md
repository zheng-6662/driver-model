# 旧代码兼容数据包：全原始车辆失稳高置信样本 v0.1

生成时间：2026-05-12

## 目的

这一步不是训练新模型，而是把前一步重新筛出的高置信车辆失稳事件转成旧代码可以读取的格式，用来快速测试旧车辆代码在这些样本上的表现。

## 输入

- 高置信失稳事件：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_primary_high_confidence_v0_1.csv`
- 原始车辆 CSV：`F:/data_set_process/data_process/01_datasets/数据预处理/原始车辆数据/<被试名>/*.csv`
- 锚点规则：非转向车辆动力学 onset，主要来自 `ay` 和 `roll_rate`，转向响应只作为后验证据，不用于定义锚点。

## 输出

- 旧阶段 3 `.npz` 窗口：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/arrays`
- 样本索引与 split：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables`
- 旧深度模型 manifest：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/oldcode_manifest_session_level_split.csv`

## 数量

- 输入高置信事件数：908
- 旧代码可用事件数：906
- 旧代码不可用事件数：2
- 窗口样本行数：2718

## 旧代码 split 分布

split  n_events
train       611
  val       156
 test       139

## 窗口输出

             window_config_id                                                                                                                                                              npz_path                                                                                                                                                                         index_path  sample_count   input_shape label_shape  mean_input_valid_ratio  mean_label_valid_ratio  label_peak_abs_mean  label_peak_abs_p75  label_peak_abs_p90
    pre1_label2_event_trigger     F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/arrays/pre1_label2_event_trigger.npz     F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/sample_index_pre1_label2_event_trigger.csv           906 [906, 201, 9]  [906, 401]                     1.0                     1.0             1.043273            1.357212            1.951628
         pre2_label2_old_main          F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/arrays/pre2_label2_old_main.npz          F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/sample_index_pre2_label2_old_main.csv           906 [906, 401, 9]  [906, 401]                     1.0                     1.0             1.043273            1.357212            1.951628
pre3_label3_response_coverage F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/arrays/pre3_label3_response_coverage.npz F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_processed_datasets/vehicle_instability_allraw_highconf_v0_1/tables/sample_index_pre3_label3_response_coverage.csv           906 [906, 601, 9]  [906, 601]                     1.0                     1.0             1.222546            1.521010            2.141082

## 边界说明

1. 原始 CSV 未修改。
2. 窗口和 manifest 只使用车辆数据，不包含生理和脑电。
3. 标准化没有在这里做；后续旧代码测试时只能在训练集内拟合统计量。
4. 这一步仍然是诊断，不能由此宣称连续风格或生理有效。
5. 旧代码 manifest 使用 `session_level_split` 作为默认 split，另外也输出 random/session/subject 三种 split 字段供对照。
