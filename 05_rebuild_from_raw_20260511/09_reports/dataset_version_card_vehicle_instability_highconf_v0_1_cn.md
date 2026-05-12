# 数据版本卡：车辆失稳高置信正式样本清单 v0.1

生成时间：2026-05-12

## 数据版本

- 数据版本：`vehicle_instability_highconf_v0_1`
- 事件来源版本：`vehicle_instability_all_raw_rescreen_v0_1`
- 处理后车辆窗口版本：`vehicle_instability_allraw_highconf_v0_1`
- 主输入事件表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_all_raw_rescreen_v0_1/tables/all_raw_vehicle_instability_primary_high_confidence_v0_1.csv`
- 正式样本目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/02_samples/vehicle_instability_highconf_v0_1`

## 构建规则

1. 只使用全原始车辆重筛得到的高置信车辆失稳事件。
2. 事件锚点来自非方向盘车辆动力学 `ay/roll_rate`，方向盘只作为事件后响应标签和评估元数据。
3. v0.1 要求每个事件具备完整 3 秒历史和 2 秒未来覆盖，因此从 908 个高置信事件中保留 906 个，排除 2 个。
4. 同一事件生成 3 个窗口：`pre1_label2_event_trigger`、`pre2_label2_old_main`、`pre3_label3_response_coverage`。
5. 默认正式切分为 `session_level_split`，同时保留 `random_event_split` 和 `subject_level_split`，但不使用任何标签统计分配 split。
6. 本版本不提取生理、脑电或连续风格窗口，只记录对应模态在原始文件层面的可用性和路径。

## 数量

- 输入高置信事件：908
- 可用事件：906
- 排除事件：2
- 样本行数：2718
- 主窗口样本数：906

## 窗口分布

             window_config_id  n_samples
    pre1_label2_event_trigger        906
         pre2_label2_old_main        906
pre3_label3_response_coverage        906

## 默认 session-level split

split  n_primary_samples
train                611
  val                156
 test                139

## 模态可用性，按主窗口样本计数

                 modality_flag  n_primary_samples
             vehicle_available                906
              physio_available                815
                 eeg_available                846
all_three_modalities_available                755

## 响应类型，按主窗口 eval-only 标签计数

eval_label_morphology  n_primary_samples
     multi_correction                642
   reverse_correction                192
          single_lobe                 72

## 无泄漏说明

- split 由事件、session 或 subject 标识的稳定哈希决定，不用方向盘未来标签和测试集统计。
- manifest 未做标准化；后续训练必须只在 train split 拟合 scaler。
- `eval_label_*` 字段来自未来方向盘标签，只允许用于评估分层、固定图和困难样本分析，不允许作为训练输入、split 决策或特征学习依据。
- 生理和脑电在本版本只记录原始文件是否可用，未抽取窗口，因此不会引入生理窗口泄漏。

## 关键输出

- `samples_master.csv/jsonl`
- `event_anchor_table.csv`
- `split_table.csv`
- `split_feasibility_report.csv`
- `sample_exclusion_reasons.csv`
- `label_eval_only_response_summary.csv`
