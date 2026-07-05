# 连续车辆源数据审计报告（2026-06-24）

- 生成时间：2026-06-24 20:39:10
- 审计边界：只读扫描连续车辆 CSV；不使用样本集、训练标签、模型预测；不修改原始数据。
- 审计方法：按时序数据质量常见框架检查资产盘点、完整性、一致性、唯一性、时间轴、分布尾部和下游风险。
- 输出目录：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624`

## 1. 资产盘点

- 候选文件总数：358
- 纳入连续车辆审计文件数：182
- 纳入文件覆盖被试：18
- 纳入文件覆盖记录键：91
- 纳入文件总时长：约 25.31 小时
- 主车辆层文件数：91
- 补充车辆 200Hz 层文件数：91

候选文件类别计数见 `tables/vehicle_file_inventory.csv`；核心计数如下：

| file_class                       |   count |
|:---------------------------------|--------:|
| main_vehicle_aligned_cleaned     |      91 |
| supplement_vehicle_fixed_200hz   |      91 |
| supplement_eeg_in_vehicle_dir    |      85 |
| supplement_physio_in_vehicle_dir |      82 |
| main_vehicle_dir_non_vehicle     |       5 |
| main_vehicle_aux_or_derived      |       4 |

纳入审计文件按源层分开看如下：

| source_layer                   |   file_count |   recording_count |   total_duration_h |   median_dt_ms |   median_nominal_hz |   files_nominal_hz_outside_150_250 |   files_high_key_missing_rate |   files_low_road_ref_ok_rate |
|:-------------------------------|-------------:|------------------:|-------------------:|---------------:|--------------------:|-----------------------------------:|------------------------------:|-----------------------------:|
| main_vehicle_aligned_cleaned   |           91 |                91 |            12.6798 |              5 |                 200 |                                  0 |                             0 |                           32 |
| supplement_vehicle_fixed_200hz |           91 |                91 |            12.634  |              5 |                 200 |                                 45 |                            83 |                            0 |

## 2. 主要发现

### 1. [P1] 补充采集的“车辆清理后”目录存在明显命名混杂

- 证据：该目录/文件名里带 vehicle，但检出非车辆字段文件 167 个；其中 PhysioLAB=82，EEG/加速度=85。
- 影响：如果脚本只按文件名 glob *_vehicle_fixed_200Hz.csv，会把生理/EEG 文件误当车辆源，导致字段缺失或错配。
- 下一步检查：后续读取补充采集数据时必须按字段白名单判定车辆源，而不是按目录/文件名判定。

### 2. [P2] 连续车辆源文件规模足够做源级统计，而不是只看样本集

- 证据：纳入连续车辆审计 182 个文件，18 名被试，91 个记录键，总时长约 25.31 小时。
- 影响：可以直接在源记录层面检查采样、缺失、道路标定和动作分布，再回头解释样本/窗口问题。
- 下一步检查：把源级异常记录映射回后续样本锚点，检查坏样本是否集中来自少数源记录。

### 3. [P2] 时间戳没有大断裂，但补充层存在采样率/行数层级不一致

- 证据：文件级 median_dt_ms 的中位数为 5.000；存在 gap>50ms 的文件 0 个，非单调时间文件 0 个；nominal_hz 超出 150-250 的文件 45 个，median_dt 不接近 5ms 的文件 42 个。
- 影响：主车辆层可按 200Hz 使用；补充层有些记录更像重采样/插值后的不同时间轴，不能和主层直接混用。
- 下一步检查：优先看 source_layer_summary.csv 与 recording_cluster_summary.csv，确认后续唯一源层。

### 4. [P2] 主车辆层已有道路参考字段，但低覆盖记录会影响 road/curve 分层判断

- 证据：有道路参考字段的文件 91 个，其中 ref_nn_ok_rate<95% 的文件 32 个。
- 影响：如果低覆盖记录进入 curve/road 分层或横向偏移筛选，会把道路参考误差混入行为判断。
- 下一步检查：对低 ref_nn_ok_rate 或 ref_nn_dist_m_p95 偏高记录做地图/道路参考复核。

### 5. [P1] 主车辆层字段完整，补充车辆层字段缺失较多，必须分层使用

- 证据：关键车辆字段平均缺失率>20% 的纳入文件 83 个；其中主 vehicle_aligned_cleaned 层 0 个，补充 vehicle_fixed_200Hz 层 83 个。
- 影响：主层更适合做当前车辆建模和道路分层；补充层若直接混入，会让 speed/road/steer/ay 联合判断静默退化。
- 下一步检查：训练/样本构建前固定唯一车辆源层；若必须用补充层，先重建字段完整性和采样率规则。

### 6. [P2] 同一记录在主目录和补充目录之间需要保留 lineage 对照

- 证据：记录簇 91 个；包含主车辆层的簇 91 个；行数不一致簇 47 个；规范车辆信号抽样哈希不一致簇 83 个。
- 影响：如果不同阶段混用 main aligned 和 supplement fixed 版本，可能出现同名记录但信号/行数不完全一致。
- 下一步检查：用 recording_cluster_summary.csv 决定后续唯一源层，并记录每个样本来自哪个源层。

### 7. [P2] 被试层面的总时长和横向动作秒数不均衡

- 证据：总时长最高：zx=167.2min, hzh=122.8min, gf=100.6min；最低：zt=20.8min, xst=24.8min, lx=42.5min。
- 影响：车辆-only 模型若按随机样本切分，容易把被试/记录分布差异误当可泛化信号。
- 下一步检查：继续坚持 subject/session-level split，并检查难样本是否集中在动作秒数少或道路覆盖异常的被试。

## 3. 时序质量摘要

|   median_dt_ms_median |   nominal_hz_median |   gap_gt_20ms_files |   gap_gt_50ms_files |   nonmonotonic_files |   duplicate_storage_files |
|----------------------:|--------------------:|--------------------:|--------------------:|---------------------:|--------------------------:|
|                     5 |                 200 |                   0 |                   0 |                    0 |                         0 |

时间轴风险最高的记录：

| file_id   | subject   |       recording_key | source_layer                 |   rows |   duration_s |   median_dt_ms |   max_dt_ms |   gap_gt_50ms_count | suspect_flags                                                           |
|:----------|:----------|--------------------:|:-----------------------------|-------:|-------------:|---------------:|------------:|--------------------:|:------------------------------------------------------------------------|
| F0001     | byx       | 2025_09_28_17_05_51 | main_vehicle_aligned_cleaned | 113087 |      565.43  |              5 |           5 |                   0 | lateral_distance_tail_needs_reference_review                            |
| F0004     | byx       | 2025_09_28_17_15_52 | main_vehicle_aligned_cleaned | 106751 |      533.75  |              5 |           5 |                   0 |                                                                         |
| F0005     | byx       | 2025_09_28_17_25_18 | main_vehicle_aligned_cleaned | 117075 |      585.37  |              5 |           5 |                   0 | low_road_reference_ok_rate;lateral_distance_tail_needs_reference_review |
| F0006     | byx       | 2025_09_28_17_35_43 | main_vehicle_aligned_cleaned | 117172 |      585.855 |              5 |           5 |                   0 | lateral_distance_tail_needs_reference_review                            |
| F0007     | byx       | 2025_09_28_17_46_00 | main_vehicle_aligned_cleaned | 109483 |      547.41  |              5 |           5 |                   0 | lateral_distance_tail_needs_reference_review                            |
| F0010     | cwh       | 2025_09_26_19_35_47 | main_vehicle_aligned_cleaned | 109883 |      549.41  |              5 |           5 |                   0 | lateral_distance_tail_needs_reference_review                            |
| F0011     | cwh       | 2025_09_26_19_45_40 | main_vehicle_aligned_cleaned | 107796 |      538.975 |              5 |           5 |                   0 | lateral_distance_tail_needs_reference_review                            |
| F0012     | cwh       | 2025_09_26_19_56_16 | main_vehicle_aligned_cleaned | 105195 |      525.97  |              5 |           5 |                   0 | lateral_distance_tail_needs_reference_review                            |
| F0014     | gf        | 2025_09_26_10_03_00 | main_vehicle_aligned_cleaned | 131854 |      659.265 |              5 |           5 |                   0 | low_road_reference_ok_rate                                              |
| F0015     | gf        | 2025_09_26_10_18_49 | main_vehicle_aligned_cleaned | 116766 |      583.825 |              5 |           5 |                   0 | low_road_reference_ok_rate;lateral_distance_tail_needs_reference_review |
| F0016     | gf        | 2025_09_26_10_30_12 | main_vehicle_aligned_cleaned | 113199 |      565.99  |              5 |           5 |                   0 | low_road_reference_ok_rate;lateral_distance_tail_needs_reference_review |
| F0017     | gf        | 2025_09_26_10_40_59 | main_vehicle_aligned_cleaned | 118265 |      591.32  |              5 |           5 |                   0 | lateral_distance_tail_needs_reference_review                            |

## 4. 被试分布摘要

| subject   |   included_file_count |   recording_count |   total_duration_min |   gap_gt_50ms_files |   suspect_file_count |   drive_seconds_speed_gt_5 |   lateral_action_seconds |
|:----------|----------------------:|------------------:|---------------------:|--------------------:|---------------------:|---------------------------:|-------------------------:|
| zx        |                    26 |                13 |             167.154  |                   0 |                   23 |                       9261 |                     7259 |
| hzh       |                    12 |                 6 |             122.779  |                   0 |                   12 |                       7240 |                     5088 |
| zdq       |                    12 |                 6 |              94.7192 |                   0 |                   12 |                       5652 |                     3747 |
| cwh       |                    12 |                 6 |              90.5603 |                   0 |                   11 |                       5364 |                     3194 |
| rjy       |                    12 |                 6 |              92.1932 |                   0 |                   11 |                       5390 |                     3693 |
| gzj       |                    12 |                 6 |              95.951  |                   0 |                   10 |                       5686 |                     4014 |
| gf        |                    10 |                 5 |             100.624  |                   0 |                   10 |                       5936 |                     3402 |
| jy        |                    10 |                 5 |              98.4127 |                   0 |                   10 |                       5702 |                     3189 |
| yzy       |                    10 |                 5 |              86.8064 |                   0 |                   10 |                       4956 |                     4259 |
| byx       |                    10 |                 5 |              91.17   |                   0 |                    9 |                       5393 |                     4545 |
| zxy       |                    12 |                 6 |              91.151  |                   0 |                    8 |                       5338 |                     4910 |
| txj       |                    10 |                 5 |              88.9792 |                   0 |                    8 |                       5174 |                     4493 |
| lxy       |                     8 |                 4 |              57.7028 |                   0 |                    7 |                       3350 |                     2814 |
| tyy       |                     8 |                 4 |              70.3448 |                   0 |                    7 |                       3954 |                     2906 |
| yyl       |                     8 |                 4 |              82.1705 |                   0 |                    7 |                       4751 |                     3610 |
| lx        |                     6 |                 3 |              42.5191 |                   0 |                    6 |                       2488 |                     2106 |
| xst       |                     2 |                 1 |              24.7623 |                   0 |                    2 |                       1404 |                     1000 |
| zt        |                     2 |                 1 |              20.8296 |                   0 |                    1 |                       1222 |                      870 |

## 5. 明细文件

- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\vehicle_file_inventory.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\file_vehicle_quality_summary.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\vehicle_numeric_column_summary.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\recording_cluster_summary.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\subject_vehicle_summary.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\source_layer_summary.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\road_type_summary.csv`
- `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\01_audit\vehicle_source_audit_20260624\tables\vehicle_source_audit_findings.csv`

## 6. 图

- `05_rebuild_from_raw_20260511/01_audit/vehicle_source_audit_20260624/figures/vehicle_recording_duration_hist.png`
- `05_rebuild_from_raw_20260511/01_audit/vehicle_source_audit_20260624/figures/vehicle_duration_by_subject.png`
- `05_rebuild_from_raw_20260511/01_audit/vehicle_source_audit_20260624/figures/vehicle_audit_flag_counts.png`

## 7. 结论边界

- 这轮审计说明哪些连续车辆记录更可信、哪些目录/文件容易误读，但它不自动判定任何样本标签正确。
- 如果要解释现有模型失败样本，下一步应把 `file_id/recording_key` 映射回锚点窗口，检查失败是否来自源记录质量、道路参考、时间轴间隔、还是任务可观测性。
- 补充采集目录必须先字段分类再读取，不能只凭 `vehicle_fixed_200Hz` 文件名。
