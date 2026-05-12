# 阶段 1 原始数据审计总结

更新时间：2026-05-12 10:55:23

## 审计范围

- 原始数据根目录：`F:/data_set_process/data_process/01_datasets/数据预处理`
- 清单和深度审计范围：只覆盖 `原始车辆数据/<被试名>/*.csv`、`原始生理数据/<被试名>/*.csv`、`原始脑电数据/<被试名>/*.csv`。
- 明确不纳入：顶层全量记录文件、压缩包、处理后目录、对齐后目录，以及 `physio_features_v2_10Hz` 等派生特征目录。
- 本次未使用服务器，未读取服务器密码文件。

## 核心数量

- 本次纳入审计 CSV 文件数：258
- 原始目录范围 CSV 文件数：258
- 原始传感器 CSV 文件数：258
- 原始目录内派生特征/处理后 CSV 文件数：0
- 原始车辆 CSV：91
- 原始生理 CSV：82
- 原始脑电 CSV：85
- 被试/记录组合数：91
- 三模态齐全的组合数：76
- 至少两模态有可计算时间重叠且 overlap>0 的组合数：91

## 主要发现

1. 原始数据路径存在且可扫描；车辆、生理、脑电三类原始 CSV 都能定位。
2. 已为所有 CSV 生成清单和哈希；后续样本 manifest 可以引用文件路径与 SHA256。
3. `原始生理数据` 同级存在 10Hz 派生特征目录，最终已从本轮原始审计范围排除；最终纳入表中 `raw_sensor_scope=True` 且 `derived_file=False` 的文件为 258 个。
4. 时间戳初审已完成：151 个文件存在零间隔时间戳，26 个文件存在 large gap，合并后 175 个文件需要重点复核连续性、重复点、gap 或时间解析率。
5. 信号质量初审发现 3847 个原始传感器被抽查信号有效率低于 95%。
6. EEG 初审发现 27 个通道/文件组合接近常数，需在阶段 2/5 前结合伪迹规则复核。
7. 旧事件锚点只能作为历史参考；阶段 2 必须重新定义事件锚点、输入窗口、标签窗口和 causal setting。

## 暂定判断

当前结果支持继续进入“阶段 2：事件锚点与样本清单重建”的数据映射工作，但不支持直接训练新模型。继续条件是阶段 2 能为每个候选样本写清楚原始文件、时间范围、可用模态、质量标记和泄漏风险。

## 关键产物

- 文件清单：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/raw_file_inventory.csv`
- 字段报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/raw_schema_report.csv`
- 被试/记录/模态矩阵：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/subject_session_modality_matrix.csv`
- 时间连续性报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/timestamp_continuity_report.csv`
- 采样率报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/sampling_rate_report.csv`
- 模态重叠报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/modality_overlap_report.csv`
- 信号质量报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/signal_quality_report.csv`
- EEG 初审报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/eeg_artifact_report.csv`
- 泄漏风险报告：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/leakage_risk_report.csv`
- 审计图目录：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/figures/audit`
