# 阶段 1 用户查看版总结：原始数据审计

更新时间：2026-05-12 10:44:41

## 这个阶段为什么做

旧流程已经证明继续堆模型会遇到物理解释问题：有些预测趋势像，但方向、幅值、尾段回正、反向修正和困难样本并不可靠。因此本阶段先回到原始 CSV，检查数据本身、时间轴和跨模态同步是否值得继续。

## 这个阶段检查了什么

- 只扫描三个原始目录下被试名文件夹内的 CSV，并给每个纳入文件生成 SHA256 哈希。
- 对原始车辆、原始生理、原始脑电 CSV 读取字段、行数、时间范围、缺失率和时间戳连续性。
- 按被试和记录号整理车辆/生理/脑电是否齐全。
- 初步计算不同模态的时间重叠。
- 抽查车辆、生理和脑电关键波形，生成采样率分布图。
- 列出后续最容易造成“看起来有效但其实泄漏”的风险点。

## 目前发现了什么

- 能定位到原始车辆 CSV 91 个、原始生理 CSV 82 个、原始脑电 CSV 85 个。
- 原始目录范围总计 258 个 CSV，其中原始传感器 CSV 258 个，派生特征/处理后 CSV 0 个。
- 三模态齐全的被试/记录组合有 76 个。
- 当前至少两模态有正时间重叠的组合有 91 个。
- 时间连续性、信号质量和 EEG 通道质量中都有需要复核的条目，不能直接跳到训练。

## 哪些结果可信

- 文件是否存在、文件大小、修改时间和 SHA256 哈希是可追溯的。
- 字段名、行数、时间范围、采样间隔初值是从原始 CSV 重新读取的。
- “旧锚点不能默认相信”这一判断可信，因为旧代码确实存在按响应峰值选锚点的逻辑，必须重新定义事件因果起点。

## 哪些结果还不能下结论

- 不能说连续风格一定有效。
- 不能说生理数据一定有效。
- 不能说 EMG 的旧收益没有动作结果泄漏。
- 不能说 EEG 教师一定有效。
- 不能说车辆/生理/脑电已经完全同步。
- 不能说 2 秒预测窗口一定覆盖了完整方向盘响应。

## 下一阶段是否可以继续

可以继续到阶段 2，但只能继续做“事件锚点与样本清单重建”，不能直接训练模型。阶段 2 的核心是把每个样本的锚点、输入窗口、标签窗口、模态可用性和泄漏风险写清楚。

## 推荐优先查看

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/raw_data_audit_summary_cn.md`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/subject_session_modality_matrix.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/modality_overlap_report.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/tables/leakage_risk_report.csv`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/figures/audit/modality_overlap_timeline_sample.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/figures/audit/raw_waveform_vehicle_sample.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/figures/audit/raw_waveform_physio_sample.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/01_audit/figures/audit/raw_waveform_eeg_sample.png`
