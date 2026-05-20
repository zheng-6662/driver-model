# v0.5 脑电原始数据审计与特征提取说明

更新时间：2026-05-20

## 这一步为什么做

之前 v0.5 生理机制实验里没有真正跑脑电版本，原因不是“脑电一定没用”，而是旧脑电特征表和当前 v0.5 新样本锚点对不上。

旧脑电特征表是按“横滚峰值前 2 秒”提取的，字段里用的是 `event_row_index` 和 `roll_peak_s`。当前 v0.5 样本用的是新的 `anchor_s`，代表新筛选后的事件锚点。如果直接把旧表接进模型，就会变成“旧事件的脑电特征 + 新事件标签”，时间语义不一致，结论不可靠。

所以这一步先不训练模型，而是重新检查原始脑电数据，并按 v0.5 的 `anchor_s` 重新提取脑电特征。

## 原始脑电数据有什么

本地路径：

`F:\data_set_process\data_process\01_datasets\数据预处理\原始脑电数据`

目前扫描到 18 个被试目录：

`byx, cwh, gf, gzj, hzh, jy, lx, lxy, rjy, txj, tyy, xst, yyl, yzy, zdq, zt, zx, zxy`

典型原始脑电 CSV 包含：

- `ID`
- `StorageTime`
- 32 个脑电通道：`LSLOutletStreamName-EEG|channel0` 到 `channel31`
- 3 个加速度通道：`LSLOutletStreamName-Accelerometer|channel0` 到 `channel2`

原始脑电文件开头可能存在 EEG 全 NaN 行，所以不能直接取前几行就认为脑电缺失。旧预处理流程已经考虑了这一点。

## 现有脑电预处理做了什么

项目中已有清洗后的脑电文件，主要路径：

`F:\data_set_process\data_process\01_datasets\多模态数据\被试数据集合\<被试>\eeg_clean`

清洗后的主文件形式是：

`*_eeg_raw_clean_resamp200_ica_final_qc.fif`

旧预处理代码显示主要处理流程是：

- 用 `StorageTime` 估计真实采样率；
- 删除开头 EEG 全 NaN 行；
- 对少量 NaN 做插值；
- 建立标准通道名；
- 1-40 Hz 带通滤波；
- 50 Hz 工频滤波；
- 平均参考；
- ICA 去伪迹；
- 重采样到 200 Hz；
- 输出清洗后的 FIF 文件。

因此，后续特征提取优先使用清洗后的 FIF，而不是直接用原始 CSV 重新做全部预处理。原始 CSV 主要用于审计字段、时间戳和原始数据是否存在。

## 时间对齐怎么做

这次不是使用旧 `roll_peak_s`，而是使用 v0.5 manifest 里的 `anchor_s`。

v0.5 manifest 路径：

`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_processed_datasets\stage03_v05_server_aligned_subject_oldflow_fair09\tables\oldflow_fair09_vehicle_only_server_aligned_v05_subject_split_manifest.csv`

对齐方式：

1. 每个样本读取 `subj`、`recording_id` 和 `anchor_s`。
2. 根据 `subj + recording_id` 找对应清洗脑电 FIF。
3. 在该清洗脑电记录中截取 `[anchor_s - 2s, anchor_s)`。
4. 这个窗口只用锚点前信息，不包含预测标签窗口，所以是严格无未来泄漏的版本。
5. 额外计算一个更早的基线窗口 `[anchor_s - 8s, anchor_s - 4s)`，用于得到“相对基线变化”特征。

已抽查多条记录，原始车辆和原始脑电的开始时间差通常是几毫秒到几十毫秒。例如 byx 第一条记录车辆起点为 `2025-09-28 17:05:51.890`，脑电起点为 `2025-09-28 17:05:51.902`，相差约 12 ms。这个误差相对 2 秒脑电窗口很小，因此当前先按同一记录内的秒级时间轴对齐。

## 提取了哪些脑电特征

本轮先提取传统频域特征，方便和旧流程保持可解释性：

- 频段：
  - theta：4-8 Hz
  - alpha：8-13 Hz
  - beta：13-30 Hz
  - gamma：30-45 Hz
- 脑区：
  - 额叶
  - 颞叶
  - 枕叶
  - 中央区
  - 顶叶
- 特征类型：
  - 各脑区各频段功率；
  - 对数功率；
  - theta+alpha/beta；
  - theta/beta；
  - alpha/beta；
  - gamma 相对功率；
  - 额叶 alpha 不对称；
  - 当前窗口相对更早基线窗口的变化。

同时保留了和旧流程兼容的 8 个核心字段：

- `Frontal_alpha_asym`
- `Occipital_ta_beta`
- `Frontal_ta_beta`
- `Temporal_ta_beta`
- `Occipital_alpha_abs`
- `Temporal_gamma_rel`
- `Occipital_gamma_rel`
- `Frontal_gamma_rel`

## 当前提取结果

v0.5 manifest 总样本数：1388

成功提取严格锚点前 2 秒脑电特征：1164

失败或不可用原因：

- 缺少清洗后的脑电 FIF：210
- 锚点前不足 2 秒：10
- 锚点超过脑电记录时长：4

按数据划分看：

| 划分 | 样本数 | 成功提取 | 成功率 |
|---|---:|---:|---:|
| train | 960 | 761 | 79.27% |
| val | 263 | 244 | 92.78% |
| test | 165 | 159 | 96.36% |

记录级别看：

- v0.5 涉及记录数：87
- 有清洗脑电 FIF 的记录数：74
- 有原始脑电 CSV 的记录数：81

缺少清洗 FIF 的记录主要集中在：

- hzh：3
- lxy：1
- rjy：3
- txj：1
- yyl：1
- zdq：3
- zt：1

## 当前怎么理解

这一步说明：脑电并不是完全没有数据，也不是不能接入 v0.5。真正的问题是，旧脑电特征是按旧横滚峰值提取的，不能直接用于新锚点。

现在已经生成了按 v0.5 锚点重新对齐的脑电特征表，测试集覆盖率较高，验证集也较高，训练集有一部分缺失。后续可以先做两种策略：

1. 只使用脑电可用样本做公平对照；
2. 对缺失脑电样本使用缺失掩码，而不是把缺失直接当 0 或平均值。

## 后续建议

下一步不是直接说“脑电有效”，而是：

1. 先把这张新脑电特征表接入 v0.5 机制实验；
2. 重新跑 `车辆 + 脑电`、`车辆 + 连续风格 + 脑电`；
3. 再跑脑电教师版本；
4. 同时记录脑电缺失掩码，避免模型把“有没有脑电文件”当成被试或记录线索；
5. 如果脑电直接输入不稳定，但教师版本稳定改善困难样本、方向或幅值，再把脑电定位为训练期教师，而不是部署期输入。

## 产物位置

脚本：

`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v05_extract_eeg_features.py`

输出目录：

`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features`

主要表格：

`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_recording_inventory.csv`

`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_features_pre_anchor_hist2s.csv`

`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\tables\v05_eeg_feature_availability_summary.csv`

技术说明：

`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_v05_eeg_features\stage03_v05_eeg_feature_extraction_report_cn.md`
