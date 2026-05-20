# v0.5 脑电数据审计与特征提取说明

## 本轮结论

- 原始脑电 CSV 用于确认字段、时间戳和原始通道结构。
- 建议建模优先使用已经完成清洗、重采样和 ICA 处理的 `*_eeg_raw_clean_resamp200_ica_final_qc.fif`。
- 旧脑电特征表是按横滚峰值前 2 秒提取，不能直接用于 v0.5 新锚点。
- 本脚本按 v0.5 manifest 的 `anchor_s` 重新提取锚点前脑电特征，默认窗口为 `[anchor_s-2s, anchor_s)`，不使用未来标签窗口。

## 数据概况

- v0.5 manifest 样本数：1388
- v0.5 记录数：87
- 有清洗脑电 FIF 的记录数：74/87
- 有原始脑电 CSV 的记录数：81/87
- 清洗后脑电采样率分布：{200.0: 74}

## 原始脑电字段

- 典型原始 CSV 包含 `ID`、`StorageTime`、32 个 `LSLOutletStreamName-EEG|channelX` 脑电通道，以及 3 个 `Accelerometer` 加速度通道。
- 旧预处理代码会丢弃开头 EEG 全 NaN 行、插值少量 NaN、用 `StorageTime` 估计真实采样率、1-40Hz 带通、50Hz 工频滤波、平均参考、ICA 清理，并重采样到 200Hz。

## 特征设计

- 频段：theta 4-8Hz、alpha 8-13Hz、beta 13-30Hz、gamma 30-45Hz。
- 区域：额叶、颞叶、枕叶、中央区、顶叶。
- 输出包括各区域频段功率、对数功率、theta+alpha/beta、theta/beta、alpha/beta、gamma 相对功率、额叶 alpha 不对称，以及和旧流程兼容的 8 个核心特征。
- 同时输出窗口长度、有限值比例、通道数、清洗文件路径和状态字段，方便排查缺失或异常样本。

## 提取结果

- 已输出特征样本数：1388
- `eeg_status` 分布：
  - ok: 1164
  - missing_eeg_fif: 210
  - not_enough_pre_anchor: 10
  - anchor_beyond_eeg_duration: 4
- 分 split 可用情况：
  - test: ok 159/165
  - train: ok 761/960
  - val: ok 244/263

## 下一步建议

1. 先把本特征表接入 v0.5 机制实验的可用性检查。
2. 如果 `eeg_status=ok` 覆盖 train/val/test 都足够，再跑 `车辆+脑电`、`车辆+连续风格+脑电` 和脑电教师版本。
3. 如果覆盖不足，先查缺失记录是否没有清洗 FIF，不能直接用旧 roll-peak 特征补。
4. 如果要使用 `anchor_s` 后 0.5 秒早期动作窗口，必须单独命名为 early-window 特征，不能和严格 pre-anchor 特征混在一起。