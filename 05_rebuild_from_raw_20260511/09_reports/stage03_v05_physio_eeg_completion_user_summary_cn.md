# v0.5 新样本集：生理/脑电补齐实验总结

生成时间：2026-05-20

## 这一步为什么做

前一轮 v0.5 生理机制验证中，脑电相关版本没有真正跑起来，主要原因不是脑电一定无效，而是旧脑电特征表按旧锚点/旧事件编号对齐，不能直接对应 v0.5 新样本锚点。  
本轮先重新提取了 v0.5 锚点前 2 秒脑电特征，再把这些特征接入旧流程粗细双头模型，然后补齐之前缺失的脑电直接输入、全生理融合和教师蒸馏版本。

## 固定实验条件

- 数据集：v0.5 服务器处理后样本集
- 样本划分：train/val/test = 953 / 260 / 163
- 测试被试：cwh / gf / tyy
- 验证被试：byx / gzj / yyl
- 模型基础结构：旧流程粗细双头
- 训练参数：seed=2026，40 epochs，batch=64，lr=0.001，cuda
- 运行位置：AutoDL 服务器，2 张 RTX 4080 SUPER
- 本轮不写入服务器密码到任何日志或报告

## 脑电对齐结果

- v0.5 manifest 总样本：1388
- 成功提取严格锚点前脑电特征：1164
- 缺失原因主要是部分记录没有清洗后的脑电 FIF，少量样本锚点太靠前或超过脑电时长
- 按旧流程 loader 保留后的训练检查中，脑电训练集有效比例约为 0.797

这说明：脑电不是完全不可用；之前脑电版本跑不起来，主要是“旧脑电特征和 v0.5 新锚点不匹配”。

## 主要结果

按 test RMSE 排序，当前最强是：

| 排名 | 版本 | 含义 | test RMSE | 主阶段 | 尾段 | selection |
|---:|---|---|---:|---:|---:|---:|
| 1 | T1 | 脑电教师 -> 车辆 + 连续风格学生 | 0.3107 | 0.2170 | 0.2716 | 0.7877 |
| 2 | SF4 | 车辆 + 连续风格 + 脑电直接输入 | 0.3142 | 0.2279 | 0.2598 | 0.7639 |
| 3 | T2 | 非脑电生理教师 -> 车辆 + 连续风格学生 | 0.3247 | 0.2161 | 0.2907 | 0.7889 |
| 4 | SF2 | 车辆 + 连续风格 + 皮电 | 0.3329 | 0.2344 | 0.2701 | 0.7832 |
| 5 | T3 | 全生理教师 -> 车辆 + 连续风格学生 | 0.3375 | 0.2171 | 0.2849 | 0.7834 |
| 6 | B0 | 车辆-only 粗细双头 | 0.3386 | 0.2184 | 0.3105 | 0.8206 |

## 当前可以怎么理解

1. 脑电路线重新变得有价值。  
   SF4 和 T1 都明显优于 B0 车辆-only，其中 T1 是当前 seed2026 最强版本。

2. 脑电教师比全生理教师更干净。  
   T1 = 0.3107，而 T3 = 0.3375、T4 = 0.3804。说明把 HR/EDA/EMG/EEG 全部作为教师不一定更好，可能引入了额外噪声。

3. 脑电直接输入也有效，但部署意义要谨慎。  
   SF4 = 0.3142，说明脑电在当前样本上确实有预测信息；但脑电推理部署难度高，所以 T1 这种“训练时用脑电、推理时不用脑电”的路线更有应用价值。

4. 非脑电生理并没有被否定。  
   T2 仍然优于 B0，说明 HR/EDA/EMG 组合更适合做训练期教师或状态监督，而不是简单直接拼接。

5. 简单全生理直接融合不理想。  
   C3/C4/A3 整体都不如 SF4/T1/T2，说明“信号越多越好”不成立，后续应该做选择性融合或可靠性判断。

## 当前不能下的结论

- 不能只凭 seed2026 就说 T1 是最终主线。
- 不能说“脑电一定适合部署输入”，因为脑电采集和实际部署成本仍然很高。
- 不能说全生理无效，只能说当前简单融合和当前全生理教师设计不如脑电教师。
- 不能只看 RMSE，还要继续看预测图里方向、幅值、错侧、尾段是否更合理。

## 用户优先查看

- 白底总表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/figures/v05_physio_eeg_result_table_white.png`
- 指标柱状图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/figures/v05_physio_eeg_metric_overview.png`
- 脑电直接输入/全生理融合曲线对比：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/figures/v05_multiversion_overlay_eeg_direct.png`
- 教师蒸馏曲线对比：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/figures/v05_multiversion_overlay_teacher.png`
- 完整指标表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/tables/v05_physio_comparison_table.csv`
- 运行状态表：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_v05_physio_mechanism_comparison/tables/v05_physio_run_status.csv`

## 下一步建议

建议先人工看两张多版本曲线叠加图，重点看 T1/SF4 是否只是 RMSE 变小，还是在方向、幅值和尾段形态上也更合理。  
如果图上也能接受，下一步优先补 T1、SF4、T2 的 seed2027/2028；不建议优先补 T4 或 A3。
