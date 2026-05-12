# 阶段 3 用户查看版：车辆基线坏样本错误分型

生成时间：2026-05-12

## 为什么做

阶段 3 v0.1 已经有车辆-only 基线，但只看 RMSE 不够。我们需要知道模型到底错在哪里，才能决定下一步是强化车辆模型，还是以后再看风格/生理是否能补充。

## 检查了什么

- 错侧。
- 大幅响应漏召回。
- 严重幅值不足。
- 多段修正漏检。
- 反向修正数量不匹配。
- 尾段漂移。
- 零线穿越错误。
- 峰值时间和启动延迟错误。
- 和旧 `vehicle_direct` deep 对照的坏样本重叠。

## 目前发现

test 样本 139 个。错误最多的标签是 `reversal_mismatch_flag`，数量 126。错侧样本 32 个，大幅响应漏召回 23 个，严重幅值不足 81 个，多段修正漏检 4 个，多段修正过度预测 42 个。

和旧 deep 对照时，旧 deep 的整体 RMSE 仍略低；但 formal ridge 在 92/139 个单样本上逐样本 RMSE 更小，说明 formal ridge 的坏样本更集中，不能只看平均数。

## 哪些结果可信

这些错误标签只来自 test 集评估结果，不参与训练，不参与 split，也不用于标准化。它们用于解释模型失败类型。

## 哪些结果还不能下结论

不能因为某些错误多，就说生理一定能解决。现在只能说明车辆-only 浅层基线在哪些物理响应上不够好。

## 下一阶段是否可以继续

建议先做更强的车辆时序/结构化响应基线，再进入风格或生理增量验证。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/stage03_vehicle_instability_error_analysis_v0_1_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/formal_error_flag_counts.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/figures/top_bad_sample_error_matrix.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_error_analysis_v0_1/tables/per_sample_error_taxonomy.csv`
