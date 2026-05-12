# 阶段 3 用户查看版：旧 `vehicle_direct` 全量车辆-only 对照 clean v0.1

生成时间：2026-05-12

## 为什么做

你要求先用之前的旧代码测试这批重新筛选出来的车辆失稳样本，所以这次没有进入风格、生理、脑电路线，而是只跑旧 `vehicle_direct` 车辆-only 深度模型，看旧代码在新失稳样本上到底能做到什么程度。

## 检查了什么

- 906 个可用高置信车辆失稳事件。
- session-level 切分：train 611、val 156、test 139。
- 旧 `vehicle_direct` 全量训练，不是 smoke run。
- 固定预测图和坏样本图。
- 分被试结果和分响应类型结果。

## 目前发现

旧脚本选择的 active checkpoint 测试集 RMSE 为 0.637366。需要注意，这次是用 clean manifest 跑出的可信旧代码对照；此前直接读原始 CSV 的 run 因缺失点被旧代码填 0，已标为无效诊断。当前结果仍然有物理错误：错侧率 0.129496，严重幅值不足率 0.683453。所以它能拟合一部分轨迹，但还不能说明方向、幅值和复杂响应都可靠。

## 哪些结果可信

- 本次确实是全量 run，不是 96/32/32 的 smoke run。
- 输入只来自车辆，不含生理、脑电、连续风格。
- 固定图和坏样本图是按固定规则生成的，不是挑好看的图。

## 哪些结果还不能下结论

- 不能说这是新流程最终强车辆基线。
- 不能说旧模型已经解决车辆失稳响应预测。
- 不能说连续风格、生理、脑电有效。
- 不能把 `structure_best` 的结果直接当主结果，因为旧脚本本次 active 选择规则是 `legacy_rmse`。

## 下一阶段是否可以继续

可以继续，但建议下一步不是直接上生理，而是把这 906 个高置信失稳事件整理成新流程正式 `samples_master`，再建立新流程强车辆基线。旧代码结果只作为历史对照和坏样本来源。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/09_reports/oldcode_vehicle_direct_full_clean_on_instability_v0_1_cn.md`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_fixed_predictions_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/figures/oldcode_vehicle_direct_full_bad_samples_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_metrics.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/oldcode_vehicle_direct_full_clean_on_instability_v0_1/tables/oldcode_vehicle_direct_full_per_sample_metrics.csv`
