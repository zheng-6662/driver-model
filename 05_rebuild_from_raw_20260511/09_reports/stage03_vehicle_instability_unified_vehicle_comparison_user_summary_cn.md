# 阶段 3 用户查看版：车辆-only 模型统一对照

## 这个阶段为什么做

前面已经分别跑过 formal ridge、旧 `vehicle_direct`、RBF/KNN/template 和真正 Transformer。单独看每次结果容易误判，所以这一步把它们放到同一张表里比较，避免只看 RMSE。

## 这个阶段检查了什么

- 整体 RMSE。
- 方向是否预测反了。
- 大幅响应有没有召回。
- 幅值是否严重不足。
- 尾段误差和漂移风险。
- 反向修正和多段修正是否能识别。
- 坏样本是否集中在同一批事件。

## 目前发现了什么

1. formal ridge 是最低公平参照，test RMSE=0.649341。
2. 旧 `vehicle_direct active` test RMSE=0.637366，只能作历史对照。
3. RBF/KNN/template 的 RMSE 更低，KNN template test RMSE=0.516941，但 KNN 训练集几乎记住模板，风险很高。
4. 真正 Transformer test RMSE=0.567162，比 formal ridge 好，但还没有解决多段修正问题。

## 哪些结果可信

这些结果都来自同一批正式失稳样本、同一主窗口和同一 session-level test 集。它们都没有使用生理、脑电、连续风格或驾驶员 ID，所以可以作为车辆-only 对照。

## 哪些结果还不能下结论

不能说 KNN/RBF/Transformer 任何一个已经是最终强车辆主线。KNN/RBF 可能受模板记忆或局部分布影响，Transformer 还漏多段修正。也不能根据这些结果说生理或连续风格有效。

## 下一阶段是否可以继续

可以继续，但下一步仍属于阶段 3：先做强车辆基线稳健性验证和坏样本复盘。只有车辆-only 主参照冻结后，才适合进入连续风格和生理增量验证。

## 推荐优先查看

1. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\tables\unified_vehicle_comparison_metrics_test.csv`
2. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\tables\unified_vehicle_candidate_decision_table.csv`
3. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\figures\unified_vehicle_key_metrics_test.png`
4. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\figures\unified_vehicle_physical_failure_heatmap_test.png`
5. `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\stage03_vehicle_instability_unified_vehicle_comparison_v0_1\figures\unified_vehicle_top_bad_overlap.png`
