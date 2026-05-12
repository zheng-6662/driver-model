# 阶段 3 用户查看版：强车辆基线稳健性验证

## 为什么做

之前 KNN/RBF 的 RMSE 很低，但这不一定代表它们真正学到了可泛化规律。这个阶段专门检查：换成跨被试划分或换输入窗口后，低 RMSE 是否还稳定。

## 检查了什么

- 主 2 秒窗口的 random-event、session-level、subject-level 对照。
- 事件前 1 秒和前 3 秒窗口敏感性。
- RBF、KNN、方向门控 KNN、峰值缩放模板与 formal ridge 的比较。

## 目前发现

subject-level 主窗口中，val 选择模型是 `rbf_kernel_ridge_context_no_subject`，test RMSE=0.609792，formal RMSE=0.672788。

## 还不能下什么结论

KNN 即使 test RMSE 低，只要 train RMSE 仍接近 0，就要继续按模板记忆风险处理。RBF/KNN 是否能作为主线，还需要结合固定图、坏样本图和跨被试物理指标判断。

## 下一步

继续阶段 3：复盘 subject-level 和窗口敏感性下的坏样本，决定是否转向响应分解、关键点残差或多假设车辆模型。仍不进入生理/连续风格有效性结论。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_decision_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/tables/strong_vehicle_robustness_metrics.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_rmse_heatmap.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_large_recall_heatmap.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_strong_vehicle_robustness_v0_1/figures/strong_vehicle_robustness_reversal_heatmap.png`
