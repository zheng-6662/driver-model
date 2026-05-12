# Stage 6d 用户查看版：RBF/KNN 可靠性门控 v0.1

## 为什么做

这里不是在把 Transformer 当主线继续训练。当前可用的车辆-only 主参照仍然是 RBF/KNN 类强基线；Stage 6c 的 RF selector 只是尝试在“保留 RBF/KNN 预测”和“切换到 keypoint 候选预测”之间做选择。Stage 6c 虽然改善了错侧率和大幅响应召回，但 RMSE 退化，所以这一阶段用更保守的 reliability gate 控制错选 keypoint 的风险。规则只用 val 选择，test 只做最终评估。

## 目前发现

- RBF/KNN 主参照 test RMSE=0.533667。
- 当前最好的 reliability policy：`val_rmse_noninferior_conservative`，test RMSE=0.534545，相对 RBF/KNN delta=+0.000878。
- 该 policy wrong-side=0.225，large recall=0.750。
- gate 若显示 `no_upgrade`，说明 reliability gate 仍不能升级为主车辆路线，只能作为诊断候选。

## 当前判断

如果保守门控仍不能同时改善 RMSE 和物理指标，Stage 6 的 selector 路线需要暂时降级。下一步应考虑更直接的多假设候选生成/选择，或回到车辆-only 表示和样本规则复查；生理/EEG 继续阻塞。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_gate_table.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/tables/reliability_gate_policy_metrics.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/figures/reliability_gate_test_rmse.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06d_reliability_gate_v0_1/figures/reliability_gate_physical_metrics.png`
