# Stage 7b 用户查看版：非 oracle top-K selector 轻量实验 v0.1

## 为什么做

Stage 7a 固定了不能用 test 标签选候选的规则。本轮用已有 top-K/RBF 特征做一个轻量 selector，检查非 oracle 选择器是否能把 Stage 6e 的 oracle 上限转成实际可部署收益。

## 目前发现

- RBF/KNN 主参照 test RMSE=0.533667。
- val 选中的非 oracle policy：`logreg_balanced_c0_2__fallback_rbf_conf_lt_0.80`，test RMSE=0.533667，相对 RBF delta=+0.000000。
- 该 policy wrong-side=0.225，large recall=0.750，difficult RMSE=0.678907。
- 该 policy 在 test 上选择 RBF 的比例为 1.000；如果比例接近 1，说明当前 selector 实际只是退回主参照，没有带来新选择能力。
- 如果 gate 为 `no_upgrade`，说明当前轻量 selector 还不能升级主线。

## 可信边界

本轮 selector 输入只使用事件/道路上下文、候选概率、候选分歧和候选预测自身形态。`test_sample_rmse`、`wrong_side`、`best_candidate_oracle` 等 label-derived 字段没有进入输入。test 只用于最终评估。

## 下一步

如果当前轻量 selector 不升级，应继续改候选表示、导出完整预测轨迹差异特征，或考虑更明确的置信度 fallback；生理/EEG 继续阻塞。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_selected_policy_metrics.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_gate_table.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/tables/stage07b_feature_audit.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/figures/stage07b_selector_test_rmse.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07b_non_oracle_topk_selector_v0_1/figures/stage07b_coverage_risk.png`
