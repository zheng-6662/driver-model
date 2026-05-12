# Stage 7a 用户查看版：非 oracle 多候选选择协议 v0.1

## 为什么做

Stage 6e 发现 broad oracle 候选池上限很高，但当前可部署 selector 没有超过 RBF/KNN。这个阶段先把 Stage 7 的规则写清楚，防止后面把 best-of-K 或用真实标签选候选的结果误当成模型能力。

## 这个阶段检查了什么

- 固定候选池：RBF/KNN、ridge、template、keypoint 和 top-K 分支。
- 固定禁止信息：test label、test RMSE、oracle winner、测试集统计、生理/EEG/连续风格、驾驶员 ID。
- 固定选择规则：train 拟合选择器、val 选模型/阈值/校准、test 只最终评估。
- 固定评价：RMSE、错侧、大幅响应、困难样本、校准、coverage-risk、固定图和坏样本图。

## 当前发现

当前可以进入 Stage 7 的“协议准备”状态，但还不能说 Stage 7 模型有效。真正需要证明的是：不用真实标签选择候选时，selector 是否能稳定超过 RBF/KNN。

## 下一阶段是否可以继续

可以继续做 Stage 7 非 oracle 选择器设计和轻量实验；但生理/EEG 仍不能进入有效性结论。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_selection_protocol.csv`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_feature_guard_table.csv`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/tables/stage07a_gate_table.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/figures/stage07a_candidate_pool_rmse.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07a_non_oracle_selection_protocol_v0_1/figures/stage07a_protocol_gate_status.png`
