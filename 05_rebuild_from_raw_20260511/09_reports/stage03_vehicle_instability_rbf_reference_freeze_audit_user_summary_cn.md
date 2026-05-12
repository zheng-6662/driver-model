# 阶段 3 用户查看版：RBF 主参照冻结审计 v0.1

## 这个阶段为什么做

上一轮车辆-only 决策表显示，RBF KRR 仍是当前最稳的车辆-only 主参照，但它不是一个已经解决物理响应问题的模型。这一步正式回答：后面能不能固定 RBF 作为“车辆历史和事件信息本身能做到什么程度”的参照，同时避免把它误说成最终强模型。

## 这个阶段检查了什么

- RBF 是否可以作为 B 轨道后续增量实验的固定车辆-only 主参照。
- RBF 的主要失败类型是什么。
- top-K / oracle 上限是否会被误用为实际可部署模型性能。
- 是否可以进入连续风格阶段，是否可以进入生理/EEG 阶段。

## 目前发现了什么

- RBF test RMSE=0.533667，错侧率=0.225，大幅响应召回=0.750。
- 反向修正计数完全匹配率=0.000，这是当前最大物理缺陷。
- 失败类型中，反向修正计数不匹配覆盖 40/40，错侧 9/40，严重幅值不足 5/40，大幅响应漏召回 2/40。
- 结论是“有限冻结”：RBF 可以固定为 B 轨道后续增量实验的保守主参照，但不能宣称车辆-only 已经解决物理响应问题。

## 哪些结果可信

可信的是：RBF 是车辆-only、无 subject ID、无生理、无脑电、无连续风格的参照；错误类型来自 test 逐样本物理指标和坏样本复查表，不是只看 RMSE。

## 哪些结果还不能下结论

不能说 RBF 是最终强模型，不能说 top-K oracle 是实际性能，也不能说连续风格或生理已经有效。连续风格最多可以进入阶段 4 的协议设计和探索性验证；生理/EEG 仍阻塞。

## 下一阶段是否可以继续

可以进入阶段 4 的连续风格协议设计/探索性实验，但所有比较必须以固定 RBF 主参照为底线，并带置乱、分被试、物理指标和坏样本分析。生理/EEG 不能跳过阶段 4 直接验证。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/figures/rbf_reference_failure_profile.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/figures/rbf_reference_key_metrics.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_freeze_gate_table.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_rbf_reference_freeze_audit_v0_1/tables/rbf_reference_failure_profile.csv`
