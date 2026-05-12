# 阶段 3 用户查看版：车辆-only 主参照决策表 v0.2

## 这个阶段为什么做

前面已经补跑了 RBF/KNN、direct Transformer、结构化 Transformer、keypoint+residual、RBF/keypoint selector、top-K 和 top-K 可靠性回退。单看某一次结果容易误把 oracle 上限或弱候选当成主线，所以这一步把阶段 3 的车辆-only 结果放到一张决策表里。

## 这个阶段检查了什么

- 哪个结果可以作为当前车辆-only 主参照。
- 哪些模型只是历史/诊断/弱候选。
- 哪些模型是 no-go。
- 哪些只是事后 oracle 上限，不能作为可部署结果。
- 是否已经允许进入连续风格、生理或 EEG 增量验证。

## 目前发现了什么

- 当前主参照仍是 RBF KRR：test RMSE=0.533667，错侧率=0.225，大幅响应召回=0.750。
- keypoint+residual 错侧率更低、大幅召回更高，但 test RMSE=0.548994，困难样本 RMSE=0.728866，不能单独替代 RBF。
- top-K fallback test RMSE=0.542071，没有超过 RBF，所以可靠性选择 v0.1 是 no-go。
- best-of-RBF+topK oracle RMSE=0.415652，说明候选池有潜力，但这是事后上限，不能作为真实模型表现。

## 哪些结果可信

可信的是：这些对照都限制在车辆-only 输入，不使用生理、脑电、连续风格或 subject ID；选择器和阈值均按 train/val/test 协议处理，test 只报告。

## 哪些结果还不能下结论

不能说 top-K 已经解决问题，不能说 keypoint 结构已经优于 RBF，也不能因为 oracle 上限好就进入生理或风格结论。当前也不能说强车辆基线已经完全冻结，因为 RBF 的错侧、反向修正和多段修正问题仍未闭环。

## 下一阶段是否可以继续

可以继续阶段 3。当前建议有两条：要么把 RBF KRR 接受为保守主参照并写清物理缺陷，进入阶段 4 前再做一次冻结审查；要么继续做更强的车辆-only 分响应类型/关键点条件多假设。连续风格、生理、EEG 增量验证仍然阻塞。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/figures/vehicle_only_decision_key_metrics_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/figures/vehicle_only_decision_rmse_vs_wrong_side_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_candidate_decision_table_v0_2.csv`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_vehicle_only_decision_table_v0_2/tables/vehicle_only_stage3_gate_status_v0_2.csv`
