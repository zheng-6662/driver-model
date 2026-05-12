# 阶段 6：车辆-only结构化路线审计 v0.1

生成时间：2026-05-13 05:50

## 输入

- structured Transformer 指标表
- RBF/keypoint selector 指标表
- top-K Transformer 指标表
- top-K reliability selector 指标表
- RBF/keypoint multihypothesis review 指标表

本轮只读已有指标表，不重新训练模型，不读取原始数据，不使用服务器。

## Gate 结论

| gate                                       | status                         | decision                              |
| ------------------------------------------ | ------------------------------ | ------------------------------------- |
| rbf_primary_reference                      | pass_limited                   | 保留为当前车辆-only主参照，但不是问题已解决。             |
| direct_transformer_upgrade                 | fail                           | 不升级为主线。                               |
| response_decomposition_transformer_upgrade | fail                           | 当前版本 no-go，只保留为失败样本和辅助头诊断。            |
| keypoint_residual_upgrade                  | weak_no_go                     | 有方向/大幅响应信号，但整体和若干物理指标不足，不能单独升级。       |
| rbf_keypoint_selector                      | weak_candidate_continue        | 作为下一版选择器/可靠性候选继续，但不能作为已完成强基线。         |
| multi_hypothesis_oracle_bound              | research_signal_not_deployable | 说明多候选空间有潜力；必须做可部署选择策略，不能用 oracle 当结论。 |
| stage05_physio_eeg_allowed                 | blocked                        | 继续阻塞生理/EEG有效性结论。                      |
| stage06_next_route                         | go_stage06b                    | 下一步优先做RBF+关键点/多假设的可部署选择器、可靠性门控和坏样本复盘。 |

## 关键候选 scorecard

| model_name                                       | route_status                   | rmse_steer | delta_vs_rbf__rmse_steer | wrong_side_rate | large_response_recall | difficult_top20_rmse |
| ------------------------------------------------ | ------------------------------ | ---------- | ------------------------ | --------------- | --------------------- | -------------------- |
| oracle_best_of_rbf_plus_topk_upper_bound         | research_signal_not_deployable | 0.415652   | -0.118014                | 0.075000        | 0.875000              | 0.604369             |
| oracle_best_of_rbf_keypoint_upper_bound          | research_signal_not_deployable | 0.475095   | -0.058571                | 0.200000        | 0.875000              | 0.648368             |
| topk_vehicle_transformer_best_of_3_oracle        | research_signal_not_deployable | 0.477534   | -0.056132                | 0.025000        | 0.875000              | 0.634191             |
| rbf_kernel_ridge_context_no_subject              | keep_limited_primary_reference | 0.533667   | 0.000000                 | 0.225000        | 0.750000              | 0.678907             |
| selector_logreg_rbf_keypoint_no_subject          | weak_candidate_continue        | 0.533912   | 0.000245                 | 0.200000        | 0.875000              | 0.648368             |
| ridge_rich_context_no_subject                    | reference_or_no_go             | 0.536450   | 0.002784                 | 0.175000        | 0.500000              | 0.757102             |
| ridge_rich_history_no_subject                    | reference_or_no_go             | 0.538776   | 0.005109                 | 0.175000        | 0.500000              | 0.770213             |
| topk_top1_rbf_fallback_logreg_no_subject         | reference_or_no_go             | 0.542071   | 0.008405                 | 0.225000        | 0.750000              | 0.678907             |
| keypoint_residual_vehicle_transformer_no_subject | weak_no_go_current_form        | 0.548994   | 0.015327                 | 0.125000        | 0.875000              | 0.728866             |
| topk_vehicle_transformer_branch0_no_subject      | no_go_current_form             | 0.555098   | 0.021431                 | 0.050000        | 0.750000              | 0.698174             |
| ridge_vehicle_history_no_subject                 | no_go_current_form             | 0.565210   | 0.031543                 | 0.200000        | 0.375000              | 0.773604             |
| vehicle_transformer_context_no_subject           | no_go_current_form             | 0.566011   | 0.032345                 | 0.225000        | 0.625000              | 0.770506             |
| topk_rbf_branch_logreg_selector_no_subject       | no_go_current_form             | 0.576630   | 0.042963                 | 0.150000        | 0.625000              | 0.640533             |
| topk_vehicle_transformer_top1_no_subject         | no_go_current_form             | 0.587883   | 0.054217                 | 0.100000        | 0.750000              | 0.717094             |
| topk_vehicle_transformer_branch1_no_subject      | no_go_current_form             | 0.589604   | 0.055937                 | 0.025000        | 0.750000              | 0.840132             |
| structured_vehicle_transformer_aux_no_subject    | no_go_current_form             | 0.602174   | 0.068507                 | 0.225000        | 0.500000              | 0.802289             |
| topk_branch_logreg_selector_no_subject           | no_go_current_form             | 0.610735   | 0.077068                 | 0.125000        | 0.625000              | 0.769374             |
| knn_template_context_no_subject                  | no_go_current_form             | 0.625829   | 0.092162                 | 0.175000        | 0.750000              | 0.710014             |
| formal_ridge_vehicle_context_no_subject          | no_go_current_form             | 0.652392   | 0.118726                 | 0.150000        | 0.125000              | 0.975715             |
| topk_vehicle_transformer_branch2_no_subject      | no_go_current_form             | 0.685254   | 0.151587                 | 0.150000        | 0.625000              | 0.722768             |

## 解释边界

- RBF 继续作为 limited primary reference，不代表车辆-only任务已解决。
- 响应分解 Transformer v0.1 是 no-go 当前形式，不代表结构化建模方向被否定。
- keypoint selector 是弱候选，需要样本级选择错误复盘和可靠性门控。
- oracle/best-of-K 是研究上限，不可作为可部署结果或论文主结论。
- 生理/EEG 仍 blocked。

## 图

- RMSE 汇总图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/figures/vehicle_structured_route_rmse_summary.png`
- 相对 RBF delta 图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/06_structured_models/stage06_vehicle_only_structured_route_audit_v0_1/figures/vehicle_structured_route_delta_vs_rbf.png`
