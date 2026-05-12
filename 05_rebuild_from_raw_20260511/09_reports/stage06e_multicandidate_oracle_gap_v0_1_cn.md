# Stage 6e：多候选 oracle gap 复核 v0.1

生成时间：2026-05-13 06:24

说明：本轮不训练新模型，不使用生理、EEG、连续风格或被试 ID。oracle 结果只作为研究上限，不可部署。

## Gate

| gate                       | status                         | evidence                                                                          | decision                                         |
| -------------------------- | ------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------ |
| oracle_pool_signal         | research_signal_not_deployable | broad oracle test RMSE=0.375182, delta=-0.158484; this uses labels for selection. | 只说明候选池存在上限，不可作为部署模型或生理有效性证据。                     |
| deployable_selection_gap   | blocked                        | best deployable selector test RMSE=0.533912, delta=+0.000245.                     | Stage 7 若继续，必须优先解决非 oracle 选择策略；不能只报告 best-of-K。 |
| stage05_physio_eeg_allowed | blocked                        | 车辆-only 多候选选择仍未闭环。                                                                | 继续阻塞生理/EEG增量结论。                                  |

## Test gap table

| model_name                                       | role                              | rmse_steer | delta_vs_rbf_rmse | wrong_side_rate | large_response_recall | difficult_top20_rmse |
| ------------------------------------------------ | --------------------------------- | ---------- | ----------------- | --------------- | --------------------- | -------------------- |
| oracle_broad_vehicle_pool                        | oracle_upper_bound_not_deployable | 0.375182   | -0.158484         | 0.050000        | 1.000000              | 0.581930             |
| oracle_rbf_plus_topk3                            | oracle_upper_bound_not_deployable | 0.415656   | -0.118011         | 0.075000        | 0.875000              | 0.604378             |
| oracle_rbf_plus_keypoint                         | oracle_upper_bound_not_deployable | 0.475095   | -0.058571         | 0.200000        | 0.875000              | 0.648368             |
| oracle_topk3_only                                | oracle_upper_bound_not_deployable | 0.477534   | -0.056132         | 0.025000        | 0.875000              | 0.634191             |
| rbf_kernel_ridge_context_no_subject              | current_rbf_knn_reference         | 0.533667   | 0.000000          | 0.225000        | 0.750000              | 0.678907             |
| selector_logreg_rbf_keypoint_no_subject          | deployable_selector_attempt       | 0.533912   | 0.000245          | 0.200000        | 0.875000              | 0.648365             |
| stage06d_val_rmse_noninferior_conservative       | deployable_selector_attempt       | 0.534545   | 0.000878          | 0.225000        | 0.750000              | 0.678907             |
| ridge_rich_context_no_subject                    | single_or_branch_candidate        | 0.536450   | 0.002784          | 0.175000        | 0.500000              | 0.757102             |
| topk_top1_rbf_fallback_logreg_no_subject         | deployable_selector_attempt       | 0.542071   | 0.008405          | 0.225000        | 0.750000              | 0.678907             |
| stage06d_val_best_rmse                           | deployable_selector_attempt       | 0.544356   | 0.010689          | 0.175000        | 0.875000              | 0.659949             |
| keypoint_residual_vehicle_transformer_no_subject | single_or_branch_candidate        | 0.548994   | 0.015327          | 0.125000        | 0.875000              | 0.728866             |
| topk_rbf_branch_logreg_selector_no_subject       | deployable_selector_attempt       | 0.576630   | 0.042963          | 0.150000        | 0.625000              | 0.640533             |

## Broad oracle winner summary

| oracle_winner_model                              | n_wins   | mean_oracle_gain_vs_rbf |
| ------------------------------------------------ | -------- | ----------------------- |
| knn_template_context_no_subject                  | 6.000000 | 0.099305                |
| rbf_kernel_ridge_context_no_subject              | 5.000000 | 0.000000                |
| topk_vehicle_transformer_branch0_no_subject      | 5.000000 | 0.132120                |
| topk_vehicle_transformer_branch2_no_subject      | 5.000000 | 0.099571                |
| direction_gated_knn_template_no_subject          | 4.000000 | 0.239664                |
| topk_vehicle_transformer_branch1_no_subject      | 4.000000 | 0.238950                |
| peak_scaled_template_context_no_subject          | 3.000000 | 0.180147                |
| ridge_rich_context_no_subject                    | 3.000000 | 0.100148                |
| ridge_rich_history_no_subject                    | 3.000000 | 0.330291                |
| keypoint_residual_vehicle_transformer_no_subject | 2.000000 | 0.171685                |
