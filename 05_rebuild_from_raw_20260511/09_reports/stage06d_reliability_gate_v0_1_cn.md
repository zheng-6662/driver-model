# Stage 6d：RBF/KNN reliability gate v0.1

生成时间：2026-05-13 06:14

说明：当前主参照是 RBF/KNN 类车辆-only 强基线，不是 Transformer。本文的 gate 只是在 RBF/KNN 与 keypoint 候选之间做保守切换，不使用生理、EEG、连续风格或被试 ID。

## Gate

| gate                                       | status     | decision                        |
| ------------------------------------------ | ---------- | ------------------------------- |
| val_best_rmse                              | no_upgrade | 若RMSE仍退化则不能升级；若仅物理指标改善则保留为诊断候选。 |
| val_rmse_noninferior_conservative          | no_upgrade | 若RMSE仍退化则不能升级；若仅物理指标改善则保留为诊断候选。 |
| val_rmse_physical_noninferior_conservative | no_upgrade | 若RMSE仍退化则不能升级；若仅物理指标改善则保留为诊断候选。 |
| stage05_physio_eeg_allowed                 | blocked    | 继续阻塞生理/EEG有效性结论。                |

## Test metrics

| policy_label                                     | model_name                                       | rmse_steer | wrong_side_rate | large_response_recall | difficult_top20_rmse |
| ------------------------------------------------ | ------------------------------------------------ | ---------- | --------------- | --------------------- | -------------------- |
| oracle_best_of_rbf_keypoint_upper_bound          | oracle_best_of_rbf_keypoint_upper_bound          | 0.475095   | 0.200000        | 0.875000              | 0.648368             |
| rbf_kernel_ridge_context_no_subject              | rbf_kernel_ridge_context_no_subject              | 0.533667   | 0.225000        | 0.750000              | 0.678907             |
| val_rmse_noninferior_conservative                | logreg_engineered_balanced__thr_0.95             | 0.534545   | 0.225000        | 0.750000              | 0.678907             |
| val_rmse_physical_noninferior_conservative       | logreg_engineered_balanced__thr_0.95             | 0.534545   | 0.225000        | 0.750000              | 0.678907             |
| val_best_rmse                                    | rf_engineered_shallow__thr_0.35                  | 0.544356   | 0.175000        | 0.875000              | 0.659949             |
| keypoint_residual_vehicle_transformer_no_subject | keypoint_residual_vehicle_transformer_no_subject | 0.548994   | 0.125000        | 0.875000              | 0.728866             |

## Best policy confusion

| selection_outcome        | n_samples | mean_selector_prob_keypoint | mean_keypoint_delta_vs_rbf |
| ------------------------ | --------- | --------------------------- | -------------------------- |
| FN_missed_keypoint_gain  | 17.000000 | 0.312661                    | -0.122770                  |
| FP_select_keypoint_hurts | 1.000000  | 0.981281                    | 0.068107                   |
| TN_keep_rbf_correct      | 22.000000 | 0.355516                    | 0.137817                   |
