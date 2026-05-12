# Stage 6c：selector feature revision v0.1

生成时间：2026-05-13 06:05

## Gate

| gate                        | status                      | decision                          |
| --------------------------- | --------------------------- | --------------------------------- |
| selector_revision_test_gain | no_upgrade_current_revision | 只有同时稳定改善RMSE和物理指标才可升级；否则只作为下一版诊断。 |
| test_tuning_leakage_guard   | pass                        | 本轮选择规则无test调参。                    |
| feature_leakage_guard       | pass_protocol               | 当前特征协议可继续扩展，但不能加入未来真实标签。          |
| stage05_physio_eeg_allowed  | blocked                     | 继续阻塞生理/EEG有效性结论。                  |

## Test 指标

| model_name                                       | rmse_steer | wrong_side_rate | large_response_recall | difficult_top20_rmse |
| ------------------------------------------------ | ---------- | --------------- | --------------------- | -------------------- |
| oracle_best_of_rbf_keypoint_upper_bound          | 0.475095   | 0.200000        | 0.875000              | 0.648368             |
| rbf_kernel_ridge_context_no_subject              | 0.533667   | 0.225000        | 0.750000              | 0.678907             |
| logreg_original_balanced                         | 0.533912   | 0.200000        | 0.875000              | 0.648368             |
| logreg_engineered_balanced                       | 0.534545   | 0.225000        | 0.750000              | 0.678907             |
| logreg_engineered_conservative                   | 0.534545   | 0.225000        | 0.750000              | 0.678907             |
| rf_engineered_shallow                            | 0.544356   | 0.175000        | 0.875000              | 0.659949             |
| keypoint_residual_vehicle_transformer_no_subject | 0.548994   | 0.125000        | 0.875000              | 0.728866             |

## 选择错误摘要

| selection_outcome          | n_samples | mean_selector_prob_keypoint | mean_keypoint_delta_vs_rbf |
| -------------------------- | --------- | --------------------------- | -------------------------- |
| FN_missed_keypoint_gain    | 6.000000  | 0.232065                    | -0.174375                  |
| FP_select_keypoint_hurts   | 13.000000 | 0.436101                    | 0.115949                   |
| TN_keep_rbf_correct        | 10.000000 | 0.258839                    | 0.159274                   |
| TP_select_keypoint_correct | 11.000000 | 0.422183                    | -0.094621                  |

## 边界

- 不使用 test 调参。
- 不使用生理、脑电、连续风格或驾驶员 ID。
- 不使用未来真实标签作为 selector 输入。
