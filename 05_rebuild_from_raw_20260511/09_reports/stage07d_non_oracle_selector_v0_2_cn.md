# Stage 7d 技术报告：non-oracle selector v0.2

## Scope

- Input table: `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_feature_and_label_diagnosis.csv`
- Per-sample metrics: `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_export_per_sample_metrics.csv`
- Candidate pool: `rbf_kernel_ridge_context_no_subject, keypoint_residual_vehicle_transformer_no_subject, topk_vehicle_transformer_branch0_no_subject, topk_vehicle_transformer_branch1_no_subject, topk_vehicle_transformer_branch2_no_subject`
- Target for train/val diagnostics: broad oracle winner from Stage 7c.
- No server used. Credential file not read.
- Excluded inputs: label diagnostics, subject/session identifiers, physio, EEG, continuous style.

## Selected Policy

- selected_policy=`always_rbf_reference`
- gate=`no_upgrade`
- reason=val gate 没有发现比 RBF/KNN 更可靠的非 oracle 策略。

## Validation Candidate Table

```text
                                     model_name  rmse_steer  rmse_delta_vs_rbf  wrong_side_rate  large_response_recall  rbf_selected_rate  oracle_match_rate  selected_by_val_gate
  rf_depth3_balanced__fallback_rbf_conf_lt_0.35    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth3_balanced__fallback_rbf_conf_lt_0.45    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth3_balanced__fallback_rbf_conf_lt_0.55    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth3_balanced__fallback_rbf_conf_lt_0.65    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth3_balanced__fallback_rbf_conf_lt_0.75    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth4_balanced__fallback_rbf_conf_lt_0.35    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth4_balanced__fallback_rbf_conf_lt_0.45    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth4_balanced__fallback_rbf_conf_lt_0.55    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth4_balanced__fallback_rbf_conf_lt_0.65    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth4_balanced__fallback_rbf_conf_lt_0.75    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
logreg_balanced_c0_2__fallback_rbf_conf_lt_0.35    0.575926           0.004444         0.095238                    0.5           0.547619           0.261905                     0
logreg_balanced_c0_2__fallback_rbf_conf_lt_0.45    0.579446           0.007964         0.119048                    0.5           0.833333           0.309524                     0
```

## Test Table

```text
              model_name  rmse_steer  rmse_delta_vs_rbf  wrong_side_rate  large_response_recall  difficult_top20_rmse
    always_rbf_reference    0.533667           0.000000            0.225                  0.750              0.678907
    topk_top1_non_oracle    0.587865           0.054198            0.100                  0.750              0.717094
broad_oracle_upper_bound    0.410957          -0.122710            0.075                  0.875              0.604369
```

## Figures

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_policy_metrics_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_validation_rmse_delta.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_selected_choice_counts.png`

## Interpretation

The model selection gate is deliberately conservative. A policy must pass validation without test information before its test score is interpreted. If selected policy is RBF or test delta is non-negative, Stage 7 remains an oracle-gap problem rather than a deployable multi-hypothesis solution.
