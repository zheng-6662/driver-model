# Stage 7j 技术报告：session-CV stability audit v0.1

## Scope

- 5-fold grouped by `session_stamp`.
- Per-fold RBF retraining.
- Feature protocol: event/road context + fold-retrained RBF shape features only.
- Fixed-split topK/Transformer/keypoint prediction features excluded due leakage risk under new folds.
- No physio, EEG, continuous style or subject ID.
- No server used. Credential file not read.

## Aggregate

```text
              policy_name  n_folds  mean_test_rmse  mean_test_delta_vs_rbf  median_test_delta_vs_rbf  std_test_delta_vs_rbf  improved_fold_count  improved_fold_rate  mean_wrong_side_delta  mean_large_recall_delta  mean_difficult_delta  difficult_improved_fold_rate                                                                                                 selected_models
stage7g_original_val_gate        5        0.624985               -0.004170                 -0.009860               0.009537                    3                 0.6               0.014815                 0.154762             -0.019418                           0.8                          rbf_resid_keypoint_scaled, rbf_resid_keypoint_scaled_blend_50, segment_abs_rf_blend_25
     always_rbf_reference        5        0.629156                0.000000                  0.000000               0.000000                    0                 0.0               0.000000                 0.000000              0.000000                           0.0                                                                             rbf_kernel_ridge_context_no_subject
    stability_penalty_l05        5        0.629485                0.000329                 -0.011193               0.016275                    3                 0.6               0.011111                 0.126190             -0.014632                           0.8 rbf_abs_keypoint_scaled_blend_50, rbf_resid_keypoint_scaled, segment_abs_rf_blend_25, segment_resid_rf_blend_25
```

## Gate

```text
                 gate_item                            status                                                                                                                        evidence
       cv_feature_protocol strict_retrained_rbf_context_only          RBF was retrained per fold; fixed-split topK/Transformer candidate-prediction features were excluded to avoid leakage.
stability_policy_cv_result                        no_upgrade                                         mean test delta=+0.000329; improved fold rate=0.600; difficult improved fold rate=0.800
          mainline_upgrade                         not_final Even a positive CV result would still need full upstream candidate retraining and fixed-plot review before freezing a mainline.
stage08_physio_eeg_allowed                           blocked                           Vehicle-only candidate stability is still under validation; no physio/EEG evidence is evaluated here.
               server_used                                no                                                                        Local CPU diagnostic run only; credential file not read.
```
