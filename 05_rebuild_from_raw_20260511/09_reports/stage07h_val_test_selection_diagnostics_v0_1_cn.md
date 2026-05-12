# Stage 7h 技术报告：val/test selection diagnostics v0.1

## Scope

- Source stage: Stage 7g keypoint/segment candidates.
- No model training in this stage.
- No server used. Credential file not read.
- Excluded from modeling claims: physio, EEG, continuous style, subject ID.

## Core Finding

- selected_by_val=`segment_abs_rf_blend_25`
- selected_test_delta_vs_rbf=+0.002509
- test_best_non_oracle=`rbf_resid_keypoint_scaled`, test_delta_vs_rbf=-0.025129
- gate remains no_upgrade because test-best is diagnostic only.

## Candidate Stability

```text
                         model_name  rmse_delta_vs_rbf_val  rmse_delta_vs_rbf_test  val_rmse_rank  test_rmse_rank  val_test_delta_swing         diagnostic_status
          rbf_resid_keypoint_scaled               0.017149               -0.025129            8.0             1.0             -0.042278 test_best_diagnostic_only
 rbf_resid_keypoint_scaled_blend_50               0.004254               -0.018202            6.0             2.0             -0.022456           other_candidate
          segment_resid_rf_blend_25              -0.006778               -0.005620            3.0             4.0              0.001157           other_candidate
rbf_kernel_ridge_context_no_subject               0.000000                0.000000            5.0             5.0              0.000000           other_candidate
            segment_abs_rf_blend_25              -0.010339                0.002509            1.0             7.0              0.012848      selected_by_val_gate
```

## Gate

```text
                 gate_item          status                                                                                                          evidence
       diagnosis_completed            pass                                              val/test candidate stability and distribution shift tables generated
        deployable_upgrade      no_upgrade val-selected segment_abs_rf_blend_25 test delta +0.002509; test-best rbf_resid_keypoint_scaled is diagnostic only
       test_best_candidate diagnostic_only                                 rbf_resid_keypoint_scaled test delta vs RBF -0.025129; not selected by validation
stage08_physio_eeg_allowed         blocked                                                              vehicle-only candidate selection is still not stable
               server_used              no                                                               local diagnostic run only; credential file not read
```
