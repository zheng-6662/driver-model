# Stage 7c 技术报告：candidate trajectory export v0.1

## Scope

- Track: `B_response3s_strict_core`.
- Dataset split: `session_level_split`.
- Source checkpoints:
  - `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_topk_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_topk_vehicle_transformer_top1_no_subject_best.pt`
  - `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_vehicle_instability_keypoint_residual_vehicle_transformer_v0_1/checkpoints/B_response3s_strict_core_keypoint_residual_vehicle_transformer_no_subject_best.pt`
- No training was run.
- No server was used.
- Credential file was not read.
- Modalities used: vehicle history and causal road/event context only.
- Excluded: subject ID, continuous style, physio, EEG, test labels as inputs.

## Test Metrics

```text
                                      model_name  rmse_steer  wrong_side_rate  large_response_recall  difficult_top20_rmse
keypoint_residual_vehicle_transformer_no_subject    0.548993            0.125                  0.875              0.728858
    oracle_best_of_rbf_keypoint_topk_upper_bound    0.410957            0.075                  0.875              0.604369
             oracle_best_of_rbf_topk_upper_bound    0.415652            0.075                  0.875              0.604369
             rbf_kernel_ridge_context_no_subject    0.533667            0.225                  0.750              0.678907
     topk_vehicle_transformer_branch0_no_subject    0.555089            0.050                  0.750              0.698197
     topk_vehicle_transformer_branch1_no_subject    0.589621            0.025                  0.750              0.840177
     topk_vehicle_transformer_branch2_no_subject    0.685207            0.150                  0.625              0.722737
        topk_vehicle_transformer_top1_no_subject    0.587865            0.100                  0.750              0.717094
```

## Oracle Interpretation

RBF+topK oracle RMSE=0.415652, delta vs RBF=-0.118014. Broad oracle RMSE=0.410957, delta vs RBF=-0.122710. These rows are upper-bound diagnostics only.

## Gate

- `candidate_trajectories_exported=pass`
- `deployable_upgrade=no`
- `reason`: no non-oracle policy in this stage; previous Stage 7b selected RBF for all test samples.
- `stage08_physio_eeg_allowed=blocked`

## Tables

- `candidate_export_metrics.csv`
- `candidate_export_per_sample_metrics.csv`
- `candidate_pairwise_disagreement_long.csv`
- `candidate_pairwise_disagreement_summary.csv`
- `candidate_feature_and_label_diagnosis.csv`
- `candidate_oracle_summary.csv`
- `candidate_export_gate_table.csv`

## Figures

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_metric_summary_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_disagreement_vs_oracle_gain_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_fixed_predictions_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_high_disagreement_predictions_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/figures/candidate_oracle_gain_predictions_test.png`

## Arrays

The replayable trajectory export is stored at `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/arrays/stage07c_candidate_trajectories.npz`.
