# Stage 7e 技术报告：candidate generation redesign audit v0.1

## Scope

- Input trajectories: `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/arrays/stage07c_candidate_trajectories.npz`
- Feature diagnosis: `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/tables/candidate_feature_and_label_diagnosis.csv`
- Stage 7d gate: `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_gate_table.csv`
- Candidate pool audited: `rbf_kernel_ridge_context_no_subject, keypoint_residual_vehicle_transformer_no_subject, topk_vehicle_transformer_branch0_no_subject, topk_vehicle_transformer_branch1_no_subject, topk_vehicle_transformer_branch2_no_subject`
- No new model training.
- No server used. Credential file not read.
- Excluded modalities: physio, EEG, continuous style, subject ID.

## Aggregate Test Result

- RBF/KNN RMSE: 0.533667
- Deployable-candidate oracle RMSE: 0.410957
- Mean sample gain over RBF/KNN: 0.108576
- Non-RBF oracle winner rate: 0.700000

## Gate

```text
                           gate_item      status                                                                                               evidence
         stage07d_deployable_upgrade  no_upgrade                               Stage 7d val gate selected always_rbf_reference; test delta vs RBF is 0.
        continue_selector_only_route     blocked       Two selector rounds fell back to RBF/KNN; next improvement should redesign candidate generation.
candidate_generation_redesign_needed        pass                                               test bucket statuses: selector_gap=16, generation_gap=0.
               next_training_allowed conditional Allowed only after implementing response-factorized candidates and keeping RBF/KNN as fixed reference.
          stage08_physio_eeg_allowed     blocked   Do not enter physio/EEG until vehicle-only candidate generation and non-oracle selection are stable.
                         server_used          no                                                        Local audit only; credential file was not read.
```

## Interpretation

Stage 7d blocked selector-only continuation. Stage 7e therefore upgrades the next action from selector tuning to response-factorized candidate generation. This is still vehicle-only; it does not authorize physio/EEG claims.

## Tables

- `stage07e_response_label_table.csv`
- `stage07e_sample_candidate_gap_table.csv`
- `stage07e_existing_candidate_coverage_by_bucket.csv`
- `stage07e_oracle_winner_distribution.csv`
- `stage07e_candidate_generation_blueprint.csv`
- `stage07e_next_experiment_plan.csv`
- `stage07e_gate_table.csv`

## Figures

- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_oracle_gain_by_response_family_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_oracle_winner_distribution_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_candidate_gap_scatter_test.png`
- `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_candidate_generation_blueprint.png`
