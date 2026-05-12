# Stage 7f 技术报告：response-factorized vehicle-only prototype candidates v0.1

## Scope

- Track: `B_response3s_strict_core`
- Input trajectories: `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/arrays/stage07c_candidate_trajectories.npz`
- Response labels: `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_response_label_table.csv`
- No server used. Credential file not read.
- Excluded: subject ID, session ID, physio, EEG, continuous style, test labels as inputs.

## Selected Policy

- selected_policy=`rbf_kernel_ridge_context_no_subject`
- gate=`no_upgrade`
- test_delta_vs_rbf=+0.000000

## Test Summary

- RBF/KNN RMSE=0.533667
- selected RMSE=0.533667
- response-factorized oracle RMSE=0.440217
- response-factorized + existing candidates oracle RMSE=0.388119

## Factor Prediction Metrics

```text
         factor split  accuracy  balanced_accuracy  mean_confidence
 direction_mode  test  0.925000           0.919437         0.849657
 direction_mode   val  0.904762           0.916667         0.879187
 amplitude_mode  test  0.475000           0.495951         0.466405
 amplitude_mode   val  0.500000           0.449020         0.491323
    peak_timing  test  0.650000           0.661616         0.637007
    peak_timing   val  0.595238           0.576389         0.615503
      tail_mode  test  0.825000           0.380392         0.814633
      tail_mode   val  0.785714           0.305556         0.786491
correction_mode  test  0.950000           0.657895         0.725166
correction_mode   val  0.880952           0.316239         0.695280
```

## Gate

```text
                 gate_item                              status                                                           evidence
           selected_policy rbf_kernel_ridge_context_no_subject                                   selected by validation gate only
        deployable_upgrade                          no_upgrade                                        test delta vs RBF +0.000000
response_factorized_oracle                     diagnostic_only                      oracle uses true labels and is not deployable
stage08_physio_eeg_allowed                             blocked vehicle-only response-factorized candidate route is not yet stable
               server_used                                  no                           local run only; credential file not read
```
