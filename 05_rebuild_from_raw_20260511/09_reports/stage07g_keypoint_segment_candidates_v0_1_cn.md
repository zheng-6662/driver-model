# Stage 7g 技术报告：keypoint/segment vehicle-only candidates v0.1

## Scope

- Track: `B_response3s_strict_core`
- Input trajectories: `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07c_candidate_trajectory_export_v0_1/arrays/stage07c_candidate_trajectories.npz`
- Response labels: `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_response_label_table.csv`
- No server used. Credential file not read.
- Excluded: subject ID, session ID, physio, EEG, continuous style, test labels as inputs.

## Selected Policy

- selected_policy=`segment_abs_rf_blend_25`
- gate=`no_upgrade`
- test_delta_vs_rbf=+0.002509

## Test Summary

- RBF/KNN RMSE=0.533667
- selected RMSE=0.536176
- keypoint/segment oracle RMSE=0.462003
- best non-oracle by test only: `rbf_resid_keypoint_scaled`, RMSE=0.508538, delta=-0.025129; this is diagnostic only because it was not selected by validation.

## Target Metrics

```text
model_prefix       target split     rmse      mae      bias      corr
         abs  peak_signed   val 0.757952 0.547474 -0.203269  0.845982
         abs  peak_signed  test 0.858810 0.660271 -0.349729  0.796070
         abs  peak_time_s   val 0.544119 0.417799  0.079468  0.065754
         abs  peak_time_s  test 0.421546 0.322936  0.074206  0.195029
         abs onset_time_s   val 0.570218 0.371245  0.029750 -0.058195
         abs onset_time_s  test 0.361616 0.314167  0.209032  0.238896
         abs  tail_signed   val 0.261202 0.200247  0.000715  0.092358
         abs  tail_signed  test 0.242938 0.196777  0.007812 -0.204303
       resid  peak_signed   val 0.715985 0.486161 -0.102149  0.855030
       resid  peak_signed  test 0.789667 0.543640 -0.270118  0.820470
       resid  peak_time_s   val 0.655622 0.517787  0.107814  0.124052
       resid  peak_time_s  test 0.557853 0.426311  0.040161  0.072001
       resid onset_time_s   val 0.604058 0.385686  0.130072  0.366844
       resid onset_time_s  test 0.366992 0.275653  0.168456  0.313355
       resid  tail_signed   val 0.271793 0.210021  0.006594  0.066353
       resid  tail_signed  test 0.237278 0.189408 -0.015358  0.040017
```

## Gate

```text
                 gate_item                  status                                                        evidence
           selected_policy segment_abs_rf_blend_25           validation gate reason: val_rmse_improvement_gt_0_002
        deployable_upgrade              no_upgrade                 test delta vs RBF +0.002509; physical_gain=True
   keypoint_segment_oracle         diagnostic_only                   oracle uses true labels and is not deployable
stage08_physio_eeg_allowed                 blocked vehicle-only keypoint/segment candidate route is not yet stable
               server_used                      no                        local run only; credential file not read
```
