# v274 no-harm bio override

## 本轮目的

- v273 的 bio top5 candidate oracle 有上界，但强制选择候选会失败。
- v274 默认使用 fixed wait-latest，只在 pair model 对生理候选高置信时才覆盖。
- 阈值只在 val bad_top10 上选，test 只报告。

## test bad_top10 决策收口

| source                         | label                                                                          |     rmse | deployable   |   override_rate |   delta_vs_fixed_latest | passes_fixed_latest   |
|:-------------------------------|:-------------------------------------------------------------------------------|---------:|:-------------|----------------:|------------------------:|:----------------------|
| baseline                       | policy_keep_0ms_anchor                                                         | 1.19771  | True         |    nan          |             0.502658    | False                 |
| baseline                       | policy_wait_to_latest_anchor                                                   | 0.695048 | True         |    nan          |             4.15347e-07 | False                 |
| oracle                         | oracle_best_anchor_upper_bound                                                 | 0.612475 | False        |    nan          |            -0.0825726   | True                  |
| bio_prefilter_candidate_oracle | subject_summary64:pair_candidate_oracle_k5                                     | 0.646591 | False        |    nan          |            -0.0484573   | True                  |
| test_best_override_diagnostic  | override_best_active_subject_seq_pca72_pair_bio_hgb                            | 0.690235 | False        |      0.0869565  |            -0.00481295  | True                  |
| val_best_any                   | override_best_any_subject_seq_pca72_pair_vehicle_bio_badweighted_hgb           | 0.695048 | True         |      0.00543478 |             4.15347e-07 | False                 |
| val_best_active                | override_best_active_subject_seq_pca72_pair_vehicle_bio_badweighted_hgb        | 0.695048 | True         |      0.00543478 |             4.15347e-07 | False                 |
| val_best_noharm_active         | override_best_noharm_active_subject_seq_pca72_pair_vehicle_bio_badweighted_hgb | 0.695048 | True         |      0.00543478 |             4.15347e-07 | False                 |

## val 选择的 override 策略

| raw_set                   | pred_col                              |   score_threshold |   margin_threshold |   val_bad_top10_rmse |   val_bad_top10_latest_rmse |   val_bad_top10_delta_vs_latest |   val_bad_top10_override_rate |   val_bad_top10_override_n | chosen_type   |
|:--------------------------|:--------------------------------------|------------------:|-------------------:|---------------------:|----------------------------:|--------------------------------:|------------------------------:|---------------------------:|:--------------|
| subject_seq_pca72         | pred_pair_vehicle_bio_badweighted_hgb |         0.0722127 |         0.00941309 |              1.0573  |                     1.07279 |                    -0.015491    |                     0.0645161 |                          2 | best_active   |
| subject_summary64         | pred_pair_vehicle_bio_badweighted_hgb |         0.101442  |         0.00320514 |              1.06313 |                     1.07279 |                    -0.00965567  |                     0.129032  |                          4 | best_active   |
| subject_seq_pca72         | pred_pair_vehicle_bio_hgb             |         0.139406  |         0.0101681  |              1.06376 |                     1.07279 |                    -0.00902852  |                     0.193548  |                          6 | best_active   |
| calibrated_screened64     | pred_pair_vehicle_bio_hgb             |         0.0938691 |         0.00162112 |              1.06525 |                     1.07279 |                    -0.00753516  |                     0.0967742 |                          3 | best_active   |
| recording_seq_pca72       | pred_pair_vehicle_bio_hgb             |         0.0872378 |         0.015089   |              1.06641 |                     1.07279 |                    -0.00637631  |                     0.0322581 |                          1 | best_active   |
| calibrated_screened64     | pred_pair_base_hgb                    |         0.0775107 |         0.0225226  |              1.06641 |                     1.07279 |                    -0.00637631  |                     0.0322581 |                          1 | best_active   |
| calibrated_screened64     | pred_pair_vehicle_hgb                 |         0.0793083 |         0.0107393  |              1.06641 |                     1.07279 |                    -0.00637631  |                     0.0322581 |                          1 | best_active   |
| calibrated_low_identity48 | pred_pair_vehicle_bio_hgb             |         0.0824817 |         0.0127073  |              1.06641 |                     1.07279 |                    -0.00637631  |                     0.0322581 |                          1 | best_active   |
| recording_summary64       | pred_pair_vehicle_hgb                 |         0.0789112 |         0.0258902  |              1.0665  |                     1.07279 |                    -0.00628663  |                     0.0967742 |                          3 | best_active   |
| subject_seq_pca72         | pred_pair_vehicle_hgb                 |         0.0876095 |         0.0281111  |              1.06757 |                     1.07279 |                    -0.00521616  |                     0.0645161 |                          2 | best_active   |
| calibrated_low_identity48 | pred_pair_vehicle_hgb                 |         0.0808001 |         0.00822623 |              1.06785 |                     1.07279 |                    -0.00494251  |                     0.0645161 |                          2 | best_active   |
| subject_summary64         | pred_pair_base_hgb                    |         0.0887712 |         0.0271502  |              1.07199 |                     1.07279 |                    -0.000800787 |                     0.0645161 |                          2 | best_active   |
| recording_summary64       | pred_pair_bio_hgb                     |         0.0960284 |         0.00426552 |              1.07207 |                     1.07279 |                    -0.00072247  |                     0.0967742 |                          3 | best_active   |
| subject_summary64         | pred_pair_vehicle_bio_hgb             |         0.105744  |         0.00616532 |              1.07244 |                     1.07279 |                    -0.000348873 |                     0.193548  |                          6 | best_active   |
| calibrated_low_identity48 | pred_pair_bio_hgb                     |         0.104731  |         0.0102628  |              1.07279 |                     1.07279 |                     0           |                     0.0322581 |                          1 | best_active   |
| calibrated_low_identity48 | pred_pair_vehicle_bio_badweighted_hgb |         0.0924522 |         0.0161657  |              1.0732  |                     1.07279 |                     0.000407984 |                     0.0322581 |                          1 | best_active   |
| subject_summary64         | pred_pair_vehicle_hgb                 |         0.0925807 |         0.0206458  |              1.07422 |                     1.07279 |                     0.0014338   |                     0.0322581 |                          1 | best_active   |
| calibrated_screened64     | pred_pair_vehicle_bio_badweighted_hgb |         0.08676   |         0.00357583 |              1.07947 |                     1.07279 |                     0.00667901  |                     0.129032  |                          4 | best_active   |
| subject_summary64         | pred_pair_bio_hgb                     |         0.0913249 |         0.012752   |              1.08061 |                     1.07279 |                     0.00781783  |                     0.0322581 |                          1 | best_active   |
| recording_seq_pca72       | pred_pair_base_hgb                    |         0.0747678 |         0.0229605  |              1.08137 |                     1.07279 |                     0.0085853   |                     0.0322581 |                          1 | best_active   |
| recording_seq_pca72       | pred_pair_vehicle_bio_badweighted_hgb |         0.0850688 |         0.0123329  |              1.08137 |                     1.07279 |                     0.0085853   |                     0.0322581 |                          1 | best_active   |
| recording_seq_pca72       | pred_pair_vehicle_hgb                 |         0.0841097 |         0.0125668  |              1.08151 |                     1.07279 |                     0.00872001  |                     0.0322581 |                          1 | best_active   |
| calibrated_screened64     | pred_pair_bio_hgb                     |         0.112589  |         0.0118922  |              1.08512 |                     1.07279 |                     0.0123371   |                     0.0322581 |                          1 | best_active   |
| recording_summary64       | pred_pair_vehicle_bio_badweighted_hgb |         0.118166  |         0.0254723  |              1.08832 |                     1.07279 |                     0.0155291   |                     0.0967742 |                          3 | best_active   |
| subject_seq_pca72         | pred_pair_bio_hgb                     |         0.0978197 |         0.00812831 |              1.09826 |                     1.07279 |                     0.0254698   |                     0.0645161 |                          2 | best_active   |
| recording_summary64       | pred_pair_vehicle_bio_hgb             |         0.0968693 |         0.00227483 |              1.09852 |                     1.07279 |                     0.0257327   |                     0.129032  |                          4 | best_active   |
| calibrated_low_identity48 | pred_pair_base_hgb                    |         0.0832326 |         0.0247231  |              1.10676 |                     1.07279 |                     0.0339729   |                     0.0645161 |                          2 | best_active   |
| subject_seq_pca72         | pred_pair_base_hgb                    |         0.0774624 |         0.0282832  |              1.12639 |                     1.07279 |                     0.0535997   |                     0.0645161 |                          2 | best_active   |
| recording_seq_pca72       | pred_pair_bio_hgb                     |         0.0939719 |         0.0121232  |              1.14048 |                     1.07279 |                     0.0676949   |                     0.0322581 |                          1 | best_active   |
| recording_summary64       | pred_pair_base_hgb                    |         0.0847473 |         0.0311172  |              1.24937 |                     1.07279 |                     0.176582    |                     0.0645161 |                          2 | best_active   |

## test bad_top10 override top

| strategy                                                               |   selected_tail_rmse_mean |   delta_selected_minus_latest_mean |   override_rate |   selected_delay_ms_mean |   selected_latest_rate |
|:-----------------------------------------------------------------------|--------------------------:|-----------------------------------:|----------------:|-------------------------:|-----------------------:|
| override_best_active_subject_seq_pca72_pair_bio_hgb                    |                  0.690235 |                        -0.00481337 |      0.0869565  |                  960.526 |               0.947368 |
| override_best_any_subject_summary64_pair_base_hgb                      |                  0.695048 |                         0          |      0.0869565  |                 1000     |               1        |
| override_best_noharm_active_subject_summary64_pair_base_hgb            |                  0.695048 |                         0          |      0.0869565  |                 1000     |               1        |
| override_best_active_subject_summary64_pair_base_hgb                   |                  0.695048 |                         0          |      0.0869565  |                 1000     |               1        |
| override_best_active_subject_summary64_pair_vehicle_hgb                |                  0.695048 |                         0          |      0.0271739  |                 1000     |               1        |
| override_best_any_subject_summary64_pair_bio_hgb                       |                  0.695048 |                         0          |      0          |                 1000     |               1        |
| override_best_active_subject_summary64_pair_bio_hgb                    |                  0.695048 |                         0          |      0.0217391  |                 1000     |               1        |
| override_best_any_subject_summary64_pair_vehicle_hgb                   |                  0.695048 |                         0          |      0          |                 1000     |               1        |
| override_best_any_recording_summary64_pair_base_hgb                    |                  0.695048 |                         0          |      0          |                 1000     |               1        |
| override_best_any_recording_summary64_pair_vehicle_hgb                 |                  0.695048 |                         0          |      0.00543478 |                 1000     |               1        |
| override_best_any_subject_seq_pca72_pair_vehicle_hgb                   |                  0.695048 |                         0          |      0          |                 1000     |               1        |
| override_best_any_recording_summary64_pair_vehicle_bio_badweighted_hgb |                  0.695048 |                         0          |      0          |                 1000     |               1        |
| override_best_noharm_active_recording_summary64_pair_vehicle_hgb       |                  0.695048 |                         0          |      0.00543478 |                 1000     |               1        |
| override_best_active_recording_summary64_pair_vehicle_bio_hgb          |                  0.695048 |                         0          |      0.163043   |                 1000     |               1        |
| override_best_active_recording_summary64_pair_vehicle_hgb              |                  0.695048 |                         0          |      0.00543478 |                 1000     |               1        |
| override_best_active_recording_summary64_pair_base_hgb                 |                  0.695048 |                         0          |      0.0543478  |                 1000     |               1        |
| override_best_noharm_active_subject_seq_pca72_pair_vehicle_hgb         |                  0.695048 |                         0          |      0          |                 1000     |               1        |
| override_best_active_subject_seq_pca72_pair_vehicle_hgb                |                  0.695048 |                         0          |      0          |                 1000     |               1        |
| override_best_any_subject_seq_pca72_pair_bio_hgb                       |                  0.695048 |                         0          |      0          |                 1000     |               1        |
| override_best_any_subject_seq_pca72_pair_base_hgb                      |                  0.695048 |                         0          |      0          |                 1000     |               1        |

## 判读

- val 选择的 no-harm override 仍未低于 fixed wait-latest，稀疏覆盖也没有兑现生理上界。
- 若本轮失败，现有生理在可部署层面的主增量基本被证伪，应回到车辆多未来/不确定性主线。

## 关键图

- `figures\v274_test_badtop10_noharm_override.png`