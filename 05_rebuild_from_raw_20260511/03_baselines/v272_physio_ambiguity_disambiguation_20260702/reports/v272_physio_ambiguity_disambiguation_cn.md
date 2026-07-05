# v272 physiology ambiguity disambiguation

## 本轮目的

- v266/v267/v270/v271 都显示候选库有 headroom，但可部署选择器选不准。
- v272 不再训练新模型，而是检查：在车辆 topK 相似候选内部，生理距离能不能把真正更好的候选排到前面。
- 如果生理 top1/top3 排序不能接近 vehicle candidate oracle，说明生理不是稳定消歧信号。

## 使用的 v271 特征集

| raw_set                   |   feature_n |   behavior_eta_max_mean |   identity_eta_max_mean |   identity_to_behavior_ratio_median |
|:--------------------------|------------:|------------------------:|------------------------:|------------------------------------:|
| subject_summary64         |          64 |              0.00564221 |               0.18942   |                            37.6941  |
| recording_summary64       |          64 |              0.00546261 |               0.062182  |                            13.3725  |
| subject_seq_pca72         |          72 |              0.00365873 |               0.129357  |                            27.7506  |
| recording_seq_pca72       |          72 |              0.00375531 |               0.133097  |                            21.8042  |
| calibrated_screened64     |          64 |              0.00693037 |               0.0444822 |                             6.58732 |
| calibrated_low_identity48 |          48 |              0.00316252 |               0.0248705 |                             9.23949 |

## test bad_top10 决策收口

| source                        | label                                                |     rmse | deployable   |   delta_vs_fixed_latest | passes_fixed_latest   |
|:------------------------------|:-----------------------------------------------------|---------:|:-------------|------------------------:|:----------------------|
| baseline                      | policy_keep_0ms_anchor                               | 1.19771  | True         |             0.502658    | False                 |
| baseline                      | policy_wait_to_latest_anchor                         | 0.695048 | True         |             4.15347e-07 | False                 |
| oracle                        | oracle_best_anchor_upper_bound                       | 0.612475 | False        |            -0.0825726   | True                  |
| vehicle                       | vehicle_nearest_train_prototype_k40                  | 0.878536 | True         |             0.183488    | False                 |
| vehicle_oracle                | vehicle_candidate_oracle_k40                         | 0.616603 | False        |            -0.0784452   | True                  |
| bio_top1_val_chosen           | subject_seq_pca72:bio_nearest_within_vehicle_k40     | 0.89398  | True         |             0.198932    | False                 |
| bio_top3_oracle_val_chosen    | subject_seq_pca72:bio_top3_oracle_within_vehicle_k40 | 0.772747 | False        |             0.0776989   | False                 |
| bio_top1_test_best_diagnostic | calibrated_screened64:bio_nearest_within_vehicle_k40 | 0.87444  | False        |             0.179392    | False                 |
| bio_top3_oracle_test_best     | subject_summary64:bio_top3_oracle_within_vehicle_k40 | 0.673823 | False        |            -0.0212249   | True                  |

## test bad_top10 K=40 生理排序诊断

| raw_set                   |   n |   vehicle_nearest_rmse_mean |   vehicle_candidate_oracle_rmse_mean |   bio_top1_oracle_rmse_mean |   bio_top3_oracle_rmse_mean |   bio_best_candidate_rank_mean |   bio_best_in_top3_rate |   bio_best_in_top5_rate |   bio_distance_rmse_rank_corr_mean |
|:--------------------------|----:|----------------------------:|-------------------------------------:|----------------------------:|----------------------------:|-------------------------------:|------------------------:|------------------------:|-----------------------------------:|
| calibrated_screened64     |  19 |                    0.878536 |                             0.616603 |                    0.87444  |                    0.771211 |                        19.8947 |               0.0526316 |               0.105263  |                        -0.039844   |
| subject_seq_pca72         |  19 |                    0.878536 |                             0.616603 |                    0.89398  |                    0.772747 |                        18.7895 |               0.105263  |               0.157895  |                        -0.0240681  |
| recording_summary64       |  19 |                    0.878536 |                             0.616603 |                    0.903212 |                    0.752034 |                        19.7895 |               0.0526316 |               0.105263  |                        -0.00374735 |
| calibrated_low_identity48 |  19 |                    0.878536 |                             0.616603 |                    0.908005 |                    0.830328 |                        20.1579 |               0.0526316 |               0.0526316 |                        -0.0661491  |
| recording_seq_pca72       |  19 |                    0.878536 |                             0.616603 |                    0.922643 |                    0.751149 |                        19.7368 |               0.105263  |               0.157895  |                        -0.036025   |
| subject_summary64         |  19 |                    0.878536 |                             0.616603 |                    0.934999 |                    0.673823 |                        20.6842 |               0.105263  |               0.157895  |                         0.00984655 |

## 判读

- 最好的 bio top1 检索为 `calibrated_screened64`，test bad_top10 RMSE `0.8744`。
- 最好的 bio top3 oracle 为 `subject_summary64`，test bad_top10 RMSE `0.6738`。
- vehicle candidate oracle 为 `0.6166`，fixed wait-latest 为 `0.6950`。
- 生理最近邻 top1 未低于 fixed wait-latest，说明直接用生理距离不能完成 goal。
- bio top3 oracle 低于 fixed wait-latest，说明若另有强选择器，生理邻域内仍有少量上界。
- 本实验的核心不是提交新模型，而是判断生理是否真的能在车辆相似样本之间消歧。

## 关键图

- `figures\v272_test_badtop10_ambiguity_decision.png`
- `figures\v272_test_badtop10_bio_rank_capture.png`