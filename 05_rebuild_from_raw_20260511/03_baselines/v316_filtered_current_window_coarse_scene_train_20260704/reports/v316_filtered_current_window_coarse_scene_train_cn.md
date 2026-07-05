# 第316版过滤当前窗口样本后完整重训

## 结论

- 本轮按第315版保留清单完整重训，候选选择只看过滤后的验证集。
- 选中候选：`v316_filtered_scene_init_aux003_film005_h64`。
- 参照第300版：`v300_full_joint_h64_no_subject`。
- 旧第307版对照：`v307_previous_v307_coarse_scene_init_aux003_film005_h64`。
- 训练保留事件：`650`；验证保留事件：`211`；测试保留事件：`222`。

## 过滤后测试集核心结果

- 全部：第300版 `0.525580`；旧第307版 `0.496950`；第316版 `0.502633`。
- 原困难前10：第300版 `0.859987`；旧第307版 `0.777797`；第316版 `0.800171`。
- 原困难前20：第300版 `0.703038`；旧第307版 `0.651121`；第316版 `0.660814`。

## 测试集摘要表

| model_name | split | group | n | sample_rmse_mean | sample_rmse_median | sample_rmse_p90 |
| --- | --- | --- | --- | --- | --- | --- |
| v300_full_joint_h64_no_subject | test | all | 222 | 0.52558 | 0.429346 | 0.945879 |
| v300_full_joint_h64_no_subject | test | within_bad_top10 | 24 | 0.859987 | 0.690337 | 1.34919 |
| v300_full_joint_h64_no_subject | test | within_bad_top20 | 46 | 0.703038 | 0.633825 | 1.21012 |
| v300_full_joint_h64_no_subject | test | strong_steer | 125 | 0.621347 | 0.518684 | 1.04093 |
| v307_previous_v307_coarse_scene_init_aux003_film005_h64 | test | all | 222 | 0.49695 | 0.395903 | 0.825269 |
| v307_previous_v307_coarse_scene_init_aux003_film005_h64 | test | within_bad_top10 | 24 | 0.777797 | 0.694593 | 1.33936 |
| v307_previous_v307_coarse_scene_init_aux003_film005_h64 | test | within_bad_top20 | 46 | 0.651121 | 0.59218 | 1.1223 |
| v307_previous_v307_coarse_scene_init_aux003_film005_h64 | test | strong_steer | 125 | 0.59405 | 0.477759 | 0.965698 |
| v316_filtered_scene_init_aux003_film005_h64 | test | all | 222 | 0.502633 | 0.397823 | 0.856381 |
| v316_filtered_scene_init_aux003_film005_h64 | test | within_bad_top10 | 24 | 0.800171 | 0.70186 | 1.40982 |
| v316_filtered_scene_init_aux003_film005_h64 | test | within_bad_top20 | 46 | 0.660814 | 0.590447 | 1.18266 |
| v316_filtered_scene_init_aux003_film005_h64 | test | strong_steer | 125 | 0.601547 | 0.471727 | 1.00905 |

## 验证选模表

| model_name | test_used_for_selection | selected_by | best_epoch | best_val_loss | training_seconds | validation_selection_score | val_sample_rmse_weighted | val_tail_rmse_weighted | val_strong_under_rate_weighted | val_peak_ratio_weighted | delay0_val_all_delta_vs_v300 | delay0_val_bad10_delta_vs_v300 | delay0_val_bad20_delta_vs_v300 | passes_val_all_noharm_vs_v300 | passes_val_bad10_noharm_vs_v300 | passes_val_bad20_noharm_vs_v300 | passes_v304_noharm_gate | validation_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v316_filtered_scene_init_aux003_film005_h64 | False | validation_original_remaining_plus_delay0_noharm | 18 | 0.420891 | 27.0553 | 0.733762 | 0.46656 | 0.514556 | 0.0661607 | 1.45592 | -0.01111 | -0.118104 | -0.053523 | True | True | True | True | 1 |
| v316_filtered_scene_init_aux006_film010_hard110_h64 | False | validation_original_remaining_plus_delay0_noharm | 23 | 0.482757 | 27.1098 | 0.739577 | 0.469143 | 0.520061 | 0.0693567 | 1.45195 | -0.015834 | -0.0991994 | -0.0687178 | True | True | True | True | 2 |
| v316_filtered_scene_init_aux005_film010_h64 | False | validation_original_remaining_plus_delay0_noharm | 16 | 0.434451 | 21.2204 | 0.742316 | 0.471233 | 0.520325 | 0.0728078 | 1.48239 | -0.0131618 | -0.151097 | -0.0853578 | True | True | True | True | 3 |

## 事件辅助头

| model_name | split | group | n | accuracy | balanced_accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v316_filtered_scene_init_aux003_film005_h64 | train | all_rolling | 3900 | 0.628974 | 0.715627 | 0.666164 | 0.626494 |
| v316_filtered_scene_init_aux003_film005_h64 | train | delay0_only | 650 | 0.62 | 0.67914 | 0.646282 | 0.620566 |
| v316_filtered_scene_init_aux003_film005_h64 | val | all_rolling | 1266 | 0.488942 | 0.567822 | 0.512465 | 0.480212 |
| v316_filtered_scene_init_aux003_film005_h64 | val | delay0_only | 211 | 0.454976 | 0.481219 | 0.460283 | 0.456052 |
| v316_filtered_scene_init_aux003_film005_h64 | test | all_rolling | 1332 | 0.472973 | 0.523495 | 0.500959 | 0.476195 |
| v316_filtered_scene_init_aux003_film005_h64 | test | delay0_only | 222 | 0.468468 | 0.495956 | 0.48829 | 0.47478 |

## 边界

- 本轮没有重切第315版重锚定候选，只是把它们隔离出当前窗口任务。
- 第315版隔离清单不参与训练、验证选模或测试主统计。
- 旧第307版在同一过滤后测试集上只作为对照，不参与选模。
