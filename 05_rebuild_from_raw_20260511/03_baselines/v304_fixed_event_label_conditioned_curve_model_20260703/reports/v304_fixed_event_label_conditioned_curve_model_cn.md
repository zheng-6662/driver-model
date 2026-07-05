# v304 fixed event-label conditioned 曲线模型

## 这一步做了什么

v304 在 v303 的基础上继续改结构：输出仍是 21 点 steering_delta 曲线，但把训练前固定的 event_primary_type 做成 event embedding，作为曲线预测的条件输入。

结构上：历史/道路/phase/point token 仍走曲线解码主干；roll-cause summary 走单独 MLP 编码器；固定事件标签走 event embedding；两者共同通过 token 拼接和 FiLM 调制影响曲线隐藏表示。

重要边界：当前 event_primary_type 来自 v301 future_behavior_auto_draft，因此 v304 是 known-label/oracle upper-bound 实验。只有当事件标签由人工审核或实验条件在预测前给出时，它才可以被解释为正式可部署输入。

## validation-only 选择

| model_name                                       | test_used_for_selection   | selected_by                                      |   best_epoch |   best_val_loss |   training_seconds | config_json                                                                                                                                                                                                                                                                                                                                                                                                                             |   validation_selection_score |   val_sample_rmse_weighted |   val_tail_rmse_weighted |   val_strong_under_rate_weighted |   val_peak_ratio_weighted |   delay0_val_all_delta_vs_v300 |   delay0_val_bad10_delta_vs_v300 |   delay0_val_bad20_delta_vs_v300 | passes_val_all_noharm_vs_v300   | passes_val_bad10_noharm_vs_v300   | passes_val_bad20_noharm_vs_v300   | passes_v304_noharm_gate   |   validation_rank |
|:-------------------------------------------------|:--------------------------|:-------------------------------------------------|-------------:|----------------:|-------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------:|---------------------------:|-------------------------:|---------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|---------------------------------:|:--------------------------------|:----------------------------------|:----------------------------------|:--------------------------|------------------:|
| v304_fixed_event_init_aux005_film010_h64         | False                     | validation_original_remaining_plus_delay0_noharm |           12 |        0.494838 |            18.9915 | {"aux_weight": 0.05, "batch_size": 384, "dropout": 0.08, "event_embed_dim": 64, "film_scale": 0.1, "hard_event_extra": 0.0, "hidden_dim": 64, "init_from_v300": true, "lr": 0.0002, "max_epochs": 60, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 128, "n_heads": 4, "n_layers": 3, "patience": 10, "roll_hidden": 160, "smooth_weight": 0.025, "v300_init_model_name": "v300_full_joint_h64_no_subject", "weight_decay": 0.0004} |                     0.727624 |                   0.461051 |                 0.511387 |                        0.0725283 |                   4.04435 |                    -0.0139821  |                       -0.0595518 |                       -0.047654  | True                            | True                              | True                              | True                      |                 1 |
| v304_fixed_event_init_aux006_film010_hard110_h64 | False                     | validation_original_remaining_plus_delay0_noharm |           15 |        0.51407  |            20.9686 | {"aux_weight": 0.06, "batch_size": 384, "dropout": 0.08, "event_embed_dim": 64, "film_scale": 0.1, "hard_event_extra": 0.1, "hidden_dim": 64, "init_from_v300": true, "lr": 0.0002, "max_epochs": 60, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 128, "n_heads": 4, "n_layers": 3, "patience": 10, "roll_hidden": 160, "smooth_weight": 0.025, "v300_init_model_name": "v300_full_joint_h64_no_subject", "weight_decay": 0.0004} |                     0.731049 |                   0.463    |                 0.514629 |                        0.071562  |                   4.25234 |                    -0.0159971  |                       -0.100254  |                       -0.0653324 | True                            | True                              | True                              | True                      |                 2 |
| v304_fixed_event_init_aux003_film005_h64         | False                     | validation_original_remaining_plus_delay0_noharm |           10 |        0.480429 |            19.8004 | {"aux_weight": 0.03, "batch_size": 384, "dropout": 0.06, "event_embed_dim": 64, "film_scale": 0.05, "hard_event_extra": 0.0, "hidden_dim": 64, "init_from_v300": true, "lr": 0.0002, "max_epochs": 55, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 112, "n_heads": 4, "n_layers": 3, "patience": 9, "roll_hidden": 128, "smooth_weight": 0.02, "v300_init_model_name": "v300_full_joint_h64_no_subject", "weight_decay": 0.0003}  |                     0.743852 |                   0.470436 |                 0.52336  |                        0.0782387 |                   3.64132 |                    -0.00318401 |                       -0.0429407 |                       -0.0269513 | True                            | True                              | True                              | True                      |                 3 |

validation 选择出的 v304 候选：`v304_fixed_event_init_aux005_film010_h64`。
v300 参照模型：`v300_full_joint_h64_no_subject`。

## test delay0 对比

| model_name                               | split   | group             |   n |   sample_rmse_mean |   sample_rmse_median |   sample_rmse_p90 |
|:-----------------------------------------|:--------|:------------------|----:|-------------------:|---------------------:|------------------:|
| v300_full_joint_h64_no_subject           | test    | all               | 232 |           0.519805 |             0.425889 |          0.962683 |
| v300_full_joint_h64_no_subject           | test    | within_bad_top10  |  24 |           0.859987 |             0.690337 |          1.34919  |
| v300_full_joint_h64_no_subject           | test    | within_bad_top20  |  47 |           0.690942 |             0.633268 |          1.20787  |
| v300_full_joint_h64_no_subject           | test    | vehicle_ambiguous | 120 |           0.525913 |             0.450818 |          0.954555 |
| v300_full_joint_h64_no_subject           | test    | strong_steer      | 125 |           0.621347 |             0.518684 |          1.04093  |
| v304_fixed_event_init_aux005_film010_h64 | test    | all               | 232 |           0.498102 |             0.399941 |          0.858887 |
| v304_fixed_event_init_aux005_film010_h64 | test    | within_bad_top10  |  24 |           0.832204 |             0.712344 |          1.3902   |
| v304_fixed_event_init_aux005_film010_h64 | test    | within_bad_top20  |  47 |           0.657669 |             0.582232 |          1.14106  |
| v304_fixed_event_init_aux005_film010_h64 | test    | vehicle_ambiguous | 120 |           0.505271 |             0.419778 |          0.870389 |
| v304_fixed_event_init_aux005_film010_h64 | test    | strong_steer      | 125 |           0.595683 |             0.489962 |          1.03471  |

简表：

- test/all：v300 `0.5198` -> v304 `0.4981`。
- test/within_bad_top10：v300 `0.8600` -> v304 `0.8322`。
- test/within_bad_top20：v300 `0.6909` -> v304 `0.6577`。

## 事件辅助头

| model_name                               | split   | group       |    n |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |
|:-----------------------------------------|:--------|:------------|-----:|-----------:|--------------------:|-----------:|--------------:|
| v304_fixed_event_init_aux005_film010_h64 | train   | all_rolling | 4212 |   0.716524 |            0.769045 |   0.663838 |      0.72777  |
| v304_fixed_event_init_aux005_film010_h64 | train   | delay0_only |  702 |   0.680912 |            0.69726  |   0.610697 |      0.691468 |
| v304_fixed_event_init_aux005_film010_h64 | val     | all_rolling | 1398 |   0.519313 |            0.344123 |   0.29486  |      0.544758 |
| v304_fixed_event_init_aux005_film010_h64 | val     | delay0_only |  233 |   0.472103 |            0.281006 |   0.237473 |      0.492115 |
| v304_fixed_event_init_aux005_film010_h64 | test    | all_rolling | 1392 |   0.554598 |            0.481649 |   0.385499 |      0.570361 |
| v304_fixed_event_init_aux005_film010_h64 | test    | delay0_only |  232 |   0.517241 |            0.418335 |   0.378737 |      0.533271 |

## 当前判断

- 如果 v304 显著优于 v303，说明“事件类型已知”本身对轨迹预测有上限价值。
- 是否接受 v304，不看 test 选择，而看 validation no-harm gate 和最终 test 分组报告。
- 由于当前标签来自未来行为草稿，本轮不能直接写成部署效果；它回答的是固定事件标签是否值得人工标注/实验条件接入。
- 如果 bad_top10 仍不够，下一步应把固定事件标签用于 mixture-of-experts 路由或多模态轨迹条件分布，而不是回到删除样本/轻量 residual。

## guardrail

```json
{
  "pass": true,
  "version": "v304_fixed_event_label_conditioned_curve_model_20260703",
  "model_structure_changed": true,
  "output_target_unchanged": "21_point_steering_delta_curve",
  "uses_roll_cause_summary_as_input": true,
  "uses_fixed_event_labels_as_features": true,
  "fixed_event_label_source": "v301_future_behavior_auto_draft",
  "fixed_event_label_deployable_without_external_or_manual_label": false,
  "uses_future_event_labels_as_features": true,
  "uses_event_labels_as_auxiliary_targets": true,
  "uses_test_error_as_features": false,
  "candidate_selection_uses_test": false,
  "same_event_never_repeated_across_splits": true,
  "event_in_multiple_splits_n": 0,
  "event_without_6_delay_rows_n": 0,
  "event_n": 1167,
  "rolling_sample_n": 7002,
  "roll_cause_feature_n": 301,
  "event_class_n": 9,
  "v300_reference_model": "v300_full_joint_h64_no_subject",
  "selected_v304_model": "v304_fixed_event_init_aux005_film010_h64",
  "selected_passes_v304_noharm_gate": true,
  "selected_val_all_delta_vs_v300": -0.013982070266177149,
  "selected_val_bad10_delta_vs_v300": -0.05955176490048575,
  "selected_val_bad20_delta_vs_v300": -0.04765404586462263,
  "device": "cuda",
  "runtime_seconds": 77.30701661109924,
  "figure_paths": [
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v304_fixed_event_label_conditioned_curve_model_20260703\\figures\\v304_training_history.png",
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v304_fixed_event_label_conditioned_curve_model_20260703\\figures\\v304_test_delay0_group_rmse.png",
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v304_fixed_event_label_conditioned_curve_model_20260703\\figures\\v304_event_aux_macro_f1.png"
  ]
}
```