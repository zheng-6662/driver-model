# v307 coarse scene-label conditioned 曲线模型

## 这一步做了什么

v307 复用 v304 的 fixed-label conditioned 模型结构，但把条件标签从 v301/v305 的细事件类型替换为 v306 的粗场景标签。

粗标签包括：下坡过弯、平路过弯、连续变道/连续左右修正、紧急变道/猛打方向失稳、其他/不确定。

## validation-only 选择

| model_name                                        | test_used_for_selection   | selected_by                                      |   best_epoch |   best_val_loss |   training_seconds | config_json                                                                                                                                                                                                                                                                                                                                                                                                                             |   validation_selection_score |   val_sample_rmse_weighted |   val_tail_rmse_weighted |   val_strong_under_rate_weighted |   val_peak_ratio_weighted |   delay0_val_all_delta_vs_v300 |   delay0_val_bad10_delta_vs_v300 |   delay0_val_bad20_delta_vs_v300 | passes_val_all_noharm_vs_v300   | passes_val_bad10_noharm_vs_v300   | passes_val_bad20_noharm_vs_v300   | passes_v304_noharm_gate   |   validation_rank |
|:--------------------------------------------------|:--------------------------|:-------------------------------------------------|-------------:|----------------:|-------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------:|---------------------------:|-------------------------:|---------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|---------------------------------:|:--------------------------------|:----------------------------------|:----------------------------------|:--------------------------|------------------:|
| v307_coarse_scene_init_aux003_film005_h64         | False                     | validation_original_remaining_plus_delay0_noharm |           27 |        0.446999 |            44.4975 | {"aux_weight": 0.03, "batch_size": 384, "dropout": 0.06, "event_embed_dim": 64, "film_scale": 0.05, "hard_event_extra": 0.0, "hidden_dim": 64, "init_from_v300": true, "lr": 0.0002, "max_epochs": 55, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 112, "n_heads": 4, "n_layers": 3, "patience": 9, "roll_hidden": 128, "smooth_weight": 0.02, "v300_init_model_name": "v300_full_joint_h64_no_subject", "weight_decay": 0.0003}  |                     0.719238 |                   0.454737 |                 0.506707 |                        0.0743167 |                   4.08399 |                     -0.0228758 |                        -0.124626 |                       -0.0667719 | True                            | True                              | True                              | True                      |                 1 |
| v307_coarse_scene_init_aux005_film010_h64         | False                     | validation_original_remaining_plus_delay0_noharm |           22 |        0.460158 |            30.0225 | {"aux_weight": 0.05, "batch_size": 384, "dropout": 0.08, "event_embed_dim": 64, "film_scale": 0.1, "hard_event_extra": 0.0, "hidden_dim": 64, "init_from_v300": true, "lr": 0.0002, "max_epochs": 60, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 128, "n_heads": 4, "n_layers": 3, "patience": 10, "roll_hidden": 160, "smooth_weight": 0.025, "v300_init_model_name": "v300_full_joint_h64_no_subject", "weight_decay": 0.0004} |                     0.726854 |                   0.459755 |                 0.510689 |                        0.078361  |                   3.62273 |                     -0.0191306 |                        -0.136234 |                       -0.0771404 | True                            | True                              | True                              | True                      |                 2 |
| v307_coarse_scene_init_aux006_film010_hard110_h64 | False                     | validation_original_remaining_plus_delay0_noharm |           26 |        0.510516 |            34.4571 | {"aux_weight": 0.06, "batch_size": 384, "dropout": 0.08, "event_embed_dim": 64, "film_scale": 0.1, "hard_event_extra": 0.1, "hidden_dim": 64, "init_from_v300": true, "lr": 0.0002, "max_epochs": 60, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 128, "n_heads": 4, "n_layers": 3, "patience": 10, "roll_hidden": 160, "smooth_weight": 0.025, "v300_init_model_name": "v300_full_joint_h64_no_subject", "weight_decay": 0.0004} |                     0.733918 |                   0.464997 |                 0.517526 |                        0.0677184 |                   4.49114 |                     -0.0113373 |                        -0.088827 |                       -0.0567434 | True                            | True                              | True                              | True                      |                 3 |

validation 选择出的 v307 候选：`v307_coarse_scene_init_aux003_film005_h64`。
v300 参照模型：`v300_full_joint_h64_no_subject`。

## test delay0 对比

| model_name                                | split   | group             |   n |   sample_rmse_mean |   sample_rmse_median |   sample_rmse_p90 |
|:------------------------------------------|:--------|:------------------|----:|-------------------:|---------------------:|------------------:|
| v300_full_joint_h64_no_subject            | test    | all               | 232 |           0.519805 |             0.425889 |          0.962683 |
| v300_full_joint_h64_no_subject            | test    | within_bad_top10  |  24 |           0.859987 |             0.690337 |          1.34919  |
| v300_full_joint_h64_no_subject            | test    | within_bad_top20  |  47 |           0.690942 |             0.633268 |          1.20787  |
| v300_full_joint_h64_no_subject            | test    | vehicle_ambiguous | 120 |           0.525913 |             0.450818 |          0.954555 |
| v300_full_joint_h64_no_subject            | test    | strong_steer      | 125 |           0.621347 |             0.518684 |          1.04093  |
| v307_coarse_scene_init_aux003_film005_h64 | test    | all               | 232 |           0.496138 |             0.395261 |          0.850443 |
| v307_coarse_scene_init_aux003_film005_h64 | test    | within_bad_top10  |  24 |           0.777797 |             0.694593 |          1.33936  |
| v307_coarse_scene_init_aux003_film005_h64 | test    | within_bad_top20  |  47 |           0.639121 |             0.588337 |          1.11962  |
| v307_coarse_scene_init_aux003_film005_h64 | test    | vehicle_ambiguous | 120 |           0.504829 |             0.421173 |          0.828764 |
| v307_coarse_scene_init_aux003_film005_h64 | test    | strong_steer      | 125 |           0.59405  |             0.477759 |          0.965698 |

简表：

- test/all：v300 `0.5198` -> v307 `0.4961`。
- test/within_bad_top10：v300 `0.8600` -> v307 `0.7778`。
- test/within_bad_top20：v300 `0.6909` -> v307 `0.6391`。

## 粗场景辅助头

| model_name                                | split   | group       |    n |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |
|:------------------------------------------|:--------|:------------|-----:|-----------:|--------------------:|-----------:|--------------:|
| v307_coarse_scene_init_aux003_film005_h64 | train   | all_rolling | 4212 |   0.689934 |            0.759227 |   0.721975 |      0.689841 |
| v307_coarse_scene_init_aux003_film005_h64 | train   | delay0_only |  702 |   0.663818 |            0.709971 |   0.686376 |      0.664991 |
| v307_coarse_scene_init_aux003_film005_h64 | val     | all_rolling | 1398 |   0.508584 |            0.581336 |   0.527826 |      0.503403 |
| v307_coarse_scene_init_aux003_film005_h64 | val     | delay0_only |  233 |   0.467811 |            0.476743 |   0.460926 |      0.47064  |
| v307_coarse_scene_init_aux003_film005_h64 | test    | all_rolling | 1392 |   0.489943 |            0.525329 |   0.509218 |      0.494088 |
| v307_coarse_scene_init_aux003_film005_h64 | test    | delay0_only |  232 |   0.491379 |            0.505393 |   0.507406 |      0.499028 |

## 当前判断

- 如果 v307 接近或优于 v304，说明粗场景标签已经保留了主要条件信息，后续人工审核成本可明显降低。
- 如果 v307 明显弱于 v304，说明急左/急右/复合制动等细粒度信息仍有价值，需要在粗场景内保留少量二级标签。
- v307 中直道内连续/紧急子类仍有 v305/v301 seed 成分，不能直接写成最终人工标签。

## guardrail

```json
{
  "pass": true,
  "version": "v307_coarse_scene_label_conditioned_curve_model_20260704",
  "model_structure_changed": true,
  "output_target_unchanged": "21_point_steering_delta_curve",
  "uses_roll_cause_summary_as_input": true,
  "uses_coarse_scene_labels_as_features": true,
  "coarse_scene_label_source": "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v306_coarse_predefined_scene_label_table_20260704\\tables\\v306_coarse_scene_event_labels.csv",
  "curve_scene_labels_from_current_scene_type": true,
  "noncurve_subtypes_require_manual_or_experiment_confirmation": true,
  "uses_future_behavior_seed_for_some_noncurve_subtypes": true,
  "deployable_without_noncurve_manual_confirmation": false,
  "uses_test_error_as_features": false,
  "candidate_selection_uses_test": false,
  "same_event_never_repeated_across_splits": true,
  "event_in_multiple_splits_n": 0,
  "event_without_6_delay_rows_n": 0,
  "event_n": 1167,
  "rolling_sample_n": 7002,
  "roll_cause_feature_n": 301,
  "coarse_scene_class_n": 5,
  "coarse_scene_class_names": [
    "curve_downhill",
    "curve_flat",
    "continuous_lane_change",
    "emergency_lane_change_instability",
    "other_or_uncertain"
  ],
  "v300_reference_model": "v300_full_joint_h64_no_subject",
  "selected_v307_model": "v307_coarse_scene_init_aux003_film005_h64",
  "selected_passes_v307_noharm_gate": true,
  "selected_val_all_delta_vs_v300": -0.02287581516629633,
  "selected_val_bad10_delta_vs_v300": -0.12462632233897841,
  "selected_val_bad20_delta_vs_v300": -0.0667718942178056,
  "device": "cuda",
  "runtime_seconds": 128.97327637672424,
  "figure_paths": [
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v307_coarse_scene_label_conditioned_curve_model_20260704\\figures\\v304_training_history.png",
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v307_coarse_scene_label_conditioned_curve_model_20260704\\figures\\v304_test_delay0_group_rmse.png",
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v307_coarse_scene_label_conditioned_curve_model_20260704\\figures\\v304_event_aux_macro_f1.png"
  ]
}
```