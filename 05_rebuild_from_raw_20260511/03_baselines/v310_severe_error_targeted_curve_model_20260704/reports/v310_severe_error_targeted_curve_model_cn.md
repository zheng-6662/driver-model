# v310 severe-error targeted curve model

## 这一步做了什么

v310 针对 v309 图册中暴露的严重方向/意图错误做小步改造，但没有把 test severe 错例拿去训练或选模。

- 训练初始化：v307 selected checkpoint。
- 训练改动：train/val 目标曲线形态权重 + 方向/幅值/平直三类轻量形状约束。
- 选模规则：仍然只看 validation，不看 test。
- v309 severe 表用途：训练结束后诊断 hard cases，不参与训练和选模。

## validation-only 选择

| model_name                    | test_used_for_selection   | selected_by                                      |   best_epoch |   best_val_loss |   training_seconds | config_json                                                                                                                                                                                                                                                                                                                                                                                                                                                     |   validation_selection_score |   val_sample_rmse_weighted |   val_tail_rmse_weighted |   val_strong_under_rate_weighted |   val_peak_ratio_weighted |   delay0_val_all_delta_vs_v300 |   delay0_val_bad10_delta_vs_v300 |   delay0_val_bad20_delta_vs_v300 | passes_val_all_noharm_vs_v300   | passes_val_bad10_noharm_vs_v300   | passes_val_bad20_noharm_vs_v300   | passes_v304_noharm_gate   |   validation_rank |
|:------------------------------|:--------------------------|:-------------------------------------------------|-------------:|----------------:|-------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------:|---------------------------:|-------------------------:|---------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|---------------------------------:|:--------------------------------|:----------------------------------|:----------------------------------|:--------------------------|------------------:|
| v310_v307init_shape_guard_lo  | False                     | validation_original_remaining_plus_delay0_noharm |            9 |        0.483769 |           15.5841  | {"amplitude_weight": 0.03, "aux_weight": 0.03, "base_hard_event_extra": 0.15, "batch_size": 384, "direction_weight": 0.015, "dropout": 0.06, "event_embed_dim": 64, "film_scale": 0.05, "flat_weight": 0.012, "hidden_dim": 64, "lr": 6e-05, "max_epochs": 34, "min_lr": 5e-06, "mixer_layers": 2, "mlp_hidden": 112, "n_heads": 4, "n_layers": 3, "patience": 7, "roll_hidden": 128, "smooth_weight": 0.02, "target_shape_extra": 0.2, "weight_decay": 0.0003} |                     0.719326 |                   0.454856 |                 0.506955 |                        0.0732879 |                   4.09518 |                     -0.0244751 |                        -0.136717 |                       -0.0695029 | True                            | True                              | True                              | True                      |                 1 |
| v310_v307init_shape_guard_hi  | False                     | validation_original_remaining_plus_delay0_noharm |            5 |        0.553736 |           11.0235  | {"amplitude_weight": 0.1, "aux_weight": 0.03, "base_hard_event_extra": 0.35, "batch_size": 384, "direction_weight": 0.05, "dropout": 0.06, "event_embed_dim": 64, "film_scale": 0.05, "flat_weight": 0.04, "hidden_dim": 64, "lr": 5e-05, "max_epochs": 34, "min_lr": 5e-06, "mixer_layers": 2, "mlp_hidden": 112, "n_heads": 4, "n_layers": 3, "patience": 7, "roll_hidden": 128, "smooth_weight": 0.02, "target_shape_extra": 0.65, "weight_decay": 0.0003}   |                     0.720889 |                   0.455831 |                 0.50844  |                        0.0722591 |                   4.04712 |                     -0.0215008 |                        -0.139302 |                       -0.0676227 | True                            | True                              | True                              | True                      |                 2 |
| v310_v307init_shape_guard_mid | False                     | validation_original_remaining_plus_delay0_noharm |            4 |        0.522098 |            9.69298 | {"amplitude_weight": 0.06, "aux_weight": 0.03, "base_hard_event_extra": 0.25, "batch_size": 384, "direction_weight": 0.03, "dropout": 0.06, "event_embed_dim": 64, "film_scale": 0.05, "flat_weight": 0.025, "hidden_dim": 64, "lr": 6e-05, "max_epochs": 34, "min_lr": 5e-06, "mixer_layers": 2, "mlp_hidden": 112, "n_heads": 4, "n_layers": 3, "patience": 7, "roll_hidden": 128, "smooth_weight": 0.02, "target_shape_extra": 0.4, "weight_decay": 0.0003}  |                     0.725822 |                   0.458861 |                 0.511183 |                        0.0757982 |                   4.02425 |                     -0.0171659 |                        -0.127028 |                       -0.0652301 | True                            | True                              | True                              | True                      |                 3 |

validation 选出的 v310 候选：`v310_v307init_shape_guard_lo`。
v307 参照模型：`v307_coarse_scene_init_aux003_film005_h64`。
v300 参照模型：`v300_full_joint_h64_no_subject`。

## test delay0 常规分组

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
| v310_v307init_shape_guard_lo              | test    | all               | 232 |           0.494998 |             0.388132 |          0.823856 |
| v310_v307init_shape_guard_lo              | test    | within_bad_top10  |  24 |           0.775882 |             0.701454 |          1.31     |
| v310_v307init_shape_guard_lo              | test    | within_bad_top20  |  47 |           0.63766  |             0.592371 |          1.17631  |
| v310_v307init_shape_guard_lo              | test    | vehicle_ambiguous | 120 |           0.50309  |             0.420101 |          0.831196 |
| v310_v307init_shape_guard_lo              | test    | strong_steer      | 125 |           0.592264 |             0.460137 |          0.9693   |

简表：

- test/all：v300 `0.5198` -> v307 `0.4961` -> v310 `0.4950`
- test/within_bad_top10：v300 `0.8600` -> v307 `0.7778` -> v310 `0.7759`
- test/within_bad_top20：v300 `0.6909` -> v307 `0.6391` -> v310 `0.6377`

## v309 severe 诊断分组

| model_name                                | diagnostic_group         |   n |   sample_rmse_mean |   sample_rmse_median |   tail_rmse_mean |   direction_acc |   strong_under_rate |   peak_ratio_mean |
|:------------------------------------------|:-------------------------|----:|-------------------:|---------------------:|-----------------:|----------------:|--------------------:|------------------:|
| v300_full_joint_h64_no_subject            | v309_severe_all37        |  37 |           0.798697 |             0.611596 |         0.968796 |        0.72973  |            0.243243 |          3.60357  |
| v307_coarse_scene_init_aux003_film005_h64 | v309_severe_all37        |  37 |           0.8884   |             0.690048 |         1.10232  |        0.756757 |            0.27027  |          4.16573  |
| v310_v307init_shape_guard_lo              | v309_severe_all37        |  37 |           0.890705 |             0.686383 |         1.10587  |        0.756757 |            0.243243 |          4.29109  |
| v300_full_joint_h64_no_subject            | user_screenshot_5        |   5 |           1.60334  |             1.19885  |         1.77152  |        0.6      |            0.2      |          5.76674  |
| v307_coarse_scene_init_aux003_film005_h64 | user_screenshot_5        |   5 |           1.75506  |             1.59088  |         2.00353  |        0.6      |            0.2      |          7.83728  |
| v310_v307init_shape_guard_lo              | user_screenshot_5        |   5 |           1.77568  |             1.64219  |         2.02774  |        0.6      |            0.2      |          7.97534  |
| v300_full_joint_h64_no_subject            | opposite_peak_direction  |   8 |           0.749701 |             0.593643 |         0.924807 |        0        |            0.125    |          1.22208  |
| v307_coarse_scene_init_aux003_film005_h64 | opposite_peak_direction  |   8 |           0.734838 |             0.610958 |         0.918155 |        0        |            0.125    |          1.28825  |
| v310_v307init_shape_guard_lo              | opposite_peak_direction  |   8 |           0.730708 |             0.59605  |         0.910637 |        0        |            0.125    |          1.28887  |
| v300_full_joint_h64_no_subject            | false_large_maneuver     |   3 |           0.951468 |             1.10558  |         1.30274  |        0.666667 |            0        |         34.3734   |
| v307_coarse_scene_init_aux003_film005_h64 | false_large_maneuver     |   3 |           1.22966  |             1.4703   |         1.68826  |        0.666667 |            0        |         40.2368   |
| v310_v307init_shape_guard_lo              | false_large_maneuver     |   3 |           1.28136  |             1.54142  |         1.75954  |        0.666667 |            0        |         41.8575   |
| v300_full_joint_h64_no_subject            | missed_extreme_amplitude |   8 |           1.40987  |             1.01288  |         1.82598  |        1        |            1        |          0.304576 |
| v307_coarse_scene_init_aux003_film005_h64 | missed_extreme_amplitude |   8 |           1.39877  |             0.890424 |         1.81634  |        1        |            1        |          0.316931 |
| v310_v307init_shape_guard_lo              | missed_extreme_amplitude |   8 |           1.41269  |             0.884948 |         1.8358   |        1        |            1        |          0.31825  |

## 用户截图 5 个事件逐模型对比

|   severe_rank |   screenshot_rank |   gallery_rank | event_uid                                         | coarse_scene_label_cn   |   v307_rmse |   v300_rmse |   delta_v307_minus_v300 |   true_peak |   v307_peak | error_tags                                                              | error_reason_cn                                                                                 | model_name                                |   sample_rmse |   tail_rmse |   peak_ratio | direction_ok   | strong_under   |
|--------------:|------------------:|---------------:|:--------------------------------------------------|:------------------------|------------:|------------:|------------------------:|------------:|------------:|:------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:------------------------------------------|--------------:|------------:|-------------:|:---------------|:---------------|
|             2 |                17 |             17 | zx_Entity_Recording_2025_09_27_16_46_13_v108_035  | 紧急变道/猛打方向失稳   |    3.36184  |    3.31255  |               0.0492921 |    4.31917  |    0.507367 | missed_extreme_amplitude;large_event_high_rmse;shown_in_user_screenshot | 极端动作幅值严重低估: true_peak=4.319, v307_peak=0.507；大动作场景整体误差高: v307_rmse=3.362   | v300_full_joint_h64_no_subject            |      3.31255  |    3.93009  |     0.113737 | True           | True           |
|             2 |                17 |             17 | zx_Entity_Recording_2025_09_27_16_46_13_v108_035  | 紧急变道/猛打方向失稳   |    3.36184  |    3.31255  |               0.0492921 |    4.31917  |    0.507367 | missed_extreme_amplitude;large_event_high_rmse;shown_in_user_screenshot | 极端动作幅值严重低估: true_peak=4.319, v307_peak=0.507；大动作场景整体误差高: v307_rmse=3.362   | v307_coarse_scene_init_aux003_film005_h64 |      3.36184  |    4.01209  |     0.117469 | True           | True           |
|             2 |                17 |             17 | zx_Entity_Recording_2025_09_27_16_46_13_v108_035  | 紧急变道/猛打方向失稳   |    3.36184  |    3.31255  |               0.0492921 |    4.31917  |    0.507367 | missed_extreme_amplitude;large_event_high_rmse;shown_in_user_screenshot | 极端动作幅值严重低估: true_peak=4.319, v307_peak=0.507；大动作场景整体误差高: v307_rmse=3.362   | v310_v307init_shape_guard_lo              |      3.39925  |    4.05931  |     0.097555 | True           | True           |
|             3 |                20 |             20 | zx_Entity_Recording_2025_09_27_17_14_07_v108_016  | 连续变道/连续左右修正   |    1.59088  |    1.10558  |               0.485296  |    0.08883  |    2.98088  | false_large_maneuver;regression_vs_v300;shown_in_user_screenshot        | 真实近似无大动作但预测大动作: true_peak=0.089, v307_peak=2.981；v307 比 v300 更差: delta=+0.485 | v300_full_joint_h64_no_subject            |      1.10558  |    1.52261  |    24.3741   | True           | False          |
|             3 |                20 |             20 | zx_Entity_Recording_2025_09_27_17_14_07_v108_016  | 连续变道/连续左右修正   |    1.59088  |    1.10558  |               0.485296  |    0.08883  |    2.98088  | false_large_maneuver;regression_vs_v300;shown_in_user_screenshot        | 真实近似无大动作但预测大动作: true_peak=0.089, v307_peak=2.981；v307 比 v300 更差: delta=+0.485 | v307_coarse_scene_init_aux003_film005_h64 |      1.59088  |    2.19111  |    33.5571   | True           | False          |
|             3 |                20 |             20 | zx_Entity_Recording_2025_09_27_17_14_07_v108_016  | 连续变道/连续左右修正   |    1.59088  |    1.10558  |               0.485296  |    0.08883  |    2.98088  | false_large_maneuver;regression_vs_v300;shown_in_user_screenshot        | 真实近似无大动作但预测大动作: true_peak=0.089, v307_peak=2.981；v307 比 v300 更差: delta=+0.485 | v310_v307init_shape_guard_lo              |      1.64219  |    2.2608   |    34.2244   | True           | False          |
|             4 |                23 |             23 | gzj_Entity_Recording_2025_09_27_11_41_47_v108_048 | 平路过弯                |    1.39379  |    1.19885  |               0.194938  |   -0.544015 |    1.88366  | opposite_peak_direction;regression_vs_v300;shown_in_user_screenshot     | 峰值方向相反: true_peak=-0.544, v307_peak=1.884；v307 比 v300 更差: delta=+0.195                | v300_full_joint_h64_no_subject            |      1.19885  |    1.63853  |     2.78084  | False          | False          |
|             4 |                23 |             23 | gzj_Entity_Recording_2025_09_27_11_41_47_v108_048 | 平路过弯                |    1.39379  |    1.19885  |               0.194938  |   -0.544015 |    1.88366  | opposite_peak_direction;regression_vs_v300;shown_in_user_screenshot     | 峰值方向相反: true_peak=-0.544, v307_peak=1.884；v307 比 v300 更差: delta=+0.195                | v307_coarse_scene_init_aux003_film005_h64 |      1.39379  |    1.91765  |     3.46252  | False          | False          |
|             4 |                23 |             23 | gzj_Entity_Recording_2025_09_27_11_41_47_v108_048 | 平路过弯                |    1.39379  |    1.19885  |               0.194938  |   -0.544015 |    1.88366  | opposite_peak_direction;regression_vs_v300;shown_in_user_screenshot     | 峰值方向相反: true_peak=-0.544, v307_peak=1.884；v307 比 v300 更差: delta=+0.195                | v310_v307init_shape_guard_lo              |      1.39209  |    1.90973  |     3.45114  | False          | False          |
|             7 |                14 |             14 | gzj_Entity_Recording_2025_09_27_12_28_14_v108_054 | 平路过弯                |    0.394017 |    0.380515 |               0.0135017 |    0.589398 |   -0.836158 | opposite_peak_direction;shown_in_user_screenshot                        | 峰值方向相反: true_peak=0.589, v307_peak=-0.836                                                 | v300_full_joint_h64_no_subject            |      0.380515 |    0.199729 |     0.868397 | False          | False          |
|             7 |                14 |             14 | gzj_Entity_Recording_2025_09_27_12_28_14_v108_054 | 平路过弯                |    0.394017 |    0.380515 |               0.0135017 |    0.589398 |   -0.836158 | opposite_peak_direction;shown_in_user_screenshot                        | 峰值方向相反: true_peak=0.589, v307_peak=-0.836                                                 | v307_coarse_scene_init_aux003_film005_h64 |      0.394017 |    0.323136 |     1.41866  | False          | False          |
|             7 |                14 |             14 | gzj_Entity_Recording_2025_09_27_12_28_14_v108_054 | 平路过弯                |    0.394017 |    0.380515 |               0.0135017 |    0.589398 |   -0.836158 | opposite_peak_direction;shown_in_user_screenshot                        | 峰值方向相反: true_peak=0.589, v307_peak=-0.836                                                 | v310_v307init_shape_guard_lo              |      0.41058  |    0.341267 |     1.4634   | False          | False          |
|            20 |                19 |             19 | zx_Entity_Recording_2025_09_27_17_45_11_v108_023  | 下坡过弯                |    2.03475  |    2.01918  |               0.0155718 |    3.121    |    1.96837  | large_event_high_rmse;shown_in_user_screenshot                          | 大动作场景整体误差高: v307_rmse=2.035                                                           | v300_full_joint_h64_no_subject            |      2.01918  |    1.56664  |     0.69665  | True           | False          |
|            20 |                19 |             19 | zx_Entity_Recording_2025_09_27_17_45_11_v108_023  | 下坡过弯                |    2.03475  |    2.01918  |               0.0155718 |    3.121    |    1.96837  | large_event_high_rmse;shown_in_user_screenshot                          | 大动作场景整体误差高: v307_rmse=2.035                                                           | v307_coarse_scene_init_aux003_film005_h64 |      2.03475  |    1.57368  |     0.630687 | True           | False          |
|            20 |                19 |             19 | zx_Entity_Recording_2025_09_27_17_45_11_v108_023  | 下坡过弯                |    2.03475  |    2.01918  |               0.0155718 |    3.121    |    1.96837  | large_event_high_rmse;shown_in_user_screenshot                          | 大动作场景整体误差高: v307_rmse=2.035                                                           | v310_v307init_shape_guard_lo              |      2.03431  |    1.5676   |     0.640198 | True           | False          |

## 当前判断

- 如果 v310 在 `v309_severe_all37` 或 `user_screenshot_5` 上下降，但常规 test/all 明显变差，说明 hard-case 约束过强，不能直接作为主线。
- 如果 v310 常规 test/all 基本持平，同时 severe 组改善，才值得作为下一轮主线。
- 如果 severe 组没有改善，说明单纯 loss/权重不足，需要回到事件标签或候选轨迹结构，而不是继续加权。

## guardrail

```json
{
  "pass": true,
  "version": "v310_severe_error_targeted_curve_model_20260704",
  "model_structure_changed": false,
  "loss_changed": true,
  "output_target_unchanged": "21_point_steering_delta_curve",
  "initialized_from_v307_selected": true,
  "v307_reference_model": "v307_coarse_scene_init_aux003_film005_h64",
  "v300_reference_model": "v300_full_joint_h64_no_subject",
  "selected_v310_model": "v310_v307init_shape_guard_lo",
  "uses_coarse_scene_labels_as_features": true,
  "uses_v309_severe_table_for_training": false,
  "uses_v309_severe_table_for_validation_selection": false,
  "uses_v309_severe_table_for_diagnostic_only": true,
  "uses_test_error_as_features": false,
  "candidate_selection_uses_test": false,
  "candidate_selection_uses_validation_only": true,
  "same_event_never_repeated_across_splits": true,
  "event_in_multiple_splits_n": 0,
  "event_without_6_delay_rows_n": 0,
  "event_n": 1167,
  "rolling_sample_n": 7002,
  "v309_severe_candidate_n": 37,
  "selected_passes_v304_noharm_gate": true,
  "selected_val_all_delta_vs_v300": -0.024475123077758898,
  "selected_val_bad10_delta_vs_v300": -0.13671724746624625,
  "selected_test_all_rmse": 0.494997824124735,
  "v307_test_all_rmse": 0.49613793216774177,
  "selected_test_bad10_rmse": 0.7758816666901112,
  "v307_test_bad10_rmse": 0.7777971997857094,
  "selected_v309_severe_all37_rmse": 0.8907046475120493,
  "v307_v309_severe_all37_rmse": 0.8883995288932646,
  "device": "cuda",
  "runtime_seconds": 56.70262265205383,
  "figure_paths": [
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v310_severe_error_targeted_curve_model_20260704\\figures\\v304_training_history.png",
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v310_severe_error_targeted_curve_model_20260704\\figures\\v304_test_delay0_group_rmse.png",
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v310_severe_error_targeted_curve_model_20260704\\figures\\v304_event_aux_macro_f1.png",
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v310_severe_error_targeted_curve_model_20260704\\figures\\v310_v309_severe_group_rmse.png"
  ]
}
```