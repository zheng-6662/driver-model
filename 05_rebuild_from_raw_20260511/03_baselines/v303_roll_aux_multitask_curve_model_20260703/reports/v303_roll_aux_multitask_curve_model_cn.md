# v303 roll-cause 辅助监督多任务曲线模型

## 这一步做了什么

v303 在 v300 的 joint-curve 输出任务上改模型结构：输出仍是 21 点 steering_delta 曲线，但新增 roll-cause summary 编码器，并使用 v301 事件类型作为训练期辅助监督。v301 标签不作为推理输入。

结构上：历史/道路/phase/point token 仍走曲线解码主干；roll-cause summary 走单独 MLP 编码器；该编码一方面输出事件类型辅助头，另一方面通过 token 拼接和 FiLM 调制影响曲线隐藏表示。

## validation-only 选择

| model_name                                | test_used_for_selection   | selected_by                                      |   best_epoch |   best_val_loss |   training_seconds | config_json                                                                                                                                                                                                                                                                                                                                                                                                      |   validation_selection_score |   val_sample_rmse_weighted |   val_tail_rmse_weighted |   val_strong_under_rate_weighted |   val_peak_ratio_weighted |   delay0_val_all_delta_vs_v300 |   delay0_val_bad10_delta_vs_v300 |   delay0_val_bad20_delta_vs_v300 | passes_val_all_noharm_vs_v300   | passes_val_bad10_noharm_vs_v300   | passes_val_bad20_noharm_vs_v300   | passes_v303_noharm_gate   |   validation_rank |
|:------------------------------------------|:--------------------------|:-------------------------------------------------|-------------:|----------------:|-------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------:|---------------------------:|-------------------------:|---------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|---------------------------------:|:--------------------------------|:----------------------------------|:----------------------------------|:--------------------------|------------------:|
| v303_roll_init_aux003_film005_h64         | False                     | validation_original_remaining_plus_delay0_noharm |           14 |        0.469533 |            22.4689 | {"aux_weight": 0.03, "batch_size": 384, "dropout": 0.06, "film_scale": 0.05, "hard_event_extra": 0.0, "hidden_dim": 64, "init_from_v300": true, "lr": 0.0002, "max_epochs": 55, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 112, "n_heads": 4, "n_layers": 3, "patience": 9, "roll_hidden": 128, "smooth_weight": 0.02, "v300_init_model_name": "v300_full_joint_h64_no_subject", "weight_decay": 0.0003}  |                     0.732695 |                   0.464023 |                 0.516624 |                        0.0690642 |                   4.45513 |                    -0.00745204 |                       -0.0630989 |                      -0.033447   | True                            | True                              | True                              | True                      |                 1 |
| v303_roll_init_aux006_film010_hard110_h64 | False                     | validation_original_remaining_plus_delay0_noharm |           21 |        0.553954 |            25.6358 | {"aux_weight": 0.06, "batch_size": 384, "dropout": 0.08, "film_scale": 0.1, "hard_event_extra": 0.1, "hidden_dim": 64, "init_from_v300": true, "lr": 0.0002, "max_epochs": 60, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 128, "n_heads": 4, "n_layers": 3, "patience": 10, "roll_hidden": 160, "smooth_weight": 0.025, "v300_init_model_name": "v300_full_joint_h64_no_subject", "weight_decay": 0.0004} |                     0.745293 |                   0.471659 |                 0.524914 |                        0.0745149 |                   4.44064 |                     0.0038603  |                       -0.0385396 |                      -0.0279216  | True                            | True                              | True                              | True                      |                 2 |
| v303_roll_init_aux005_film010_h64         | False                     | validation_original_remaining_plus_delay0_noharm |            6 |        0.527583 |            13.2367 | {"aux_weight": 0.05, "batch_size": 384, "dropout": 0.08, "film_scale": 0.1, "hard_event_extra": 0.0, "hidden_dim": 64, "init_from_v300": true, "lr": 0.0002, "max_epochs": 60, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 128, "n_heads": 4, "n_layers": 3, "patience": 10, "roll_hidden": 160, "smooth_weight": 0.025, "v300_init_model_name": "v300_full_joint_h64_no_subject", "weight_decay": 0.0004} |                     0.767123 |                   0.484505 |                 0.536827 |                        0.0946991 |                   4.57105 |                     0.0132822  |                       -0.0115534 |                      -0.00233783 | False                           | True                              | True                              | False                     |                 3 |
| v303_roll_scratch_aux010_hard125_h64      | False                     | validation_original_remaining_plus_delay0_noharm |           10 |        0.728464 |            22.2342 | {"aux_weight": 0.1, "batch_size": 320, "dropout": 0.1, "film_scale": 0.2, "hard_event_extra": 0.25, "hidden_dim": 64, "init_from_v300": false, "lr": 0.00055, "max_epochs": 90, "min_lr": 1e-05, "mixer_layers": 2, "mlp_hidden": 128, "n_heads": 4, "n_layers": 3, "patience": 14, "roll_hidden": 160, "smooth_weight": 0.03, "weight_decay": 0.0005}                                                           |                     0.837587 |                   0.526743 |                 0.582263 |                        0.131416  |                   2.67971 |                     0.0732051  |                        0.0425386 |                       0.123898   | False                           | False                             | False                             | False                     |                 4 |

validation 选择出的 v303 候选：`v303_roll_init_aux003_film005_h64`。
v300 参照模型：`v300_full_joint_h64_no_subject`。

## test delay0 对比

| model_name                        | split   | group             |   n |   sample_rmse_mean |   sample_rmse_median |   sample_rmse_p90 |
|:----------------------------------|:--------|:------------------|----:|-------------------:|---------------------:|------------------:|
| v300_full_joint_h64_no_subject    | test    | all               | 232 |           0.519805 |             0.425889 |          0.962683 |
| v300_full_joint_h64_no_subject    | test    | within_bad_top10  |  24 |           0.859987 |             0.690337 |          1.34919  |
| v300_full_joint_h64_no_subject    | test    | within_bad_top20  |  47 |           0.690942 |             0.633268 |          1.20787  |
| v300_full_joint_h64_no_subject    | test    | vehicle_ambiguous | 120 |           0.525913 |             0.450818 |          0.954555 |
| v300_full_joint_h64_no_subject    | test    | strong_steer      | 125 |           0.621347 |             0.518684 |          1.04093  |
| v303_roll_init_aux003_film005_h64 | test    | all               | 232 |           0.513617 |             0.408063 |          0.882823 |
| v303_roll_init_aux003_film005_h64 | test    | within_bad_top10  |  24 |           0.843876 |             0.692444 |          1.45247  |
| v303_roll_init_aux003_film005_h64 | test    | within_bad_top20  |  47 |           0.669646 |             0.596015 |          1.19048  |
| v303_roll_init_aux003_film005_h64 | test    | vehicle_ambiguous | 120 |           0.518756 |             0.435886 |          0.887145 |
| v303_roll_init_aux003_film005_h64 | test    | strong_steer      | 125 |           0.611574 |             0.519966 |          1.04426  |

简表：

- test/all：v300 `0.5198` -> v303 `0.5136`。
- test/within_bad_top10：v300 `0.8600` -> v303 `0.8439`。
- test/within_bad_top20：v300 `0.6909` -> v303 `0.6696`。

## 事件辅助头

| model_name                        | split   | group       |    n |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |
|:----------------------------------|:--------|:------------|-----:|-----------:|--------------------:|-----------:|--------------:|
| v303_roll_init_aux003_film005_h64 | train   | all_rolling | 4212 |   0.716762 |            0.75129  |   0.637138 |      0.72397  |
| v303_roll_init_aux003_film005_h64 | train   | delay0_only |  702 |   0.675214 |            0.649522 |   0.571065 |      0.685731 |
| v303_roll_init_aux003_film005_h64 | val     | all_rolling | 1398 |   0.556509 |            0.391051 |   0.334385 |      0.575096 |
| v303_roll_init_aux003_film005_h64 | val     | delay0_only |  233 |   0.506438 |            0.310276 |   0.276439 |      0.520383 |
| v303_roll_init_aux003_film005_h64 | test    | all_rolling | 1392 |   0.570402 |            0.489049 |   0.395503 |      0.584659 |
| v303_roll_init_aux003_film005_h64 | test    | delay0_only |  232 |   0.538793 |            0.44825  |   0.416327 |      0.548882 |

## 当前判断

- 如果 v303 在事件辅助头上明显好于 v301/v302 的外部分类器，说明 roll-cause 分支确实学到了响应类型信息。
- 是否接受 v303，不看 test 选择，而看 validation no-harm gate 和最终 test 分组报告。
- 如果 test/all 改善但 bad_top10 变差，本轮只能说明结构有方向性，不算达成差样本本质改善。
- 下一步若 v303 bad_top10 仍不够，应在该结构上加入 mixture-of-experts 或不确定性多模态输出，而不是回到删除样本/轻量 residual。

## guardrail

```json
{
  "pass": true,
  "version": "v303_roll_aux_multitask_curve_model_20260703",
  "model_structure_changed": true,
  "output_target_unchanged": "21_point_steering_delta_curve",
  "uses_roll_cause_summary_as_input": true,
  "uses_future_event_labels_as_features": false,
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
  "selected_v303_model": "v303_roll_init_aux003_film005_h64",
  "selected_passes_v303_noharm_gate": true,
  "selected_val_all_delta_vs_v300": -0.007452043468860081,
  "selected_val_bad10_delta_vs_v300": -0.06309892609715462,
  "selected_val_bad20_delta_vs_v300": -0.033446998830805375,
  "device": "cuda",
  "runtime_seconds": 103.34858751296997,
  "figure_paths": [
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v303_roll_aux_multitask_curve_model_20260703\\figures\\v303_training_history.png",
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v303_roll_aux_multitask_curve_model_20260703\\figures\\v303_test_delay0_group_rmse.png",
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v303_roll_aux_multitask_curve_model_20260703\\figures\\v303_event_aux_macro_f1.png"
  ]
}
```