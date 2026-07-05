# v249 shape-aware curve model 报告

## 本轮边界

- 从 v241 checkpoint 初始化，继承 TCN + multi-head query attention backbone。
- 继续使用 `original_remaining` masked target。
- 不做 anchor selector、gate/router、response-type hard routing，不删除样本。
- 只用 validation 选择候选；test 只做 locked report。

## Validation 选择

- `v249c_shape_conditioned_residual`: score=0.8759, noharm_vs_v236=True, upgrade_vs_v241=True, shape_gain=False, accepted=False
- `v249a_shape_loss_only`: score=0.8794, noharm_vs_v236=True, upgrade_vs_v241=False, shape_gain=False, accepted=False
- `v249b_shape_aux_heads`: score=0.8847, noharm_vs_v236=True, upgrade_vs_v241=True, shape_gain=False, accepted=False

当前 best diagnostic model：`v249c_shape_conditioned_residual`。

## Test 对照摘要

| bucket             |   delay_ms |   steer_tail_rmse_mean__v241_tcn_mha_h96 |   steer_tail_rmse_mean__v249c_shape_conditioned_residual |   delta_steer_tail_rmse_mean__v249c_shape_conditioned_residual_minus_v241 |   peak_ratio_mean__v241_tcn_mha_h96 |   peak_ratio_mean__v249c_shape_conditioned_residual |
|:-------------------|-----------:|-----------------------------------------:|---------------------------------------------------------:|--------------------------------------------------------------------------:|------------------------------------:|----------------------------------------------------:|
| all                |          0 |                                 0.475053 |                                                 0.475072 |                                                               1.85668e-05 |                            1.28796  |                                            1.19037  |
| all                |        200 |                                 0.43535  |                                                 0.432157 |                                                              -0.00319365  |                            1.36296  |                                            1.26028  |
| all                |        400 |                                 0.410969 |                                                 0.407297 |                                                              -0.00367233  |                            1.26127  |                                            1.17144  |
| all                |        600 |                                 0.38532  |                                                 0.381122 |                                                              -0.00419876  |                            1.34647  |                                            1.23383  |
| all                |        800 |                                 0.346069 |                                                 0.337291 |                                                              -0.00877818  |                            1.54065  |                                            1.37545  |
| all                |       1000 |                                 0.304616 |                                                 0.296613 |                                                              -0.00800303  |                            1.81466  |                                            1.58415  |
| normal_predictable |          0 |                                 0.371164 |                                                 0.352075 |                                                              -0.019089    |                            1.68307  |                                            1.54203  |
| normal_predictable |        200 |                                 0.354169 |                                                 0.333372 |                                                              -0.0207969   |                            1.80574  |                                            1.64708  |
| normal_predictable |        400 |                                 0.326977 |                                                 0.312519 |                                                              -0.0144575   |                            1.60102  |                                            1.4605   |
| normal_predictable |        600 |                                 0.297837 |                                                 0.281225 |                                                              -0.016612    |                            1.76699  |                                            1.58115  |
| normal_predictable |        800 |                                 0.255496 |                                                 0.239161 |                                                              -0.0163344   |                            2.12654  |                                            1.85844  |
| normal_predictable |       1000 |                                 0.220595 |                                                 0.204779 |                                                              -0.0158159   |                            2.65918  |                                            2.26742  |
| observe_later_like |          0 |                                 0.792468 |                                                 0.816408 |                                                               0.0239394   |                            0.780708 |                                            0.731922 |
| observe_later_like |        200 |                                 0.661233 |                                                 0.694114 |                                                               0.0328817   |                            0.791186 |                                            0.749513 |
| observe_later_like |        400 |                                 0.620467 |                                                 0.644352 |                                                               0.0238845   |                            0.816311 |                                            0.782085 |
| observe_later_like |        600 |                                 0.588433 |                                                 0.609697 |                                                               0.0212644   |                            0.812566 |                                            0.781839 |
| observe_later_like |        800 |                                 0.554005 |                                                 0.562915 |                                                               0.0089103   |                            0.814285 |                                            0.778793 |
| observe_later_like |       1000 |                                 0.50421  |                                                 0.500727 |                                                              -0.0034833   |                            0.787938 |                                            0.768184 |
| strong_steer       |          0 |                                 0.590904 |                                                 0.617785 |                                                               0.0268813   |                            0.8076   |                                            0.766102 |
| strong_steer       |        200 |                                 0.53321  |                                                 0.552859 |                                                               0.0196496   |                            0.836541 |                                            0.804851 |
| strong_steer       |        400 |                                 0.512747 |                                                 0.522371 |                                                               0.00962365  |                            0.860344 |                                            0.834553 |
| strong_steer       |        600 |                                 0.494278 |                                                 0.504799 |                                                               0.0105213   |                            0.855367 |                                            0.831122 |
| strong_steer       |        800 |                                 0.458156 |                                                 0.456672 |                                                              -0.00148353  |                            0.861055 |                                            0.812771 |
| strong_steer       |       1000 |                                 0.405783 |                                                 0.405981 |                                                               0.000198692 |                            0.816977 |                                            0.766206 |

## Shape 指标摘要

| event_group        |    n |   mean_rmse |   mean_range_ratio |   mean_slope_ratio |   mean_delta_rmse |   mean_delta_range |   mean_delta_slope |
|:-------------------|-----:|------------:|-------------------:|-------------------:|------------------:|-------------------:|-------------------:|
| all                | 1104 |    0.34736  |           1.15231  |           0.755536 |       -0.00406346 |          -0.196419 |         -0.137388  |
| bad_top10_v241     |  111 |    0.844454 |           0.625267 |           0.535165 |       -0.0167717  |          -0.056194 |         -0.0857816 |
| normal             |  594 |    0.257699 |           1.51171  |           0.885004 |       -0.0141313  |          -0.3149   |         -0.185869  |
| observe_later_like |  162 |    0.562459 |           0.688808 |           0.538623 |        0.0121732  |          -0.056898 |         -0.0766259 |
| strong_steer       |  480 |    0.455677 |           0.725442 |           0.597453 |        0.00815607 |          -0.056487 |         -0.0800585 |

## 输入近邻不可判别审查

- test delay=0 v241 bad_top10 共审查 `19` 个样本，其中 `19` 个被标记为 `input_ambiguous`。
| event_uid                                         |   v241_rmse |   candidate_rmse |   neighbor_future_pairwise_rmse_mean |   neighbor_peak_abs_std |   query_vs_neighbor_best_rmse | ambiguity_category   |
|:--------------------------------------------------|------------:|-----------------:|-------------------------------------:|------------------------:|------------------------------:|:---------------------|
| rjy_Entity_Recording_2025_09_28_19_51_44_v108_039 |    1.60922  |         1.50878  |                             1.29375  |                0.495848 |                      0.743941 | input_ambiguous      |
| tyy_Entity_Recording_2025_09_28_14_23_43_v108_002 |    1.19596  |         1.1874   |                             0.644581 |                0.481747 |                      0.709601 | input_ambiguous      |
| rjy_Entity_Recording_2025_09_28_20_02_20_v108_031 |    1.12175  |         1.20837  |                             0.76675  |                0.53083  |                      0.504951 | input_ambiguous      |
| rjy_Entity_Recording_2025_09_28_19_51_44_v108_023 |    1.05469  |         1.05305  |                             1.02886  |                0.401739 |                      0.909768 | input_ambiguous      |
| rjy_Entity_Recording_2025_09_28_20_15_42_v108_022 |    0.969431 |         0.850446 |                             1.38602  |                0.728768 |                      0.622803 | input_ambiguous      |
| rjy_Entity_Recording_2025_09_28_19_51_44_v108_016 |    0.966272 |         1.06087  |                             0.842863 |                0.592405 |                      0.123054 | input_ambiguous      |
| rjy_Entity_Recording_2025_09_28_19_51_44_v108_014 |    0.944971 |         0.919663 |                             0.886585 |                0.303622 |                      0.522838 | input_ambiguous      |
| cwh_Entity_Recording_2025_09_26_20_06_19_v108_002 |    0.940967 |         0.966427 |                             0.907819 |                0.317937 |                      0.667317 | input_ambiguous      |
| tyy_Entity_Recording_2025_09_28_14_23_43_v108_038 |    0.894445 |         0.889814 |                             0.885682 |                0.440591 |                      0.253368 | input_ambiguous      |
| rjy_Entity_Recording_2025_09_28_20_02_20_v108_014 |    0.821724 |         0.876036 |                             1.62647  |                0.540498 |                      0.650805 | input_ambiguous      |
| lx_Entity_Recording_2025_09_26_09_17_22_v108_034  |    0.808225 |         0.775454 |                             0.715115 |                0.491317 |                      0.372238 | input_ambiguous      |
| rjy_Entity_Recording_2025_09_28_20_02_20_v108_040 |    0.78039  |         0.861856 |                             0.814948 |                0.469969 |                      1.57642  | input_ambiguous      |

## 下一步决策

| decision_item                        | decision                                      | reason                                                                                              |
|:-------------------------------------|:----------------------------------------------|:----------------------------------------------------------------------------------------------------|
| best_diagnostic_shape_model          | v249c_shape_conditioned_residual              | Best by validation selection score; not automatically a formal replacement.                         |
| accept_shape_model_as_next_candidate | False                                         | No v249 candidate passed validation no-harm + v241-upgrade + shape-gain checks.                     |
| accepted_model_name                  |                                               | Empty means v249 remains diagnostic only.                                                           |
| formal_replacement_allowed           | False                                         | v249 is a shape-aware diagnostic experiment; formal claim needs locked audit and robustness checks. |
| recommended_next_task                | v249_error_review_or_input_ambiguity_followup | Do not use test to retune; use ambiguity audit and validation evidence to decide next bounded step. |

## 关键图

- `figures\v249_shape_casebook_test_hard.png`
- `figures\v249_tail_delta_by_bucket.png`

## 关键产物

- `tables/v249_model_selection_validation_shape.csv`
- `tables/v249_metrics_by_delay_and_bucket.csv`
- `tables/v249_compare_vs_v241_original_remaining.csv`
- `tables/v249_shape_summary.csv`
- `tables/v249_per_sample_shape_delta_vs_v241.csv`
- `tables/v249_input_neighborhood_ambiguity_audit.csv`
- `figures/v249_shape_casebook_test_hard.png`
- `figures/v249_tail_delta_by_bucket.png`