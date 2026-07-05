# v246 oracle 最佳锚点遍历与 input-only selector 审查

## 结论先说

- oracle 遍历确认：test bad_top10 如果每个样本都选真实误差最小的更晚锚点，平均 RMSE 从 `1.008` 降到 `0.656`，delta=`-0.352`，改善率 `84.7%`。
- early bad_top10 的 oracle 上限更强：平均 delta=`-0.431`，改善率 `95.8%`，最常见 oracle shift=`+600ms`。
- 只用 base 锚点可见输入训练的 RF selector，在 test bad_top10 上把 RMSE 从 `1.008` 降到 `0.908`，delta=`-0.100`，改善率 `29.7%`；但只捕获了 oracle 收益的一部分，mean gain capture=`19.6%`。
- RF selector 在 early bad_top10 上：delta=`-0.153`，改善率 `45.1%`，oracle delay 命中率 `8.5%`。
- 显式固定策略 `policy_wait_to_latest_anchor` 在 test bad_top10 上 RMSE=`0.685`，delta=`-0.322`，说明 Ridge 的强表现主要可能来自“尽量等到最晚锚点”，不是已经学会逐样本精确找最佳锚点。
- 对 early bad_top10，固定等到最晚锚点的 delta=`-0.391`，接近 oracle delta=`-0.431`；这支持“变化太晚才显现的样本应该多看一点”的判断。
- 因此，这条路线有明确上限收益；但真正难点是 selector 能否仅凭锚点前输入判断该等多久。下一步不应该直接把 oracle 最佳锚点写进测试流程，而应把 selector/风险等待策略作为模型组件验证。

## Raw Table 1：oracle 上限（test）

| base_group                   |   n_base_samples |   mean_base_error_v241 |   mean_oracle_error_v241 |   mean_delta_oracle_minus_base |   oracle_improve_rate |   mean_oracle_shift_ms |   most_common_oracle_shift_ms |   most_common_oracle_delay_ms |
|:-----------------------------|-----------------:|-----------------------:|-------------------------:|-------------------------------:|----------------------:|-----------------------:|------------------------------:|------------------------------:|
| all                          |             1104 |                  0.393 |                    0.277 |                         -0.116 |                 0.687 |                362.681 |                             0 |                          1000 |
| bad_top10                    |              111 |                  1.008 |                    0.656 |                         -0.352 |                 0.847 |                499.099 |                           600 |                          1000 |
| early_bad_top10_delay_le_400 |               71 |                  1.021 |                    0.591 |                         -0.431 |                 0.958 |                667.606 |                           600 |                          1000 |
| very_bad_top5                |               56 |                  1.168 |                    0.755 |                         -0.413 |                 0.857 |                500.000 |                           400 |                          1000 |

## Raw Table 2：selector 候选误差拟合质量

| selector_name             | split   |   n_candidate_rows |   candidate_error_rmse |   candidate_error_mae |
|:--------------------------|:--------|-------------------:|-----------------------:|----------------------:|
| selector_rf_base_input    | test    |               3864 |                  0.349 |                 0.251 |
| selector_rf_base_input    | train   |              14154 |                  0.047 |                 0.037 |
| selector_rf_base_input    | val     |               6489 |                  0.672 |                 0.458 |
| selector_ridge_base_input | test    |               3864 |                  0.348 |                 0.250 |
| selector_ridge_base_input | train   |              14154 |                  0.056 |                 0.044 |
| selector_ridge_base_input | val     |               6489 |                  0.667 |                 0.456 |

## Raw Table 3：RF selector 实际选锚点效果（test）

| base_group                   |   n_base_samples |   mean_base_error_v241 |   mean_selected_error_v241 |   mean_oracle_error_v241 |   mean_delta_selected_minus_base |   mean_delta_oracle_minus_base |   selected_improve_rate |   mean_gain_capture_rate |   selected_matches_oracle_delay_rate |   mean_selected_shift_ms |   mean_oracle_shift_ms |   most_common_selected_shift_ms |   most_common_oracle_shift_ms |
|:-----------------------------|-----------------:|-----------------------:|---------------------------:|-------------------------:|---------------------------------:|-------------------------------:|------------------------:|-------------------------:|-------------------------------------:|-------------------------:|-----------------------:|--------------------------------:|------------------------------:|
| all                          |             1104 |                  0.393 |                      0.372 |                    0.277 |                           -0.021 |                         -0.116 |                   0.214 |                   -0.026 |                                0.313 |                  121.558 |                362.681 |                               0 |                             0 |
| bad_top10                    |              111 |                  1.008 |                      0.908 |                    0.656 |                           -0.100 |                         -0.352 |                   0.297 |                    0.196 |                                0.180 |                  147.748 |                499.099 |                               0 |                           600 |
| early_bad_top10_delay_le_400 |               71 |                  1.021 |                      0.868 |                    0.591 |                           -0.153 |                         -0.431 |                   0.451 |                    0.256 |                                0.085 |                  225.352 |                667.606 |                               0 |                           600 |
| very_bad_top5                |               56 |                  1.168 |                      1.027 |                    0.755 |                           -0.141 |                         -0.413 |                   0.393 |                    0.236 |                                0.179 |                  207.143 |                500.000 |                               0 |                           400 |

## Raw Table 4：Ridge selector 参考（test）

| base_group                   |   n_base_samples |   mean_base_error_v241 |   mean_selected_error_v241 |   mean_oracle_error_v241 |   mean_delta_selected_minus_base |   mean_delta_oracle_minus_base |   selected_improve_rate |   mean_gain_capture_rate |   selected_matches_oracle_delay_rate |   mean_selected_shift_ms |   mean_oracle_shift_ms |   most_common_selected_shift_ms |   most_common_oracle_shift_ms |
|:-----------------------------|-----------------:|-----------------------:|---------------------------:|-------------------------:|---------------------------------:|-------------------------------:|------------------------:|-------------------------:|-------------------------------------:|-------------------------:|-----------------------:|--------------------------------:|------------------------------:|
| all                          |             1104 |                  0.393 |                      0.305 |                    0.277 |                           -0.088 |                         -0.116 |                   0.616 |                    0.030 |                                0.636 |                  500.000 |                362.681 |                               0 |                             0 |
| bad_top10                    |              111 |                  1.008 |                      0.685 |                    0.656 |                           -0.322 |                         -0.352 |                   0.820 |                    0.800 |                                0.721 |                  603.604 |                499.099 |                            1000 |                           600 |
| early_bad_top10_delay_le_400 |               71 |                  1.021 |                      0.630 |                    0.591 |                           -0.391 |                         -0.431 |                   0.930 |                    0.817 |                                0.662 |                  808.451 |                667.606 |                            1000 |                           600 |
| very_bad_top5                |               56 |                  1.168 |                      0.789 |                    0.755 |                           -0.378 |                         -0.413 |                   0.839 |                    0.790 |                                0.679 |                  625.000 |                500.000 |                            1000 |                           400 |

## Raw Table 5：固定等到最晚锚点策略（test）

| base_group                   |   n_base_samples |   mean_base_error_v241 |   mean_selected_error_v241 |   mean_oracle_error_v241 |   mean_delta_selected_minus_base |   mean_delta_oracle_minus_base |   selected_improve_rate |   mean_gain_capture_rate |   selected_matches_oracle_delay_rate |   mean_selected_shift_ms |   mean_oracle_shift_ms |   most_common_selected_shift_ms |   most_common_oracle_shift_ms |
|:-----------------------------|-----------------:|-----------------------:|---------------------------:|-------------------------:|---------------------------------:|-------------------------------:|------------------------:|-------------------------:|-------------------------------------:|-------------------------:|-----------------------:|--------------------------------:|------------------------------:|
| all                          |             1104 |                  0.393 |                      0.305 |                    0.277 |                           -0.088 |                         -0.116 |                   0.616 |                    0.030 |                                0.636 |                  500.000 |                362.681 |                               0 |                             0 |
| bad_top10                    |              111 |                  1.008 |                      0.685 |                    0.656 |                           -0.322 |                         -0.352 |                   0.820 |                    0.800 |                                0.721 |                  603.604 |                499.099 |                            1000 |                           600 |
| early_bad_top10_delay_le_400 |               71 |                  1.021 |                      0.630 |                    0.591 |                           -0.391 |                         -0.431 |                   0.930 |                    0.817 |                                0.662 |                  808.451 |                667.606 |                            1000 |                           600 |
| very_bad_top5                |               56 |                  1.168 |                      0.789 |                    0.755 |                           -0.378 |                         -0.413 |                   0.839 |                    0.790 |                                0.679 |                  625.000 |                500.000 |                            1000 |                           400 |

## 解释

1. oracle_best_anchor 是用真实误差选出来的最佳锚点，只能作为理论上限，不能部署。
2. input-only selector 没有使用未来真实曲线、人工响应标签、event_uid 或 recording；它只看 base 锚点可见的历史/道路/phase 特征和候选等待时长。
3. 如果 selector 能稳定接近 oracle，说明“每个样本自适应锚点”可以进入正式训练任务；如果 selector 只能捕获一小部分收益，就需要先做更强的风险/不确定性判定。
4. Ridge selector 和固定等到最晚锚点很接近，说明当前收益里有相当一部分来自“多看一些时间”这个简单机制；后续要加入等待代价或触发条件，否则模型可能退化成一律晚预测。

## 产物

- `tables/v246_sample_tail_errors.csv`
- `tables/v246_base_input_features.csv`
- `tables/v246_anchor_candidate_table.csv`
- `tables/v246_oracle_best_anchor_by_base_sample.csv`
- `tables/v246_oracle_best_anchor_summary.csv`
- `tables/v246_selector_candidate_error_fit_metrics.csv`
- `tables/v246_selector_predictions_by_candidate.csv`
- `tables/v246_selector_selected_anchor_by_base_sample.csv`
- `tables/v246_policy_selected_anchor_by_base_sample.csv`
- `tables/v246_selector_policy_summary.csv`
- `tables/v246_anchor_shift_distribution.csv`
- `figures/v246_test_oracle_vs_selector_error.png`
- `figures/v246_test_bad_top10_shift_distribution.png`
- ZIP：`v246_oracle_best_anchor_and_selector_audit_pack.zip`

![v246_test_oracle_vs_selector_error](../figures/v246_test_oracle_vs_selector_error.png)

![v246_test_bad_top10_shift_distribution](../figures/v246_test_bad_top10_shift_distribution.png)
