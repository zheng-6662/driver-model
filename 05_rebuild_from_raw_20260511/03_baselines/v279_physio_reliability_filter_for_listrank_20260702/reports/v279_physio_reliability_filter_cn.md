# v279 生理可靠性过滤实验

## 目的

v278 已经说明：如果只看 test diagnostic，vehicle-only listwise 候选在 bad_top10 上有一点空间，但验证集阈值选不出可部署提升；加入生理直接排序也没有赢过 vehicle-only。

v279 换一个任务构造：生理不直接选轨迹，而是判断 v278 车辆候选是否可信。也就是先让车辆模型给出“一个看起来值得替换 latest 的候选”，再让可靠性模型决定是否真的覆盖 latest。

## 方法

- 候选来源：复现 v278 的 `listrank_vehicle`，每个事件只保留第一候选。
- 监督目标：`actual_gain_vs_latest = latest_tail_rmse_v241 - candidate_tail_rmse_v241`。
- 二级模型：HGB 回归器预测收益，HGB 分类器预测替换是否为正收益。
- 得分形式：纯收益分数、风险校正收益分数、正收益概率分数。
- 阈值选择：仍只用 validation，test 只报告。
- 对照组：保留 `v278_vehicle_rank_score_only`，确认这一版口径能复现 v278 的车辆排序筛选。

## 核心结论

- fixed wait-latest test bad_top10: `0.695048`
- val 选择的最好可部署 test bad_top10: `0.695048`
- test diagnostic 最好 bad_top10: `0.679116`
- 车辆可靠性最好 diagnostic: `0.679116`
- 生理可靠性最好 diagnostic: `0.679116`
- 生理是否赢过车辆可靠性: `False`
- 可部署规则是否超过 fixed latest: `False`
- diagnostic 是否超过 fixed latest: `True`

## v278 第一候选真实收益分布

| split   |   event_n |   actual_gain_mean |   actual_gain_median |   actual_good_rate |   bad_top10_n |
|:--------|----------:|-------------------:|---------------------:|-------------------:|--------------:|
| test    |       184 |         -0.0565261 |           -0.0023231 |           0.255435 |            19 |
| train   |       674 |          0.032312  |            0.0181504 |           0.587537 |            68 |
| val     |       309 |         -0.112153  |           -0.0415694 |           0.245955 |            31 |

## 决策汇总

| source               | label                                                                |     rmse | deployable   |   override_rate |   val_bad_delta |   val_all_delta |   stable_pass |   delta_vs_fixed_latest | passes_fixed_latest   |
|:---------------------|:---------------------------------------------------------------------|---------:|:-------------|----------------:|----------------:|----------------:|--------------:|------------------------:|:----------------------|
| baseline             | policy_wait_to_latest_anchor                                         | 0.695048 | True         |      nan        |     nan         |     nan         |           nan |             4.15347e-07 | False                 |
| oracle               | oracle_best_anchor_upper_bound                                       | 0.612475 | False        |      nan        |     nan         |     nan         |           nan |            -0.0825726   | True                  |
| best_any             | reliability_vehicle_bio_pair_gain threshold=inf                      | 0.695048 | True         |        0        |       0         |       0         |             0 |             4.15347e-07 | False                 |
| best_active          | v278_vehicle_rank_score_only threshold=1.518402327900305             | 0.695048 | True         |        0        |       0         |       0         |             1 |             4.15347e-07 | False                 |
| best_stable_active   | v278_vehicle_rank_score_only threshold=1.518402327900305             | 0.695048 | True         |        0        |       0         |       0         |             1 |             4.15347e-07 | False                 |
| best_noharm_all      | v278_vehicle_rank_score_only threshold=1.518402327900305             | 0.695048 | True         |        0        |       0         |       0         |             1 |             4.15347e-07 | False                 |
| test_best_diagnostic | reliability_vehicle_bio_state_prob_good threshold=0.4991992890887337 | 0.679116 | False        |        0.105263 |       0.0431756 |       0.0111631 |             0 |            -0.0159321   | True                  |

## 可靠性特征组

| feature_set                         | model_kind                          |   feature_n |   train_rows |   val_rows |   val_gain_mae |   val_gain_corr |   val_good_rate_actual |   val_good_prob_mean |
|:------------------------------------|:------------------------------------|------------:|-------------:|-----------:|---------------:|----------------:|-----------------------:|---------------------:|
| v278_vehicle_rank_score_only        | baseline_score                      |           1 |          674 |        309 |     nan        |     nan         |             nan        |           nan        |
| reliability_vehicle                 | gain_regressor_plus_good_classifier |          11 |          674 |        309 |       0.213521 |      -0.0747245 |               0.245955 |             0.77437  |
| reliability_vehicle_bio_pair        | gain_regressor_plus_good_classifier |          24 |          674 |        309 |       0.212214 |      -0.0233669 |               0.245955 |             0.765154 |
| reliability_vehicle_bio_state       | gain_regressor_plus_good_classifier |         120 |          674 |        309 |       0.212531 |      -0.0726588 |               0.245955 |             0.803471 |
| reliability_vehicle_style_bio_state | gain_regressor_plus_good_classifier |         217 |          674 |        309 |       0.211967 |      -0.0573761 |               0.245955 |             0.803263 |

## val 口径排名前 18

| feature_set                                       |   threshold |   val_bad_top10_selected_rmse |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   val_bad_top10_override_rate |   test_bad_top10_selected_rmse |   test_bad_top10_override_rate |   selection_score |
|:--------------------------------------------------|------------:|------------------------------:|--------------------------------:|--------------------------:|------------------------------:|-------------------------------:|-------------------------------:|------------------:|
| v278_vehicle_rank_score_only                      |   1.5184    |                       1.07279 |                               0 |               0           |                     0.0322581 |                       0.695048 |                              0 |        0          |
| v278_vehicle_rank_score_only                      | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_gain                          |   0.0900396 |                       1.07279 |                               0 |              -4.0785e-05  |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_gain                          | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_risk_adjusted                 |   0.214769  |                       1.07279 |                               0 |              -4.0785e-05  |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_risk_adjusted                 | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_prob_good                     | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_bio_pair_gain                 | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_bio_pair_risk_adjusted        | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_bio_pair_prob_good            | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_bio_state_gain                | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_bio_state_risk_adjusted       | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_bio_state_prob_good           | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_style_bio_state_gain          | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_style_bio_state_risk_adjusted | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_style_bio_state_prob_good     | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |        0          |
| reliability_vehicle_bio_state_gain                |   0.076023  |                       1.07279 |                               0 |               0.000169221 |                     0         |                       0.695048 |                              0 |        0.00182588 |
| reliability_vehicle_bio_state_risk_adjusted       |   0.200921  |                       1.07279 |                               0 |               0.000169221 |                     0         |                       0.695048 |                              0 |        0.00182588 |

## test diagnostic 排名前 18

| feature_set                                       |   threshold |   val_bad_top10_selected_rmse |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate |
|:--------------------------------------------------|------------:|------------------------------:|--------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|
| reliability_vehicle_bio_state_prob_good           |   0.499199  |                       1.11596 |                       0.0431756 |                0.0111631  |                       0.679116 |                       -0.0159326 |                       0.105263 |
| reliability_vehicle_bio_state_prob_good           |   0.499098  |                       1.11596 |                       0.0431756 |                0.0145863  |                       0.679116 |                       -0.0159326 |                       0.105263 |
| reliability_vehicle_gain                          |   0.0649569 |                       1.13638 |                       0.0635876 |                0.00543952 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_gain                          |   0.0679082 |                       1.10232 |                       0.0295291 |                0.0014159  |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_risk_adjusted                 |   0.196021  |                       1.10232 |                       0.0295291 |                0.00132361 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_risk_adjusted                 |   0.192625  |                       1.10232 |                       0.0295291 |                0.0014159  |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_gain                          |   0.0629265 |                       1.13638 |                       0.0635876 |                0.00716249 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_risk_adjusted                 |   0.186564  |                       1.13638 |                       0.0635876 |                0.00918886 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_risk_adjusted                 |   0.189695  |                       1.13638 |                       0.0635876 |                0.00543952 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_style_bio_state_risk_adjusted |   0.185817  |                       1.10792 |                       0.0351326 |                0.00734834 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_gain                          |   0.0713521 |                       1.10232 |                       0.0295291 |                0.00132361 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_gain                          |   0.0758609 |                       1.10232 |                       0.0295291 |                0.00150198 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_style_bio_state_risk_adjusted |   0.182133  |                       1.10792 |                       0.0351326 |                0.00546659 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_bio_state_risk_adjusted       |   0.184235  |                       1.11594 |                       0.0431497 |                0.00640149 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_bio_state_risk_adjusted       |   0.18054   |                       1.11594 |                       0.0431497 |                0.00898718 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_bio_state_risk_adjusted       |   0.177896  |                       1.11594 |                       0.0431497 |                0.00868722 |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_bio_state_gain                |   0.0536134 |                       1.11594 |                       0.0431497 |                0.0104145  |                       0.680631 |                       -0.0144176 |                       0.105263 |
| reliability_vehicle_bio_state_gain                |   0.0621879 |                       1.10792 |                       0.0351326 |                0.00721878 |                       0.680631 |                       -0.0144176 |                       0.105263 |

## 产物

- `figures\v279_test_badtop10_physio_reliability_filter.png`
- `tables/v279_vehicle_listrank_top_candidate_rich.csv`
- `tables/v279_reliability_feature_set_audit.csv`
- `tables/v279_reliability_predictions.csv`
- `tables/v279_score_top_candidates.csv`
- `tables/v279_threshold_search.csv`
- `tables/v279_selected_by_strategy.csv`
- `tables/v279_chosen_configs.csv`
- `tables/v279_decision_summary.csv`
- `logs/guardrail_check.json`
