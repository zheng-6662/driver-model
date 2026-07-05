# v280 cross-fit 生理可靠性过滤实验

## 目的

v279 发现二级可靠性模型存在一个关键偏差：train top 候选由 in-sample ranker 产生，真实正收益率明显高于 val/test，导致可靠性模型在验证集上过度乐观。

v280 修正这个偏差：train split 的 top 候选由 recording 分组的 OOF listwise ranker 产生，val/test 由 full-train ranker 产生。其余可靠性模型和阈值搜索口径保持不变。

## 方法

- 候选来源：v278 `listrank_vehicle`，每个事件只保留第一候选。
- train top：recording 分组 OOF ranker，避免 in-sample top 假好。
- val/test top：full-train ranker，模拟部署时可用模型。
- 监督目标：`actual_gain_vs_latest = latest_tail_rmse_v241 - candidate_tail_rmse_v241`。
- 二级模型：HGB 回归器预测收益，HGB 分类器预测替换是否为正收益。
- 得分形式：纯收益分数、风险校正收益分数、正收益概率分数。
- 阈值选择：仍只用 validation，test 只报告。
- 对照组：保留 `v280_crossfit_rank_score_only`，检验只用 rank score 的阈值效果。

## 核心结论

- fixed wait-latest test bad_top10: `0.695048`
- val 选择的最好可部署 test bad_top10: `0.695048`
- test diagnostic 最好 bad_top10: `0.689064`
- 车辆可靠性最好 diagnostic: `0.689889`
- 生理可靠性最好 diagnostic: `0.689889`
- 生理是否赢过车辆可靠性: `False`
- 可部署规则是否超过 fixed latest: `False`
- diagnostic 是否超过 fixed latest: `True`

## v278 第一候选真实收益分布

| split   |   event_n |   actual_gain_mean |   actual_gain_median |   actual_good_rate |   bad_top10_n |
|:--------|----------:|-------------------:|---------------------:|-------------------:|--------------:|
| test    |       184 |         -0.0620046 |          -0.00303589 |           0.255435 |            19 |
| train   |       674 |          0.0309605 |           0.0162136  |           0.590504 |            68 |
| val     |       309 |         -0.12223   |          -0.0426034  |           0.223301 |            31 |

## 决策汇总

| source               | label                                                      |     rmse | deployable   |   override_rate |   val_bad_delta |   val_all_delta |   stable_pass |   delta_vs_fixed_latest | passes_fixed_latest   |
|:---------------------|:-----------------------------------------------------------|---------:|:-------------|----------------:|----------------:|----------------:|--------------:|------------------------:|:----------------------|
| baseline             | policy_wait_to_latest_anchor                               | 0.695048 | True         |      nan        |     nan         |    nan          |           nan |             4.15347e-07 | False                 |
| oracle               | oracle_best_anchor_upper_bound                             | 0.612475 | False        |      nan        |     nan         |    nan          |           nan |            -0.0825726   | True                  |
| best_any             | reliability_vehicle_bio_pair_gain threshold=inf            | 0.695048 | True         |        0        |       0         |      0          |             0 |             4.15347e-07 | False                 |
| best_active          | v280_crossfit_rank_score_only threshold=1.5767133524365269 | 0.695048 | True         |        0        |       0         |      0          |             1 |             4.15347e-07 | False                 |
| best_stable_active   | v280_crossfit_rank_score_only threshold=1.5767133524365269 | 0.695048 | True         |        0        |       0         |      0          |             1 |             4.15347e-07 | False                 |
| best_noharm_all      | v280_crossfit_rank_score_only threshold=1.5767133524365269 | 0.695048 | True         |        0        |       0         |      0          |             1 |             4.15347e-07 | False                 |
| test_best_diagnostic | v280_crossfit_rank_score_only threshold=0.7101544552805573 | 0.689064 | False        |        0.105263 |       0.0403587 |      0.00120608 |             0 |            -0.00598449  | True                  |

## 可靠性特征组

| feature_set                         | model_kind                          |   feature_n |   train_rows |   val_rows |   val_gain_mae |   val_gain_corr |   val_good_rate_actual |   val_good_prob_mean |
|:------------------------------------|:------------------------------------|------------:|-------------:|-----------:|---------------:|----------------:|-----------------------:|---------------------:|
| v280_crossfit_rank_score_only       | baseline_score                      |           1 |          674 |        309 |     nan        |     nan         |             nan        |           nan        |
| reliability_vehicle                 | gain_regressor_plus_good_classifier |          11 |          674 |        309 |       0.21694  |       0.0322539 |               0.223301 |             0.767284 |
| reliability_vehicle_bio_pair        | gain_regressor_plus_good_classifier |          24 |          674 |        309 |       0.217652 |      -0.0193892 |               0.223301 |             0.765037 |
| reliability_vehicle_bio_state       | gain_regressor_plus_good_classifier |         120 |          674 |        309 |       0.217283 |      -0.0644959 |               0.223301 |             0.785664 |
| reliability_vehicle_style_bio_state | gain_regressor_plus_good_classifier |         217 |          674 |        309 |       0.217458 |      -0.0931786 |               0.223301 |             0.791956 |

## val 口径排名前 18

| feature_set                                       |   threshold |   val_bad_top10_selected_rmse |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   val_bad_top10_override_rate |   test_bad_top10_selected_rmse |   test_bad_top10_override_rate |   selection_score |
|:--------------------------------------------------|------------:|------------------------------:|--------------------------------:|--------------------------:|------------------------------:|-------------------------------:|-------------------------------:|------------------:|
| v280_crossfit_rank_score_only                     |   1.57671   |                       1.07279 |                               0 |               0           |                     0.0322581 |                       0.695048 |                              0 |       0           |
| v280_crossfit_rank_score_only                     | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_gain                          | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_risk_adjusted                 | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_prob_good                     | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_bio_pair_gain                 | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_bio_pair_risk_adjusted        | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_bio_pair_prob_good            |   0.49963   |                       1.07279 |                               0 |              -0.00188099  |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_bio_pair_prob_good            | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_bio_state_gain                | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_bio_state_risk_adjusted       | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_bio_state_prob_good           | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_style_bio_state_gain          | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_style_bio_state_risk_adjusted | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_style_bio_state_prob_good     | inf         |                       1.07279 |                               0 |               0           |                     0         |                       0.695048 |                              0 |       0           |
| reliability_vehicle_gain                          |   0.0938562 |                       1.07279 |                               0 |               3.00417e-05 |                     0         |                       0.695048 |                              0 |       0.000324147 |
| reliability_vehicle_risk_adjusted                 |   0.218335  |                       1.07279 |                               0 |               3.00417e-05 |                     0         |                       0.695048 |                              0 |       0.000324147 |
| reliability_vehicle_prob_good                     |   0.499435  |                       1.07279 |                               0 |               0.000119794 |                     0         |                       0.695048 |                              0 |       0.00094385  |

## test diagnostic 排名前 18

| feature_set                                       |   threshold |   val_bad_top10_selected_rmse |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate |
|:--------------------------------------------------|------------:|------------------------------:|--------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|
| v280_crossfit_rank_score_only                     |   0.6424    |                       1.13282 |                      0.0600331  |               0.0099851   |                       0.689064 |                      -0.0059849  |                      0.157895  |
| v280_crossfit_rank_score_only                     |   0.652893  |                       1.13282 |                      0.0600331  |               0.00892242  |                       0.689064 |                      -0.0059849  |                      0.105263  |
| v280_crossfit_rank_score_only                     |   0.710154  |                       1.11315 |                      0.0403587  |               0.00120608  |                       0.689064 |                      -0.0059849  |                      0.105263  |
| v280_crossfit_rank_score_only                     |   0.702024  |                       1.11315 |                      0.0403587  |               0.00546337  |                       0.689064 |                      -0.0059849  |                      0.105263  |
| v280_crossfit_rank_score_only                     |   0.871494  |                       1.07175 |                     -0.00103724 |              -0.000953336 |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_style_bio_state_risk_adjusted |   0.187322  |                       1.1059  |                      0.0331087  |               0.00425853  |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_style_bio_state_risk_adjusted |   0.192284  |                       1.07279 |                      0          |               0.000119794 |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_gain                          |   0.0779643 |                       1.07599 |                      0.0032055  |              -0.00182479  |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_gain                          |   0.0630304 |                       1.10449 |                      0.0316973  |               0.00724653  |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_gain                          |   0.0652895 |                       1.10449 |                      0.0316973  |               0.0050921   |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_gain                          |   0.0722584 |                       1.10552 |                      0.0327346  |               0.00214544  |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_style_bio_state_risk_adjusted |   0.181523  |                       1.13543 |                      0.0626378  |               0.00958879  |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_style_bio_state_gain          |   0.0626197 |                       1.1059  |                      0.0331087  |               0.00425853  |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_style_bio_state_gain          |   0.0683585 |                       1.07279 |                      0          |               0.000119794 |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_style_bio_state_risk_adjusted |   0.183199  |                       1.13543 |                      0.0626378  |               0.00708972  |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_style_bio_state_gain          |   0.0585507 |                       1.13543 |                      0.0626378  |               0.00834281  |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| reliability_vehicle_style_bio_state_gain          |   0.0569814 |                       1.13543 |                      0.0626378  |               0.00958879  |                       0.689889 |                      -0.00515939 |                      0.0526316 |
| v280_crossfit_rank_score_only                     |   0.728911  |                       1.11123 |                      0.038447   |               0.00496332  |                       0.689889 |                      -0.00515939 |                      0.0526316 |

## 产物

- `figures\v280_test_badtop10_crossfit_physio_reliability_filter.png`
- `tables/v280_vehicle_listrank_top_candidate_rich.csv`
- `tables/v280_reliability_feature_set_audit.csv`
- `tables/v280_reliability_predictions.csv`
- `tables/v280_score_top_candidates.csv`
- `tables/v280_threshold_search.csv`
- `tables/v280_selected_by_strategy.csv`
- `tables/v280_chosen_configs.csv`
- `tables/v280_decision_summary.csv`
- `logs/guardrail_check.json`
