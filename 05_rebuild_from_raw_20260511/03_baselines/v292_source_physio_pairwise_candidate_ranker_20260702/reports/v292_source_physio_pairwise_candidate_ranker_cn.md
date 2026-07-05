# v292 source-physio pairwise candidate ranker

## 本轮目的

- v291 的 event-level 多信号监督 selector 没有过关。
- v292 改成 pairwise candidate ranking：在每个 query 的 40 个 vehicle-similar train prototype 之间，用源生理匹配程度做 tie-breaker。
- 这一步直接检验“车辆锚点前相似但未来分歧”的核心假设。

## route decision

| check                                                   | requirement                                                                   | pass   |    evidence | deployable   | route_viable_now   |
|:--------------------------------------------------------|:------------------------------------------------------------------------------|:-------|------------:|:-------------|:-------------------|
| deployable_pairwise_selector_beats_latest_bad_top10     | val no-harm active pairwise selector 在 test bad_top10 上低于 latest          | False  | nan         | True         | False              |
| deployable_pairwise_selector_beats_latest_bad_ambiguous | 同一 pairwise selector 在 test bad_top10_vehicle_ambiguous 上低于 latest      | False  | nan         | True         | False              |
| candidate_pool_oracle_has_headroom                      | vehicle top40 candidate pool 在 test bad_top10 上有至少 0.05 RMSE oracle 空间 | True   |  -0.0784456 | False        | False              |

## 候选池 oracle / vehicle top1 边界

| policy                          | event_group                 |   n |   latest_rmse_mean |   selected_rmse_mean |   delta_vs_latest_mean |   candidate_beats_latest_rate |
|:--------------------------------|:----------------------------|----:|-------------------:|---------------------:|-----------------------:|------------------------------:|
| vehicle_score_top1_no_threshold | all                         | 184 |           0.304615 |             0.361142 |              0.0565261 |                      0.255435 |
| vehicle_score_top1_no_threshold | bad_top10                   |  19 |           0.695048 |             0.840349 |              0.1453    |                      0.210526 |
| vehicle_score_top1_no_threshold | bad_top10_vehicle_ambiguous |  15 |           0.744423 |             0.896034 |              0.151611  |                      0.2      |
| oracle_best_candidate           | all                         | 184 |           0.304615 |             0.24189  |             -0.0627255 |                      0.777174 |
| oracle_best_candidate           | bad_top10                   |  19 |           0.695048 |             0.616603 |             -0.0784456 |                      0.631579 |
| oracle_best_candidate           | bad_top10_vehicle_ambiguous |  15 |           0.744423 |             0.656314 |             -0.0881092 |                      0.733333 |

## validation 选择出的 selector

| chosen_type                         | selector_tag                       |   threshold | feature_block           | model_name   |   feature_n |   val_bad_top10_delta_vs_latest_mean |   val_all_delta_vs_latest_mean |   test_bad_top10_delta_vs_latest_mean |   test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean |   test_bad_top10_override_rate |
|:------------------------------------|:-----------------------------------|------------:|:------------------------|:-------------|------------:|-------------------------------------:|-------------------------------:|--------------------------------------:|--------------------------------------------------------:|-------------------------------:|
| fallback_no_override                | vehicle_candidate_score__ridge_a10 | inf         | vehicle_candidate_score | ridge_a10    |           8 |                             0        |                      0         |                             0         |                                               0         |                       0        |
| test_best_diagnostic_not_deployable | bio_all_top_pair_only__hgb_d3      |   0.0084312 | bio_all_top_pair_only   | hgb_d3       |         180 |                             0.140236 |                      0.0367297 |                            -0.0247791 |                                              -0.0313868 |                       0.421053 |

## test diagnostic top selector

| selector_tag                                    |   threshold | feature_block                   | model_name     |   feature_n |   val_bad_top10_delta_vs_latest_mean |   val_all_delta_vs_latest_mean |   test_bad_top10_delta_vs_latest_mean |   test_bad_top10_vehicle_ambiguous_delta_vs_latest_mean |   test_bad_top10_override_rate | noharm_val   |
|:------------------------------------------------|------------:|:--------------------------------|:---------------|------------:|-------------------------------------:|-------------------------------:|--------------------------------------:|--------------------------------------------------------:|-------------------------------:|:-------------|
| bio_all_top_pair_only__hgb_d3                   |  0.0084312  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.140236   |                    0.0367297   |                           -0.0247791  |                                             -0.0313868  |                      0.421053  | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.00625508 | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.145355   |                    0.0376246   |                           -0.0247791  |                                             -0.0313868  |                      0.421053  | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0128709  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0740307  |                    0.0272053   |                           -0.0163421  |                                             -0.0207     |                      0.315789  | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0256679  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0561354  |                    0.0177597   |                           -0.0151604  |                                             -0.0192032  |                      0.105263  | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0280178  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0561354  |                    0.0164094   |                           -0.0151604  |                                             -0.0192032  |                      0.105263  | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0233177  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0627271  |                    0.0185521   |                           -0.0151604  |                                             -0.0192032  |                      0.105263  | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0200152  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0749394  |                    0.0197381   |                           -0.0151604  |                                             -0.0192032  |                      0.105263  | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0180082  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0724012  |                    0.0212362   |                           -0.0134217  |                                             -0.0170008  |                      0.157895  | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0432189  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0          |                    0.00326482  |                           -0.0124128  |                                             -0.0157229  |                      0.0526316 | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0500887  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0          |                    0.00218637  |                           -0.0124128  |                                             -0.0157229  |                      0.0526316 | True         |
| bio_all_top_pair_only__hgb_d3                   |  0.0353748  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0235558  |                    0.00657928  |                           -0.0124128  |                                             -0.0157229  |                      0.0526316 | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0367569  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0235558  |                    0.00589784  |                           -0.0124128  |                                             -0.0157229  |                      0.0526316 | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0324212  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0561354  |                    0.013599    |                           -0.0124128  |                                             -0.0157229  |                      0.0526316 | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0160228  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0724012  |                    0.0212222   |                           -0.00618769 |                                             -0.00783773 |                      0.263158  | False        |
| bio_all_top_pair_only__hgb_d3                   |  0.0141709  | bio_all_top_pair_only           | hgb_d3         |         180 |                           0.0740307  |                    0.0227965   |                           -0.00618769 |                                             -0.00783773 |                      0.263158  | False        |
| vehicle_plus_bio_lowid_top_pair__extra_trees_d5 |  0.0607681  | vehicle_plus_bio_lowid_top_pair | extra_trees_d5 |         188 |                           0.0433862  |                    0.00254335  |                           -0.0059849  |                                             -0.00653523 |                      0.105263  | False        |
| vehicle_candidate_score__extra_trees_d5         |  0.0650429  | vehicle_candidate_score         | extra_trees_d5 |           8 |                           0.00629571 |                    0.00404229  |                           -0.00515939 |                                             -0.00653523 |                      0.0526316 | False        |
| vehicle_candidate_score__ridge_a10              |  0.0448223  | vehicle_candidate_score         | ridge_a10      |           8 |                           0.0371487  |                    0.00459916  |                           -0.00515939 |                                             -0.00653523 |                      0.0526316 | False        |
| vehicle_plus_bio_lowid_top_pair__extra_trees_d5 |  0.0627918  | vehicle_plus_bio_lowid_top_pair | extra_trees_d5 |         188 |                           0.0371487  |                    0.00280685  |                           -0.00515939 |                                             -0.00653523 |                      0.0526316 | False        |
| vehicle_candidate_score__ridge_a10              |  0.0415154  | vehicle_candidate_score         | ridge_a10      |           8 |                           0.0390604  |                    0.00669399  |                           -0.00515939 |                                             -0.00653523 |                      0.0526316 | False        |
| vehicle_candidate_score__extra_trees_d5         |  0.0621499  | vehicle_candidate_score         | extra_trees_d5 |           8 |                           0.0433862  |                    0.00601719  |                           -0.00515939 |                                             -0.00653523 |                      0.0526316 | False        |
| vehicle_plus_bio_all_top_pair__extra_trees_d5   |  0.0627345  | vehicle_plus_bio_all_top_pair   | extra_trees_d5 |         188 |                           0.0476917  |                    0.00575958  |                           -0.00515939 |                                             -0.00653523 |                      0.0526316 | False        |
| bio_all_top_pair_only__ridge_a10                |  0.0278023  | bio_all_top_pair_only           | ridge_a10      |         180 |                           0.00722229 |                    0.000673983 |                            0.00104984 |                                              0.0013298  |                      0.105263  | False        |
| bio_all_top_pair_only__extra_trees_d5           |  0.0100529  | bio_all_top_pair_only           | extra_trees_d5 |         180 |                           0.00637149 |                    0.00198527  |                            0.00224878 |                                              0.00284846 |                      0.0526316 | False        |

## feature block

| feature_block                   | model_name     |   feature_n |
|:--------------------------------|:---------------|------------:|
| vehicle_candidate_score         | ridge_a10      |           8 |
| vehicle_candidate_score         | hgb_d3         |           8 |
| vehicle_candidate_score         | extra_trees_d5 |           8 |
| bio_all_top_pair_only           | ridge_a10      |         180 |
| bio_all_top_pair_only           | hgb_d3         |         180 |
| bio_all_top_pair_only           | extra_trees_d5 |         180 |
| vehicle_plus_bio_all_top_pair   | ridge_a10      |         188 |
| vehicle_plus_bio_all_top_pair   | hgb_d3         |         188 |
| vehicle_plus_bio_all_top_pair   | extra_trees_d5 |         188 |
| bio_lowid_top_pair_only         | ridge_a10      |         180 |
| bio_lowid_top_pair_only         | hgb_d3         |         180 |
| bio_lowid_top_pair_only         | extra_trees_d5 |         180 |
| vehicle_plus_bio_lowid_top_pair | ridge_a10      |         188 |
| vehicle_plus_bio_lowid_top_pair | hgb_d3         |         188 |
| vehicle_plus_bio_lowid_top_pair | extra_trees_d5 |         188 |

## input audit

```json
{
  "candidate_rows": 46680,
  "event_n": 1167,
  "prototype_missing_n": 0,
  "prototype_train_only": true,
  "query_split_event_counts": {
    "test": 184,
    "train": 674,
    "val": 309
  },
  "same_subject_pair_rate_by_split": {
    "test": 0.0,
    "train": 0.20070474777448072,
    "val": 0.0
  }
}
```

## guardrail

```json
{
  "pass": true,
  "event_n": 1167,
  "candidate_rows": 46680,
  "train_event_n": 674,
  "val_event_n": 309,
  "test_event_n": 184,
  "prototype_train_only": true,
  "bio_all_feature_n": 45,
  "bio_lowid_feature_n": 45,
  "selector_config_n": 15,
  "route_viable_now": false,
  "candidate_pool_test_badtop10_oracle_delta": -0.07844557573920805,
  "vehicle_score_top1_test_badtop10_delta": 0.14530041656996073,
  "best_val_noharm_active_exists": false,
  "best_deployable_test_badtop10_delta": null,
  "best_test_diagnostic_badtop10_delta": -0.024779082128876147,
  "test_used_for_feature_screen_or_threshold": false,
  "v291_route_viable_now": false
}
```

## 判断

- v292 没有找到可部署 pairwise 源生理候选排序路线。
- 如果候选池 oracle 很好但 selector 过不了，说明问题不在候选池没有好未来，而在源生理无法稳定识别哪个 prototype 才是对的。
- 这一步比 event-level selector 更贴近“车辆相似但未来分歧”的假设，因此失败会进一步削弱继续堆生理匹配模型的理由。