# v267 supervised bio prototype reranker

## 本轮问题

- v266 说明 vehicle-matched prototype 候选库有 headroom，但简单距离/投票规则选不准。
- v267 用 train query-prototype pair 监督训练 reranker，检验更强的可部署候选选择器能否把 headroom 转成 test bad_top10 收益。

## 方法边界

- prototype 只来自 train split。
- query 只使用 0ms 车辆上下文和 observation_s 之前的 bio260_sp64。
- 标签是 train query 在 prototype oracle delay 下的真实 tail RMSE；val/test 只用于选择/报告，不参与训练。
- 生理只参与候选内部 reranking，不直接生成轨迹。

## 特征块

| model                            | family       | feature_block   | bad_weight   |   feature_n |
|:---------------------------------|:-------------|:----------------|:-------------|------------:|
| pair_base_hgb                    | base         | base            | False        |           9 |
| pair_vehicle_hgb                 | vehicle_only | vehicle         | False        |         102 |
| pair_bio_hgb                     | vehicle_bio  | bio             | False        |         204 |
| pair_vehicle_bio_hgb             | vehicle_bio  | vehicle_bio     | False        |         297 |
| pair_vehicle_bio_badweighted_hgb | vehicle_bio  | vehicle_bio     | True         |         297 |

## Pair 构造审计

|   event_n |   pair_n |   train_pair_n |   val_pair_n |   test_pair_n |   vehicle_feature_n |   bio260_sp64_feature_n |   max_k |   bio260_uses_post_observation_max |
|----------:|---------:|---------------:|-------------:|--------------:|--------------------:|------------------------:|--------:|-----------------------------------:|
|      1167 |    46680 |          26960 |        12360 |          7360 |                  31 |                      65 |      40 |                                  0 |

## Test bad_top10 关键对照

| strategy                                                       | strategy_family   | deployable   |   n |   selected_tail_rmse_mean |   delta_selected_minus_keep0_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |   selected_latest_rate |
|:---------------------------------------------------------------|:------------------|:-------------|----:|--------------------------:|----------------------------------:|-----------------------------------:|-------------------------:|-----------------------:|
| policy_keep_0ms_anchor                                         | baseline          | True         |  19 |                  1.19771  |                          0        |                          0.502658  |                    0     |              0         |
| policy_wait_to_latest_anchor                                   | baseline          | True         |  19 |                  0.695048 |                         -0.502658 |                          0         |                 1000     |              1         |
| oracle_best_anchor_upper_bound                                 | oracle            | False        |  19 |                  0.612475 |                         -0.585231 |                         -0.082573  |                  818.421 |              0.368421  |
| pair_candidate_oracle_k40                                      | candidate_oracle  | False        |  19 |                  0.616603 |                         -0.581103 |                         -0.0784456 |                  831.579 |              0.368421  |
| pair_candidate_oracle_k20                                      | candidate_oracle  | False        |  19 |                  0.625011 |                         -0.572695 |                         -0.0700371 |                  813.158 |              0.315789  |
| pair_candidate_oracle_k10                                      | candidate_oracle  | False        |  19 |                  0.642559 |                         -0.555148 |                         -0.0524897 |                  818.421 |              0.263158  |
| pair_vehicle_hgb_k20                                           | vehicle_only      | True         |  19 |                  0.860336 |                         -0.33737  |                          0.165288  |                  555.263 |              0.0526316 |
| pair_vehicle_hgb_k5                                            | vehicle_only      | True         |  19 |                  0.866039 |                         -0.331667 |                          0.170991  |                  560.526 |              0.0526316 |
| pair_vehicle_hgb_k3                                            | vehicle_only      | True         |  19 |                  0.86834  |                         -0.329366 |                          0.173291  |                  518.421 |              0.0526316 |
| pair_vehicle_bio_hgb_k20                                       | vehicle_bio       | True         |  19 |                  0.8046   |                         -0.393106 |                          0.109552  |                  623.684 |              0.0526316 |
| pair_bio_hgb_k5                                                | vehicle_bio       | True         |  19 |                  0.814229 |                         -0.383477 |                          0.119181  |                  594.737 |              0.0526316 |
| pair_bio_hgb_k10                                               | vehicle_bio       | True         |  19 |                  0.820958 |                         -0.376748 |                          0.12591   |                  623.684 |              0.0526316 |
| pair_base_hgb_k10                                              | base              | True         |  19 |                  0.794692 |                         -0.403015 |                          0.0996432 |                  750     |              0.263158  |
| pair_base_hgb_k40                                              | base              | True         |  19 |                  0.800515 |                         -0.397191 |                          0.105467  |                  671.053 |              0.0526316 |
| pair_base_hgb_k3                                               | base              | True         |  19 |                  0.827802 |                         -0.369904 |                          0.132753  |                  568.421 |              0.105263  |
| val_best_pair_vehicle: pair_vehicle_hgb_k40                    | vehicle_only      | True         |  19 |                  0.874617 |                         -0.32309  |                          0.179568  |                  502.632 |              0.105263  |
| val_best_pair_vehicle_bio: pair_vehicle_bio_badweighted_hgb_k5 | vehicle_bio       | True         |  19 |                  0.849545 |                         -0.348161 |                          0.154497  |                  536.842 |              0         |
| val_best_pair_any: pair_vehicle_hgb_k40                        | vehicle_only      | True         |  19 |                  0.874617 |                         -0.32309  |                          0.179568  |                  502.632 |              0.105263  |

## Val 选择的可部署策略

| chosen_label              | chosen_strategy                     | chosen_family   | split   | event_group   |   n |   selected_tail_rmse_mean |   delta_selected_minus_latest_mean |   selected_delay_ms_mean |
|:--------------------------|:------------------------------------|:----------------|:--------|:--------------|----:|--------------------------:|-----------------------------------:|-------------------------:|
| val_best_pair_vehicle     | pair_vehicle_hgb_k40                | vehicle_only    | train   | all           | 674 |                 0.0915448 |                         -0.0334472 |                  606.825 |
| val_best_pair_vehicle     | pair_vehicle_hgb_k40                | vehicle_only    | train   | bad_top10     |  68 |                 0.145897  |                         -0.0408686 |                  661.765 |
| val_best_pair_vehicle     | pair_vehicle_hgb_k40                | vehicle_only    | val     | all           | 309 |                 0.576553  |                          0.0986937 |                  610.841 |
| val_best_pair_vehicle     | pair_vehicle_hgb_k40                | vehicle_only    | val     | bad_top10     |  31 |                 1.49206   |                          0.419271  |                  693.548 |
| val_best_pair_vehicle     | pair_vehicle_hgb_k40                | vehicle_only    | test    | all           | 184 |                 0.360166  |                          0.0555504 |                  638.859 |
| val_best_pair_vehicle     | pair_vehicle_hgb_k40                | vehicle_only    | test    | bad_top10     |  19 |                 0.874617  |                          0.179568  |                  502.632 |
| val_best_pair_vehicle_bio | pair_vehicle_bio_badweighted_hgb_k5 | vehicle_bio     | train   | all           | 674 |                 0.101775  |                         -0.0232168 |                  622.774 |
| val_best_pair_vehicle_bio | pair_vehicle_bio_badweighted_hgb_k5 | vehicle_bio     | train   | bad_top10     |  68 |                 0.147969  |                         -0.038797  |                  650.735 |
| val_best_pair_vehicle_bio | pair_vehicle_bio_badweighted_hgb_k5 | vehicle_bio     | val     | all           | 309 |                 0.574675  |                          0.0968157 |                  618.77  |
| val_best_pair_vehicle_bio | pair_vehicle_bio_badweighted_hgb_k5 | vehicle_bio     | val     | bad_top10     |  31 |                 1.56579   |                          0.492999  |                  491.935 |
| val_best_pair_vehicle_bio | pair_vehicle_bio_badweighted_hgb_k5 | vehicle_bio     | test    | all           | 184 |                 0.370417  |                          0.065801  |                  615.489 |
| val_best_pair_vehicle_bio | pair_vehicle_bio_badweighted_hgb_k5 | vehicle_bio     | test    | bad_top10     |  19 |                 0.849545  |                          0.154497  |                  536.842 |
| val_best_pair_any         | pair_vehicle_hgb_k40                | vehicle_only    | train   | all           | 674 |                 0.0915448 |                         -0.0334472 |                  606.825 |
| val_best_pair_any         | pair_vehicle_hgb_k40                | vehicle_only    | train   | bad_top10     |  68 |                 0.145897  |                         -0.0408686 |                  661.765 |
| val_best_pair_any         | pair_vehicle_hgb_k40                | vehicle_only    | val     | all           | 309 |                 0.576553  |                          0.0986937 |                  610.841 |
| val_best_pair_any         | pair_vehicle_hgb_k40                | vehicle_only    | val     | bad_top10     |  31 |                 1.49206   |                          0.419271  |                  693.548 |
| val_best_pair_any         | pair_vehicle_hgb_k40                | vehicle_only    | test    | all           | 184 |                 0.360166  |                          0.0555504 |                  638.859 |
| val_best_pair_any         | pair_vehicle_hgb_k40                | vehicle_only    | test    | bad_top10     |  19 |                 0.874617  |                          0.179568  |                  502.632 |

## 判读

- candidate oracle 最好为 `0.6166`，仍证明候选库 headroom 存在。
- val-best pair vehicle 在 test bad_top10 为 `0.8746`。
- val-best pair vehicle+bio 在 test bad_top10 为 `0.8495`。
- 生理监督式 reranker 比 vehicle-only 低 `0.0251`。
- vehicle+bio 仍高于 fixed wait-latest，不能算差样本本质改善。
- 若 val 选择的策略在 test 上不稳定，说明当前 pairwise 监督信号存在 split 泛化问题。

## 关键图

- `figures\v267_test_badtop10_pair_reranker.png`
- `figures\v267_val_test_badtop10_generalization.png`