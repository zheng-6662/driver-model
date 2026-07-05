# v277 style + calibrated physiology candidate gain model

## 本轮目的

- 在 v276 的 candidate gain 框架上加入驾驶风格和 v271 校准 raw 生理。
- 风格来自 v253a 当前任务口径的 `last60_guard3`，只取 delay=0 事件状态。
- 生理来自 v271 train-only 筛选后的 calibrated raw summary / PCA 特征。
- 同时加入 query-prototype 的 style distance 和 bio271 distance，让模型有机会在车辆相似候选内做状态消歧。
- threshold 只由 val 选择，test 只报告。

## test bad_top10 决策收口

| source               | label                                                           |     rmse | deployable   |   override_rate |   val_bad_delta |   val_all_delta |   stable_pass |   delta_vs_fixed_latest |   passes_fixed_latest |
|:---------------------|:----------------------------------------------------------------|---------:|:-------------|----------------:|----------------:|----------------:|--------------:|------------------------:|----------------------:|
| baseline             | policy_wait_to_latest_anchor                                    | 0.695048 | True         |     nan         |    nan          |     nan         |           nan |           nan           |                   nan |
| oracle               | oracle_best_anchor_upper_bound                                  | 0.612475 | False        |     nan         |    nan          |     nan         |           nan |           nan           |                   nan |
| best_any             | candidate_vehicle_style_query threshold=0.04600033460294534     | 0.695048 | True         |       0         |     -0.00179148 |      -0.0003514 |             1 |             4.15347e-07 |                     0 |
| best_active          | candidate_vehicle_style_query threshold=0.04600033460294534     | 0.695048 | True         |       0         |     -0.00179148 |      -0.0003514 |             1 |             4.15347e-07 |                     0 |
| best_stable_active   | candidate_vehicle_style_query threshold=0.04600033460294534     | 0.695048 | True         |       0         |     -0.00179148 |      -0.0003514 |             1 |             4.15347e-07 |                     0 |
| best_noharm_all      | candidate_vehicle_style_query threshold=0.04600033460294534     | 0.695048 | True         |       0         |     -0.00179148 |      -0.0003514 |             1 |             4.15347e-07 |                     0 |
| test_best_diagnostic | candidate_vehicle_style_bio_dist threshold=0.015090784123417186 | 0.700795 | False        |       0.0526316 |      0.240479   |       0.0309155 |             0 |             0.00574714  |                     0 |

## val 选择出的配置

| chosen_type          | deployable   | feature_set                      |   threshold |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate | stable_pass   |
|:---------------------|:-------------|:---------------------------------|------------:|--------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|:--------------|
| best_any             | True         | candidate_vehicle_style_query    |   0.0460003 |                     -0.00179148 |                -0.0003514 |                       0.695048 |                       0          |                      0         | True          |
| best_active          | True         | candidate_vehicle_style_query    |   0.0460003 |                     -0.00179148 |                -0.0003514 |                       0.695048 |                       0          |                      0         | True          |
| best_stable_active   | True         | candidate_vehicle_style_query    |   0.0460003 |                     -0.00179148 |                -0.0003514 |                       0.695048 |                       0          |                      0         | True          |
| best_noharm_all      | True         | candidate_vehicle_style_query    |   0.0460003 |                     -0.00179148 |                -0.0003514 |                       0.695048 |                       0          |                      0         | True          |
| test_best_diagnostic | False        | candidate_vehicle_style_bio_dist |   0.0150908 |                      0.240479   |                 0.0309155 |                       0.700795 |                       0.00574673 |                      0.0526316 | False         |

## search top by val

| feature_set                       |   threshold |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate | stable_pass   |
|:----------------------------------|------------:|--------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|:--------------|
| candidate_vehicle_style_query     |   0.0460003 |                     -0.00179148 |              -0.0003514   |                       0.695048 |                        0         |                       0        | True          |
| candidate_vehicle_style_query     |   0.0460003 |                     -0.00179148 |              -0.0003514   |                       0.695048 |                        0         |                       0        | True          |
| candidate_vehicle_style_query     |   0.0460003 |                     -0.00179148 |              -0.0003514   |                       0.695048 |                        0         |                       0        | True          |
| candidate_vehicle_style_query     |   0.0460003 |                     -0.00179148 |              -0.0003514   |                       0.695048 |                        0         |                       0        | True          |
| candidate_vehicle                 | inf         |                      0          |               0           |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_style_dist      | inf         |                      0          |               0           |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_bio271_dist     |   0.0304115 |                      0          |              -0.000405055 |                       0.731206 |                        0.0361574 |                       0.105263 | False         |
| candidate_vehicle_bio271_dist     | inf         |                      0          |               0           |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_style_bio_dist  | inf         |                      0          |               0           |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_style_query     | inf         |                      0          |               0           |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_style_bio_query | inf         |                      0          |               0           |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle                 |   0.0816213 |                      0          |               3.55014e-06 |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_style_bio_query |   0.115092  |                      0          |               6.65547e-05 |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_style_query     |   0.0572967 |                      0          |               0.000276984 |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_style_query     |   0.0927432 |                      0          |               0.000662921 |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_style_bio_query |   0.0394829 |                      0          |               0.000549543 |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_style_query     |   0.039997  |                      0.00307982 |               0.000951928 |                       0.695048 |                        0         |                       0        | False         |
| candidate_vehicle_bio271_dist     |   0.0267444 |                      0.00427469 |               0.00116604  |                       0.731206 |                        0.0361574 |                       0.105263 | False         |

## search top by test diagnostic

| feature_set                      |   threshold |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate | stable_pass   |
|:---------------------------------|------------:|--------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|:--------------|
| candidate_vehicle_style_bio_dist |   0.0143186 |                       0.247694  |                0.0365715  |                       0.700795 |                       0.00574673 |                      0.0526316 | False         |
| candidate_vehicle_style_bio_dist |   0.0149522 |                       0.252691  |                0.0353731  |                       0.700795 |                       0.00574673 |                      0.0526316 | False         |
| candidate_vehicle_style_bio_dist |   0.0141628 |                       0.247694  |                0.0386406  |                       0.700795 |                       0.00574673 |                      0.0526316 | False         |
| candidate_vehicle_style_bio_dist |   0.0146033 |                       0.252691  |                0.0354228  |                       0.700795 |                       0.00574673 |                      0.0526316 | False         |
| candidate_vehicle_style_bio_dist |   0.0150908 |                       0.240479  |                0.0309155  |                       0.700795 |                       0.00574673 |                      0.0526316 | False         |
| candidate_vehicle                |   0.0676665 |                       0.030853  |                0.00265052 |                       0.704134 |                       0.0090851  |                      0.0526316 | False         |
| candidate_vehicle                |   0.0595185 |                       0.0603821 |                0.00654497 |                       0.704134 |                       0.0090851  |                      0.0526316 | False         |
| candidate_vehicle                |   0.0484237 |                       0.0603821 |                0.00692842 |                       0.704134 |                       0.0090851  |                      0.0526316 | False         |
| candidate_vehicle_style_bio_dist |   0.0132939 |                       0.256568  |                0.0496261  |                       0.705435 |                       0.0103866  |                      0.105263  | False         |
| candidate_vehicle_style_bio_dist |   0.0135342 |                       0.25105   |                0.0464507  |                       0.705435 |                       0.0103866  |                      0.105263  | False         |
| candidate_vehicle_style_bio_dist |   0.013434  |                       0.256568  |                0.047891   |                       0.705435 |                       0.0103866  |                      0.105263  | False         |
| candidate_vehicle_style_bio_dist |   0.0137974 |                       0.25105   |                0.0394919  |                       0.705435 |                       0.0103866  |                      0.105263  | False         |
| candidate_vehicle_style_bio_dist |   0.0135799 |                       0.25105   |                0.042627   |                       0.705435 |                       0.0103866  |                      0.105263  | False         |
| candidate_vehicle_style_bio_dist |   0.0131168 |                       0.256568  |                0.0499792  |                       0.705435 |                       0.0103866  |                      0.105263  | False         |
| candidate_vehicle_style_dist     |   0.0178487 |                       0.0512472 |                0.0124414  |                       0.707654 |                       0.0126058  |                      0.0526316 | False         |
| candidate_vehicle_style_dist     |   0.0205987 |                       0.0278268 |                0.00915649 |                       0.707654 |                       0.0126058  |                      0.0526316 | False         |
| candidate_vehicle_style_dist     |   0.0224039 |                       0.0278268 |                0.00701725 |                       0.707654 |                       0.0126058  |                      0.0526316 | False         |
| candidate_vehicle_style_dist     |   0.0184178 |                       0.0278268 |                0.00984726 |                       0.707654 |                       0.0126058  |                      0.0526316 | False         |

## 特征审计

- style audited usable feature count: `109`；query feature cap used in model: `96`
- bio271 audited usable feature count: `97`；query feature cap used in model: `96`

## 判读

- 若 best_stable_active 到 test bad_top10 仍为 `0.6950` 且覆盖率为 `0`，说明 val 能选到的稳定策略没有真正修正 test 差样本。
- 若 test_best_diagnostic 也不能低于 fixed wait-latest，说明加入驾驶风格和校准生理后，连事后少量 headroom 都没有扩大。
- 若 style/bio query 特征只在 val 上触发、不在 test bad_top10 覆盖，说明它更像验证集局部模式，而不是可泛化状态消歧信号。

## 关键图

- `figures\v277_test_badtop10_style_bio_candidate_gain.png`
