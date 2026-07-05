# v276 bio-assisted candidate gain model

## 本轮目的

- 回到 v267 full vehicle top40 候选池，而不是只用 bio top5。
- 对 query-candidate pair 训练相对 latest 的候选收益预测器。
- 比较 candidate_vehicle、candidate_vehicle_bio、candidate_bio_only 三组特征，判断生理是否能辅助车辆多未来候选选择。
- threshold 只由 val 选择，test 只报告。

## test bad_top10 决策收口

| source                    | label                                            |     rmse | deployable   |   override_rate |   val_bad_delta |   val_all_delta |   stable_pass |   delta_vs_fixed_latest | passes_fixed_latest   |
|:--------------------------|:-------------------------------------------------|---------:|:-------------|----------------:|----------------:|----------------:|--------------:|------------------------:|:----------------------|
| baseline                  | policy_wait_to_latest_anchor                     | 0.695048 | True         |     nan         |     nan         |    nan          |           nan |             4.15347e-07 | False                 |
| oracle                    | oracle_best_anchor_upper_bound                   | 0.612475 | False        |     nan         |     nan         |    nan          |           nan |            -0.0825726   | True                  |
| val_best_any              | candidate_bio_only threshold=inf                 | 0.695048 | True         |       0         |       0         |      0          |             0 |             4.15347e-07 | False                 |
| val_best_active           | candidate_bio_only threshold=0.05258202427102294 | 0.695048 | True         |       0         |       0.0042796 |      0.00376238 |             0 |             4.15347e-07 | False                 |
| test_best_gain_diagnostic | candidate_bio_only threshold=0.04017248155900985 | 0.68579  | False        |       0.0526316 |       0.0277    |      0.00786613 |             0 |            -0.00925781  | True                  |

## val 选择出的配置

| chosen_type          | deployable   | feature_set        |   threshold |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   val_normal_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate | stable_pass   |
|:---------------------|:-------------|:-------------------|------------:|--------------------------------:|--------------------------:|-----------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|:--------------|
| best_any             | True         | candidate_bio_only | inf         |                       0         |                0          |                   0          |                       0.695048 |                       0          |                      0         | False         |
| best_active          | True         | candidate_bio_only |   0.052582  |                       0.0042796 |                0.00376238 |                   0.00077947 |                       0.695048 |                       0          |                      0         | False         |
| test_best_diagnostic | False        | candidate_bio_only |   0.0401725 |                       0.0277    |                0.00786613 |                   0.00370873 |                       0.68579  |                      -0.00925823 |                      0.0526316 | False         |

## search top by val bad_top10

| feature_set           |   threshold |   val_bad_top10_selected_rmse |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   val_normal_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate | stable_pass   |
|:----------------------|------------:|------------------------------:|--------------------------------:|--------------------------:|-----------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|:--------------|
| candidate_bio_only    |   0.052582  |                       1.07707 |                       0.0042796 |                0.00376238 |                  0.00077947  |                       0.695048 |                       0          |                      0         | False         |
| candidate_bio_only    |   0.0412965 |                       1.10049 |                       0.0277    |                0.00607753 |                  0.000209218 |                       0.695048 |                       0          |                      0         | False         |
| candidate_bio_only    |   0.0440484 |                       1.10049 |                       0.0277    |                0.00648929 |                  0.000489824 |                       0.695048 |                       0          |                      0         | False         |
| candidate_bio_only    |   0.0401725 |                       1.10049 |                       0.0277    |                0.00786613 |                  0.00370873  |                       0.68579  |                      -0.00925823 |                      0.0526316 | False         |
| candidate_vehicle     |   0.0676665 |                       1.10364 |                       0.030853  |                0.00265052 |                  0.000906935 |                       0.704134 |                       0.0090851  |                      0.0526316 | False         |
| candidate_vehicle_bio |   0.0516401 |                       1.10364 |                       0.030853  |                0.00458915 |                 -0.000192561 |                       0.695048 |                       0          |                      0         | False         |
| candidate_vehicle_bio |   0.0460645 |                       1.10364 |                       0.030853  |                0.00910137 |                  0.00275531  |                       0.720304 |                       0.0252554  |                      0.105263  | False         |
| candidate_vehicle     |   0.0595185 |                       1.13317 |                       0.0603821 |                0.00654497 |                  0.00134634  |                       0.704134 |                       0.0090851  |                      0.0526316 | False         |
| candidate_vehicle     |   0.0484237 |                       1.13317 |                       0.0603821 |                0.00692842 |                  0.00187541  |                       0.704134 |                       0.0090851  |                      0.0526316 | False         |
| candidate_vehicle     |   0.0431829 |                       1.13534 |                       0.0625503 |                0.00783755 |                  0.00406153  |                       0.720304 |                       0.0252554  |                      0.105263  | False         |
| candidate_vehicle     |   0.0374135 |                       1.13534 |                       0.0625503 |                0.00699072 |                  0.010983    |                       0.720304 |                       0.0252554  |                      0.105263  | False         |
| candidate_vehicle     |   0.0406323 |                       1.13534 |                       0.0625503 |                0.00922281 |                  0.00693671  |                       0.720304 |                       0.0252554  |                      0.105263  | False         |
| candidate_vehicle     |   0.0364484 |                       1.14046 |                       0.0676687 |                0.00952413 |                  0.0120802   |                       0.720304 |                       0.0252554  |                      0.105263  | False         |
| candidate_vehicle     |   0.0354262 |                       1.14046 |                       0.0676687 |                0.0152166  |                  0.0240897   |                       0.757534 |                       0.0624856  |                      0.157895  | False         |
| candidate_vehicle     |   0.0343906 |                       1.14046 |                       0.0676687 |                0.0192728  |                  0.0305161   |                       0.757534 |                       0.0624856  |                      0.157895  | False         |
| candidate_vehicle     |   0.0331237 |                       1.14046 |                       0.0676687 |                0.0209016  |                  0.0320892   |                       0.757534 |                       0.0624856  |                      0.157895  | False         |
| candidate_vehicle     |   0.0288947 |                       1.14046 |                       0.0676687 |                0.023544   |                  0.0352661   |                       0.756708 |                       0.0616601  |                      0.210526  | False         |
| candidate_vehicle     |   0.031247  |                       1.14046 |                       0.0676687 |                0.0238649  |                  0.0340322   |                       0.757534 |                       0.0624856  |                      0.157895  | False         |
| candidate_vehicle_bio |   0.038712  |                       1.14058 |                       0.0677918 |                0.0184138  |                  0.017799    |                       0.720304 |                       0.0252554  |                      0.105263  | False         |
| candidate_vehicle_bio |   0.0329054 |                       1.14058 |                       0.0677918 |                0.0195824  |                  0.0175116   |                       0.720304 |                       0.0252554  |                      0.105263  | False         |
| candidate_vehicle_bio |   0.0318287 |                       1.14058 |                       0.0677918 |                0.0201152  |                  0.0149445   |                       0.757534 |                       0.0624856  |                      0.157895  | False         |
| candidate_vehicle_bio |   0.0344591 |                       1.14058 |                       0.0677918 |                0.0200143  |                  0.0190801   |                       0.720304 |                       0.0252554  |                      0.105263  | False         |
| candidate_vehicle_bio |   0.0300649 |                       1.14058 |                       0.0677918 |                0.0235886  |                  0.0211908   |                       0.757534 |                       0.0624856  |                      0.157895  | False         |
| candidate_vehicle_bio |   0.0287539 |                       1.14058 |                       0.0677918 |                0.0268455  |                  0.0282471   |                       0.757534 |                       0.0624856  |                      0.157895  | False         |

## 判读

- val 选择出的候选收益模型仍未低于 fixed wait-latest。
- 如果 test diagnostic 低于 fixed wait-latest 但 val 上伤害，说明模型事后能碰到少数样本，但没有稳定可部署规则。
- 若 candidate_vehicle_bio 没有稳定优于 candidate_vehicle，则当前生理仍不能作为多未来候选选择的主增量。

## 关键图

- `figures\v276_test_badtop10_candidate_gain_model.png`