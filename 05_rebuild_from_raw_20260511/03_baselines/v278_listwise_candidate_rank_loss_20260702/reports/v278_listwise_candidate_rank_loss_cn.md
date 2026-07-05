# v278 listwise candidate rank loss

## 本轮目的

- 把候选选择从“绝对收益回归”改成“同事件组内排序标签”。
- 比较 vehicle-only、vehicle+bio、vehicle+style+bio。
- 若生理能补足车辆锚点前信息不足，应该看到 `listrank_vehicle_bio` 或 `listrank_vehicle_style_bio` 优于 `listrank_vehicle`。
- 阈值仍只由 val 选择，test 只报告。

## test bad_top10 决策收口

| source               | label                                         |     rmse | deployable   |   override_rate |   val_bad_delta |   val_all_delta |   stable_pass |   delta_vs_fixed_latest |   passes_fixed_latest |
|:---------------------|:----------------------------------------------|---------:|:-------------|----------------:|----------------:|----------------:|--------------:|------------------------:|----------------------:|
| baseline             | policy_wait_to_latest_anchor                  | 0.695048 | True         |      nan        |     nan         |    nan          |           nan |           nan           |                   nan |
| oracle               | oracle_best_anchor_upper_bound                | 0.612475 | False        |      nan        |     nan         |    nan          |           nan |           nan           |                   nan |
| best_any             | listrank_vehicle threshold=inf                | 0.695048 | True         |        0        |       0         |      0          |             0 |             4.15347e-07 |                     0 |
| best_active          | listrank_vehicle threshold=1.518402327900305  | 0.695048 | True         |        0        |       0         |      0          |             1 |             4.15347e-07 |                     0 |
| best_stable_active   | listrank_vehicle threshold=1.518402327900305  | 0.695048 | True         |        0        |       0         |      0          |             1 |             4.15347e-07 |                     0 |
| best_noharm_all      | listrank_vehicle threshold=1.518402327900305  | 0.695048 | True         |        0        |       0         |      0          |             1 |             4.15347e-07 |                     0 |
| test_best_diagnostic | listrank_vehicle threshold=0.6707545714677929 | 0.683215 | False        |        0.105263 |       0.0508972 |      0.00770668 |             0 |            -0.0118333   |                     1 |

## val 选择出的配置

| chosen_type          | deployable   | feature_set      |   threshold |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate | stable_pass   |
|:---------------------|:-------------|:-----------------|------------:|--------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|:--------------|
| best_any             | True         | listrank_vehicle |  inf        |                       0         |                0          |                       0.695048 |                        0         |                       0        | False         |
| best_active          | True         | listrank_vehicle |    1.5184   |                       0         |                0          |                       0.695048 |                        0         |                       0        | True          |
| best_stable_active   | True         | listrank_vehicle |    1.5184   |                       0         |                0          |                       0.695048 |                        0         |                       0        | True          |
| best_noharm_all      | True         | listrank_vehicle |    1.5184   |                       0         |                0          |                       0.695048 |                        0         |                       0        | True          |
| test_best_diagnostic | False        | listrank_vehicle |    0.670755 |                       0.0508972 |                0.00770668 |                       0.683215 |                       -0.0118337 |                       0.105263 | False         |

## search top by val

| feature_set                |   threshold |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate | stable_pass   |
|:---------------------------|------------:|--------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|:--------------|
| listrank_vehicle           |    1.5184   |                      0          |               0           |                       0.695048 |                       0          |                      0         | True          |
| listrank_vehicle           |  inf        |                      0          |               0           |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle_bio       |  inf        |                      0          |               0           |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle_style_bio |  inf        |                      0          |               0           |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle_bio       |    0.971318 |                      0          |               0.000231366 |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle           |    0.930669 |                      0          |              -0.000856    |                       0.695048 |                       0          |                      0         | True          |
| listrank_vehicle_bio       |    0.795039 |                      0          |               0.00117631  |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle_style_bio |    0.972792 |                      0          |               0.00124991  |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle           |    0.874569 |                      0          |               0.000229361 |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle           |    0.836855 |                      0.00629571 |              -0.0039248   |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle_style_bio |    0.79921  |                      0.00511842 |               0.00209223  |                       0.732279 |                       0.0372302  |                      0.0526316 | False         |
| listrank_vehicle           |    0.806151 |                      0.0125332  |              -0.00334263  |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle           |    0.769153 |                      0.0125332  |              -0.00308329  |                       0.689889 |                      -0.00515939 |                      0.0526316 | False         |
| listrank_vehicle_bio       |    0.742308 |                      0          |               0.00468706  |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle_bio       |    0.706352 |                      0.00408118 |               0.0049588   |                       0.695048 |                       0          |                      0         | False         |
| listrank_vehicle           |    0.750697 |                      0.0125332  |              -0.000749242 |                       0.689889 |                      -0.00515939 |                      0.0526316 | False         |
| listrank_vehicle_style_bio |    0.757469 |                      0.00511842 |               0.00508023  |                       0.732279 |                       0.0372302  |                      0.0526316 | False         |
| listrank_vehicle_bio       |    0.678006 |                      0.00408118 |               0.0066313   |                       0.725604 |                       0.0305558  |                      0.157895  | False         |

## search top by test diagnostic

| feature_set                |   threshold |   val_bad_top10_delta_vs_latest |   val_all_delta_vs_latest |   test_bad_top10_selected_rmse |   test_bad_top10_delta_vs_latest |   test_bad_top10_override_rate | stable_pass   |
|:---------------------------|------------:|--------------------------------:|--------------------------:|-------------------------------:|---------------------------------:|-------------------------------:|:--------------|
| listrank_vehicle           |    0.670755 |                       0.0508972 |               0.00770668  |                       0.683215 |                      -0.0118337  |                      0.105263  | False         |
| listrank_vehicle           |    0.713857 |                       0.0508972 |               0.00645782  |                       0.689889 |                      -0.00515939 |                      0.0526316 | False         |
| listrank_vehicle           |    0.750697 |                       0.0125332 |              -0.000749242 |                       0.689889 |                      -0.00515939 |                      0.0526316 | False         |
| listrank_vehicle           |    0.692735 |                       0.0508972 |               0.00762035  |                       0.689889 |                      -0.00515939 |                      0.0526316 | False         |
| listrank_vehicle           |    0.738114 |                       0.0433862 |               0.0028651   |                       0.689889 |                      -0.00515939 |                      0.0526316 | False         |
| listrank_vehicle           |    0.769153 |                       0.0125332 |              -0.00308329  |                       0.689889 |                      -0.00515939 |                      0.0526316 | False         |
| listrank_vehicle           |    0.662886 |                       0.0508972 |               0.00959513  |                       0.704287 |                       0.00923864 |                      0.157895  | False         |
| listrank_vehicle           |    0.636852 |                       0.0508972 |               0.00788276  |                       0.704287 |                       0.00923864 |                      0.157895  | False         |
| listrank_vehicle           |    0.617418 |                       0.0589143 |               0.0118952   |                       0.704287 |                       0.00923864 |                      0.157895  | False         |
| listrank_vehicle_bio       |    0.450685 |                       0.139927  |               0.033072    |                       0.706192 |                       0.0111432  |                      0.315789  | False         |
| listrank_vehicle_bio       |    0.420863 |                       0.139927  |               0.0359858   |                       0.706192 |                       0.0111432  |                      0.315789  | False         |
| listrank_vehicle_bio       |    0.408524 |                       0.144232  |               0.0392028   |                       0.706192 |                       0.0111432  |                      0.315789  | False         |
| listrank_vehicle_bio       |    0.432424 |                       0.139927  |               0.0345349   |                       0.706192 |                       0.0111432  |                      0.315789  | False         |
| listrank_vehicle_bio       |    0.519647 |                       0.138853  |               0.0280706   |                       0.71545  |                       0.0204015  |                      0.263158  | False         |
| listrank_vehicle_bio       |    0.471215 |                       0.138853  |               0.0305616   |                       0.71545  |                       0.0204015  |                      0.263158  | False         |
| listrank_vehicle_bio       |    0.490095 |                       0.138853  |               0.0292961   |                       0.71545  |                       0.0204015  |                      0.263158  | False         |
| listrank_vehicle_bio       |    0.500283 |                       0.138853  |               0.0281922   |                       0.71545  |                       0.0204015  |                      0.263158  | False         |
| listrank_vehicle_style_bio |    0.57859  |                       0.0219738 |               0.0156053   |                       0.71545  |                       0.0204015  |                      0.263158  | False         |

## 特征组

| feature_set                |   feature_n |
|:---------------------------|------------:|
| listrank_vehicle           |          10 |
| listrank_vehicle_bio       |          24 |
| listrank_vehicle_style_bio |          25 |

## 判读

- listwise rank loss 的 vehicle-only diagnostic 可以暴露更大的候选选择 headroom。
- 如果 vehicle+bio 低于 vehicle-only，说明生理能帮助组内候选排序。
- 如果 vehicle+bio 仍差于 vehicle-only，说明当前生理没有在候选选择损失层面提供稳定增量。
- deployable 配置若 test 覆盖率为 0，则不能算差样本本质改善。

## 关键图

- `figures\v278_test_badtop10_listrank_candidate_loss.png`
