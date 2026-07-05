# v282 生理歧义消解路线门控审计

## 本轮目的

- 不继续同类 bio selector / reranker / reliability filter 微调。
- 只审计一个基础问题：车辆锚点前相似、候选未来分叉时，生理距离是否稳定指向真实更好的候选。
- `bio_top1` 是可部署近似；`bio_top3/top5 oracle` 只是上限，不可当部署结论。

## route gate 判定

| check                                             | requirement                                                                  | pass   | evidence                    | deployable   | route_viable_now   |
|:--------------------------------------------------|:-----------------------------------------------------------------------------|:-------|:----------------------------|:-------------|:-------------------|
| deployable_top1_val_chosen_bad_top10              | validation 选出的生理 top1 在 test bad_top10 上低于 latest                   | False  | 0.19893167991387217         | True         | False              |
| deployable_top1_val_chosen_bad_ambiguous          | validation 选出的生理 top1 在 test bad_top10_vehicle_ambiguous 上低于 latest | False  | 0.23469711542129515         | True         | False              |
| oracle_top3_val_test_same_direction_bad_ambiguous | 非部署 top3 上限在 val/test 歧义差样本上同向改善                             | False  | val=0.161709, test=0.072432 | False        | False              |
| test_bad_top10_any_rawset_corr_gt_005             | test bad_top10 至少一个 raw_set 的生理距离-真实误差排序相关均值 > 0.05       | False  | 0.009846546301242627        | False        | False              |
| v281_selector_deployable_passes_fixed_latest      | 前序 v281 已证明可训练 selector 能超过 fixed latest                          | False  | 0.6950484153471495          | True         | False              |

## validation 选择 raw_set 后的 test 泛化

| event_group                 | method          | deployable   | val_chosen_raw_set   |   val_delta_vs_latest_mean |   test_delta_vs_latest_mean |   test_corr_mean | test_passes_latest   | val_and_test_same_direction_gain   |
|:----------------------------|:----------------|:-------------|:---------------------|---------------------------:|----------------------------:|-----------------:|:---------------------|:-----------------------------------|
| all                         | bio_top1        | True         | subject_seq_pca72    |                 0.102705   |                  0.0779017  |      -0.00729882 | False                | False                              |
| all                         | bio_top3_oracle | False        | subject_seq_pca72    |                -0.00353788 |                 -5.9347e-05 |      -0.00729882 | True                 | True                               |
| all                         | bio_top5_oracle | False        | subject_seq_pca72    |                -0.046457   |                 -0.0263426  |      -0.00729882 | True                 | True                               |
| vehicle_ambiguous           | bio_top1        | True         | subject_seq_pca72    |                 0.133267   |                  0.0822894  |      -0.0087925  | False                | False                              |
| vehicle_ambiguous           | bio_top3_oracle | False        | subject_seq_pca72    |                 0.0155337  |                  0.00539289 |      -0.0087925  | False                | False                              |
| vehicle_ambiguous           | bio_top5_oracle | False        | subject_seq_pca72    |                -0.0367379  |                 -0.0260595  |      -0.0087925  | True                 | True                               |
| bad_top10                   | bio_top1        | True         | subject_seq_pca72    |                 0.405934   |                  0.198932   |      -0.0240681  | False                | False                              |
| bad_top10                   | bio_top3_oracle | False        | subject_seq_pca72    |                 0.147256   |                  0.0776985  |      -0.0240681  | False                | False                              |
| bad_top10                   | bio_top5_oracle | False        | recording_seq_pca72  |                 0.0234521  |                 -0.0133221  |      -0.036025   | True                 | False                              |
| bad_top10_vehicle_ambiguous | bio_top1        | True         | subject_seq_pca72    |                 0.442706   |                  0.234697   |      -0.0125562  | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top3_oracle | False        | recording_summary64  |                 0.161709   |                  0.0724324  |       0.0163403  | False                | False                              |
| bad_top10_vehicle_ambiguous | bio_top5_oracle | False        | recording_seq_pca72  |                 0.0246318  |                 -0.0272474  |      -0.0292902  | True                 | False                              |
| early_best_after_400        | bio_top1        | True         | subject_seq_pca72    |                 0.142956   |                  0.101057   |      -0.0140748  | False                | False                              |
| early_best_after_400        | bio_top3_oracle | False        | subject_seq_pca72    |                 0.0299864  |                  0.018167   |      -0.0140748  | False                | False                              |
| early_best_after_400        | bio_top5_oracle | False        | recording_seq_pca72  |                -0.0179288  |                 -0.0142362  |      -0.0202038  | True                 | True                               |

## bad_top10 分层结果

| raw_set                   | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_corr_positive_rate |   bio_best_in_top3_rate |
|:--------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|-------------------------:|------------------------:|
| calibrated_screened64     | test    |  19 |           0.695048 |             0.87444  |                     0.179392 |                    0.771211 |                    0.076163  |     -0.039844   |                 0.421053 |               0.0526316 |
| subject_seq_pca72         | test    |  19 |           0.695048 |             0.89398  |                     0.198932 |                    0.772747 |                    0.0776985 |     -0.0240681  |                 0.421053 |               0.105263  |
| recording_summary64       | test    |  19 |           0.695048 |             0.903212 |                     0.208163 |                    0.752034 |                    0.0569852 |     -0.00374735 |                 0.526316 |               0.0526316 |
| calibrated_low_identity48 | test    |  19 |           0.695048 |             0.908005 |                     0.212956 |                    0.830328 |                    0.135279  |     -0.0661491  |                 0.210526 |               0.0526316 |
| recording_seq_pca72       | test    |  19 |           0.695048 |             0.922643 |                     0.227595 |                    0.751149 |                    0.056101  |     -0.036025   |                 0.421053 |               0.105263  |
| subject_summary64         | test    |  19 |           0.695048 |             0.934999 |                     0.23995  |                    0.673823 |                   -0.0212253 |      0.00984655 |                 0.473684 |               0.105263  |
| subject_seq_pca72         | val     |  31 |           1.07279  |             1.47872  |                     0.405934 |                    1.22004  |                    0.147256  |      0.0240005  |                 0.516129 |               0.129032  |
| recording_seq_pca72       | val     |  31 |           1.07279  |             1.52961  |                     0.456821 |                    1.26583  |                    0.193042  |      0.00355563 |                 0.516129 |               0.129032  |
| calibrated_low_identity48 | val     |  31 |           1.07279  |             1.55382  |                     0.48103  |                    1.25323  |                    0.180447  |     -0.0390295  |                 0.451613 |               0.129032  |
| calibrated_screened64     | val     |  31 |           1.07279  |             1.56427  |                     0.49148  |                    1.23116  |                    0.158375  |     -0.00164106 |                 0.451613 |               0.129032  |
| recording_summary64       | val     |  31 |           1.07279  |             1.58639  |                     0.513603 |                    1.22239  |                    0.149605  |      0.0201747  |                 0.548387 |               0.129032  |
| subject_summary64         | val     |  31 |           1.07279  |             1.66146  |                     0.588672 |                    1.33951  |                    0.266723  |     -0.0160151  |                 0.451613 |               0.0322581 |

## bad_top10 + vehicle_ambiguous 分层结果

| raw_set                   | split   |   n |   latest_rmse_mean |   bio_top1_rmse_mean |   bio_top1_minus_latest_mean |   bio_top3_oracle_rmse_mean |   bio_top3_minus_latest_mean |   bio_corr_mean |   bio_corr_positive_rate |   bio_best_in_top3_rate |
|:--------------------------|:--------|----:|-------------------:|---------------------:|-----------------------------:|----------------------------:|-----------------------------:|----------------:|-------------------------:|------------------------:|
| subject_summary64         | test    |  15 |           0.744423 |             0.928752 |                     0.184329 |                    0.707873 |                   -0.0365501 |      0.0314057  |                 0.533333 |               0.133333  |
| recording_summary64       | test    |  15 |           0.744423 |             0.929904 |                     0.185481 |                    0.816855 |                    0.0724324 |      0.0163403  |                 0.533333 |               0.0666667 |
| calibrated_screened64     | test    |  15 |           0.744423 |             0.943351 |                     0.198928 |                    0.823613 |                    0.0791901 |     -0.0329134  |                 0.466667 |               0.0666667 |
| calibrated_low_identity48 | test    |  15 |           0.744423 |             0.948604 |                     0.204181 |                    0.871661 |                    0.127238  |     -0.0646187  |                 0.266667 |               0.0666667 |
| recording_seq_pca72       | test    |  15 |           0.744423 |             0.972218 |                     0.227795 |                    0.778885 |                    0.0344624 |     -0.0292902  |                 0.466667 |               0.133333  |
| subject_seq_pca72         | test    |  15 |           0.744423 |             0.97912  |                     0.234697 |                    0.825558 |                    0.0811351 |     -0.0125562  |                 0.4      |               0.133333  |
| subject_seq_pca72         | val     |  27 |           1.02949  |             1.4722   |                     0.442706 |                    1.19627  |                    0.166777  |      0.0328913  |                 0.518519 |               0.148148  |
| recording_seq_pca72       | val     |  27 |           1.02949  |             1.53063  |                     0.501131 |                    1.23978  |                    0.210283  |      0.0112467  |                 0.518519 |               0.148148  |
| calibrated_low_identity48 | val     |  27 |           1.02949  |             1.53491  |                     0.505413 |                    1.22532  |                    0.195822  |     -0.0241941  |                 0.481481 |               0.148148  |
| calibrated_screened64     | val     |  27 |           1.02949  |             1.56728  |                     0.537786 |                    1.20904  |                    0.179543  |      0.00859391 |                 0.444444 |               0.148148  |
| recording_summary64       | val     |  27 |           1.02949  |             1.59268  |                     0.563187 |                    1.1912   |                    0.161709  |      0.0258527  |                 0.555556 |               0.148148  |
| subject_summary64         | val     |  27 |           1.02949  |             1.68755  |                     0.658054 |                    1.32609  |                    0.296595  |     -0.0118159  |                 0.481481 |               0.037037  |

## split 一致性摘要

| raw_set                   | event_group                 | method          |   train_delta_vs_latest_mean |   val_delta_vs_latest_mean |   test_delta_vs_latest_mean |   train_corr_mean |   val_corr_mean |   test_corr_mean | val_test_improve_latest   | corr_positive_all_splits   |
|:--------------------------|:----------------------------|:----------------|-----------------------------:|---------------------------:|----------------------------:|------------------:|----------------:|-----------------:|:--------------------------|:---------------------------|
| subject_seq_pca72         | bad_top10                   | bio_top1        |                    0.0181612 |                   0.405934 |                   0.198932  |       -0.00584204 |      0.0240005  |      -0.0240681  | False                     | False                      |
| recording_seq_pca72       | bad_top10                   | bio_top1        |                    0.0222407 |                   0.456821 |                   0.227595  |       -0.00166135 |      0.00355563 |      -0.036025   | False                     | False                      |
| calibrated_low_identity48 | bad_top10                   | bio_top1        |                    0.0193989 |                   0.48103  |                   0.212956  |       -0.00091641 |     -0.0390295  |      -0.0661491  | False                     | False                      |
| calibrated_screened64     | bad_top10                   | bio_top1        |                    0.0232172 |                   0.49148  |                   0.179392  |       -0.00755243 |     -0.00164106 |      -0.039844   | False                     | False                      |
| recording_summary64       | bad_top10                   | bio_top1        |                    0.0307187 |                   0.513603 |                   0.208163  |        0.0109169  |      0.0201747  |      -0.00374735 | False                     | False                      |
| subject_summary64         | bad_top10                   | bio_top1        |                    0.0132475 |                   0.588672 |                   0.23995   |       -0.0244902  |     -0.0160151  |       0.00984655 | False                     | False                      |
| subject_seq_pca72         | bad_top10                   | bio_top3_oracle |                   -0.0321485 |                   0.147256 |                   0.0776985 |       -0.00584204 |      0.0240005  |      -0.0240681  | False                     | False                      |
| recording_summary64       | bad_top10                   | bio_top3_oracle |                   -0.0213403 |                   0.149605 |                   0.0569852 |        0.0109169  |      0.0201747  |      -0.00374735 | False                     | False                      |
| calibrated_screened64     | bad_top10                   | bio_top3_oracle |                   -0.0235237 |                   0.158375 |                   0.076163  |       -0.00755243 |     -0.00164106 |      -0.039844   | False                     | False                      |
| calibrated_low_identity48 | bad_top10                   | bio_top3_oracle |                   -0.0286197 |                   0.180447 |                   0.135279  |       -0.00091641 |     -0.0390295  |      -0.0661491  | False                     | False                      |
| recording_seq_pca72       | bad_top10                   | bio_top3_oracle |                   -0.0316513 |                   0.193042 |                   0.056101  |       -0.00166135 |      0.00355563 |      -0.036025   | False                     | False                      |
| subject_summary64         | bad_top10                   | bio_top3_oracle |                   -0.0270354 |                   0.266723 |                  -0.0212253 |       -0.0244902  |     -0.0160151  |       0.00984655 | False                     | False                      |
| subject_seq_pca72         | bad_top10_vehicle_ambiguous | bio_top1        |                    0.0232156 |                   0.442706 |                   0.234697  |        0.00415187 |      0.0328913  |      -0.0125562  | False                     | False                      |
| recording_seq_pca72       | bad_top10_vehicle_ambiguous | bio_top1        |                    0.0309245 |                   0.501131 |                   0.227795  |        0.0103647  |      0.0112467  |      -0.0292902  | False                     | False                      |
| calibrated_low_identity48 | bad_top10_vehicle_ambiguous | bio_top1        |                    0.0215893 |                   0.505413 |                   0.204181  |        0.00938023 |     -0.0241941  |      -0.0646187  | False                     | False                      |
| calibrated_screened64     | bad_top10_vehicle_ambiguous | bio_top1        |                    0.0238711 |                   0.537786 |                   0.198928  |        0.0196594  |      0.00859391 |      -0.0329134  | False                     | False                      |
| recording_summary64       | bad_top10_vehicle_ambiguous | bio_top1        |                    0.0415392 |                   0.563187 |                   0.185481  |        0.051567   |      0.0258527  |       0.0163403  | False                     | True                       |
| subject_summary64         | bad_top10_vehicle_ambiguous | bio_top1        |                    0.0175669 |                   0.658054 |                   0.184329  |       -0.0258914  |     -0.0118159  |       0.0314057  | False                     | False                      |
| recording_summary64       | bad_top10_vehicle_ambiguous | bio_top3_oracle |                   -0.019131  |                   0.161709 |                   0.0724324 |        0.051567   |      0.0258527  |       0.0163403  | False                     | True                       |
| subject_seq_pca72         | bad_top10_vehicle_ambiguous | bio_top3_oracle |                   -0.0395084 |                   0.166777 |                   0.0811351 |        0.00415187 |      0.0328913  |      -0.0125562  | False                     | False                      |
| calibrated_screened64     | bad_top10_vehicle_ambiguous | bio_top3_oracle |                   -0.0259674 |                   0.179543 |                   0.0791901 |        0.0196594  |      0.00859391 |      -0.0329134  | False                     | False                      |
| calibrated_low_identity48 | bad_top10_vehicle_ambiguous | bio_top3_oracle |                   -0.0341949 |                   0.195822 |                   0.127238  |        0.00938023 |     -0.0241941  |      -0.0646187  | False                     | False                      |
| recording_seq_pca72       | bad_top10_vehicle_ambiguous | bio_top3_oracle |                   -0.0352311 |                   0.210283 |                   0.0344624 |        0.0103647  |      0.0112467  |      -0.0292902  | False                     | False                      |
| subject_summary64         | bad_top10_vehicle_ambiguous | bio_top3_oracle |                   -0.0302571 |                   0.296595 |                  -0.0365501 |       -0.0258914  |     -0.0118159  |       0.0314057  | False                     | False                      |

## 关键判读

- route gate 未通过：当前生理特征不能稳定解决 subject-disjoint 的车辆相似/未来分叉问题。
- 如果继续坚持生理主线，下一步不应继续换 selector，而应重新定义生理状态特征，例如从 200Hz 连续层重提取事件前自主神经状态、响应相位、短时变化率和质量控制后的个体内变化。
- 对预测效果主线而言，当前更可靠的路线仍是车辆多未来分布、不确定性建模或 anchor-aware 联合任务；生理只能作为新的特征重构分支重新进入。

## 关键图

- `figures\v282_badtop10_val_test_bio_delta.png`
- `figures\v282_bad_ambiguous_bio_rank_corr.png`
- `figures\v282_val_chosen_test_delta.png`

## guardrail

```json
{
  "pass": true,
  "zip_testzip": true,
  "k_focus": 40,
  "input_diag_rows": 28008,
  "expanded_rows": 20472,
  "raw_set_n": 6,
  "summary_rows": 144,
  "val_chosen_rows": 15,
  "fixed_wait_latest_badtop10": 0.695048,
  "route_viable_now": false,
  "deployable_top1_badtop10_pass": false,
  "deployable_top1_bad_ambiguous_pass": false,
  "oracle_top3_bad_ambiguous_stable_pass": false,
  "v281_selector_deployable_passes_fixed_latest": false
}
```
