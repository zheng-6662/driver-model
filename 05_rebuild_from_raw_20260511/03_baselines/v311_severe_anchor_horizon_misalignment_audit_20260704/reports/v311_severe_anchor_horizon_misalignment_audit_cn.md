# v311 severe anchor / horizon misalignment audit

## 这一步做了什么

本审计专门解释 v309/v310 暴露的严重错例：比较模型真正预测范围 `0-2s` 内的真实峰值，和 raw 车辆数据中 `2-6s` 后续峰值。

如果一个样本 `0-2s` 内真实几乎没动作，但 `2s` 后才有大动作，而 v307 在 `0-2s` 就预测大动作，那么问题更像是锚点/标签窗口错位，而不是普通网络拟合不足。

## 汇总

| group                    |   n |   raw_available_n |   label_horizon_mismatch_suspected_n |   predicts_future_too_early_n |   horizon_flat_post_large_n |   horizon_post_opposite_n |   post2_dominant_n |   post2_over_horizon_ratio_median |
|:-------------------------|----:|------------------:|-------------------------------------:|------------------------------:|----------------------------:|--------------------------:|-------------------:|----------------------------------:|
| all_severe               |  37 |                37 |                                   11 |                             2 |                           3 |                         3 |                  9 |                           1.1218  |
| user_screenshot          |   5 |                 5 |                                    3 |                             1 |                           1 |                         1 |                  2 |                           1.35357 |
| opposite_peak_direction  |   8 |                 8 |                                    3 |                             1 |                           0 |                         1 |                  2 |                           1.38791 |
| false_large_maneuver     |   3 |                 3 |                                    2 |                             1 |                           2 |                         0 |                  2 |                          28.5029  |
| missed_extreme_amplitude |   8 |                 8 |                                    0 |                             0 |                           0 |                         0 |                  0 |                           1.08036 |

## 优先复核样本 Top 20

|   severe_rank |   screenshot_rank | event_uid                                         | coarse_scene_label_cn   |   true_peak |   v307_peak |   raw_2_6_peak |   post2_over_horizon_abs_ratio | misalignment_tags                                                                               |
|--------------:|------------------:|:--------------------------------------------------|:------------------------|------------:|------------:|---------------:|-------------------------------:|:------------------------------------------------------------------------------------------------|
|             8 |               nan | zx_Entity_Recording_2025_09_27_16_32_00_v108_040  | 平路过弯                |   -0.03281  |    2.63567  |       3.55401  |                      35.5401   | horizon_flat_post_large;post2_dominant;model_follows_post_not_horizon;predicts_future_too_early |
|             7 |                14 | gzj_Entity_Recording_2025_09_27_12_28_14_v108_054 | 平路过弯                |    0.589398 |   -0.836158 |      -0.79779  |                       1.35357  | horizon_post_opposite;model_follows_post_not_horizon;predicts_future_too_early                  |
|             3 |                20 | zx_Entity_Recording_2025_09_27_17_14_07_v108_016  | 连续变道/连续左右修正   |    0.08883  |    2.98088  |       2.85029  |                      28.5029   | horizon_flat_post_large;post2_dominant                                                          |
|            37 |               nan | gf_Entity_Recording_2025_09_26_10_30_12_v108_006  | 下坡过弯                |    0.15254  |    0.378227 |      -2.56231  |                      16.7977   | horizon_flat_post_large;post2_dominant                                                          |
|            13 |               nan | gzj_Entity_Recording_2025_09_27_11_53_25_v108_042 | 平路过弯                |    0.53215  |   -0.445486 |       4.21637  |                       7.92327  | post2_dominant                                                                                  |
|             4 |                23 | gzj_Entity_Recording_2025_09_27_11_41_47_v108_048 | 平路过弯                |   -0.544015 |    1.88366  |      -2.39825  |                       4.40844  | post2_dominant                                                                                  |
|            33 |               nan | zdq_Entity_Recording_2025_09_26_16_03_48_v108_019 | 平路过弯                |    0.699526 |    0.832994 |       3.03321  |                       4.33609  | post2_dominant                                                                                  |
|            25 |               nan | gzj_Entity_Recording_2025_09_27_11_41_47_v108_021 | 平路过弯                |   -0.815942 |   -1.01281  |       3.26987  |                       4.00748  | post2_dominant;horizon_post_opposite                                                            |
|            30 |               nan | lxy_Entity_Recording_2025_09_28_17_55_52_v108_021 | 紧急变道/猛打方向失稳   |    0.475602 |    1.0902   |       1.21353  |                       2.55156  | post2_dominant                                                                                  |
|            27 |               nan | byx_Entity_Recording_2025_09_28_17_35_43_v108_032 | 下坡过弯                |    1.06168  |    1.65317  |       1.66487  |                       1.56814  | post2_dominant                                                                                  |
|            24 |               nan | tyy_Entity_Recording_2025_09_28_14_23_43_v108_026 | 下坡过弯                |   -2.39354  |   -1.27266  |       1.26764  |                       0.529607 | horizon_post_opposite                                                                           |
|             5 |               nan | zdq_Entity_Recording_2025_09_26_15_14_51_v108_012 | 连续变道/连续左右修正   |   -0.20089  |   -1.3705   |       0.99292  |                       4.94261  | no_clear_anchor_misalignment                                                                    |
|            12 |               nan | gzj_Entity_Recording_2025_09_27_12_28_14_v108_052 | 平路过弯                |    0.473857 |   -0.711354 |       0.729026 |                       1.53849  | no_clear_anchor_misalignment                                                                    |
|            10 |               nan | hzh_Entity_Recording_2025_09_27_19_44_05_v108_020 | 下坡过弯                |    0.611738 |   -0.915025 |       0.870046 |                       1.42225  | no_clear_anchor_misalignment                                                                    |
|            34 |               nan | zx_Entity_Recording_2025_09_27_18_17_48_v108_031  | 连续变道/连续左右修正   |   -1.39801  |   -0.739905 |      -1.69716  |                       1.21398  | no_clear_anchor_misalignment                                                                    |
|            16 |               nan | byx_Entity_Recording_2025_09_28_17_25_18_v108_037 | 连续变道/连续左右修正   |   -2.18219  |   -0.767254 |      -2.61032  |                       1.19619  | no_clear_anchor_misalignment                                                                    |
|            18 |               nan | lxy_Entity_Recording_2025_09_28_18_19_35_v108_027 | 其他/不确定             |   -2.18253  |   -0.896909 |      -2.58814  |                       1.18584  | no_clear_anchor_misalignment                                                                    |
|            23 |               nan | byx_Entity_Recording_2025_09_28_17_35_43_v108_036 | 其他/不确定             |   -2.57767  |   -2.27656  |      -2.95187  |                       1.14517  | no_clear_anchor_misalignment                                                                    |
|            17 |               nan | zx_Entity_Recording_2025_09_27_16_46_13_v108_053  | 平路过弯                |   -2.22739  |   -0.968006 |      -2.50786  |                       1.12592  | no_clear_anchor_misalignment                                                                    |
|            15 |               nan | txj_Entity_Recording_2025_09_27_08_40_46_v108_033 | 紧急变道/猛打方向失稳   |   -3.14386  |   -1.16432  |      -3.52678  |                       1.1218   | no_clear_anchor_misalignment                                                                    |

## 当前判断

- `predicts_future_too_early=True` 的样本，模型很可能把 2s 后才发生的动作提前预测到了 0-2s。
- `horizon_flat_post_large=True` 的样本，不应该仅靠加大转向/失稳标签权重解决，因为真实预测窗口内目标就是平的。
- `horizon_post_opposite=True` 的样本，粗标签可能描述的是后续事件方向，而不是当前 0-2s 目标方向。
- 下一轮更应改成 horizon-aligned 标签或锚点重定义，而不是继续堆 loss 权重。

## guardrail

```json
{
  "pass": true,
  "version": "v311_severe_anchor_horizon_misalignment_audit_20260704",
  "training_run": false,
  "uses_v309_severe_table_for_diagnostic_only": true,
  "uses_test_error_as_training_feature": false,
  "candidate_selection_uses_test": false,
  "severe_event_n": 37,
  "raw_available_n": 37,
  "label_horizon_mismatch_suspected_n": 11,
  "predicts_future_too_early_n": 2,
  "runtime_seconds": 26.04047179222107
}
```