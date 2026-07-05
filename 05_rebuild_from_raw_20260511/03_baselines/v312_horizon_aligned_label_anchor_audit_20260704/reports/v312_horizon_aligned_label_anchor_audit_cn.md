# v312 horizon-aligned label / anchor audit

## 这一步做了什么

v312 把事件标签拆成两层：

- `local_0_2_motion_label`：模型当前 0-2s 预测窗口内真实要预测的局部动作。
- `late_2_6_context_label`：2-6s 后续真实车辆动作，只作为后续上下文和锚点审计。

这一步不训练模型，目的是为下一轮 confirmed 标签和模型输入边界做准备。

## 总体结果

- delay0 事件数：`1167`
- coarse label 与 horizon 局部窗口存在错位嫌疑：`227`
- local flat 但 late 2-6s 出现大动作：`49`
- local 与 late 方向冲突：`98`
- v309 severe overlay 事件数：`37`
- severe 中错位嫌疑：`11`

## 按 split 的 horizon alignment 分布

| split   | horizon_alignment_label         |   event_n |   severe_bad10_n |   coarse_mismatch_n |
|:--------|:--------------------------------|----------:|-----------------:|--------------------:|
| test    | roughly_aligned                 |       159 |               13 |                   0 |
| test    | late_dominant_context           |        27 |                3 |                  27 |
| test    | local_dominant_current          |        24 |                5 |                   0 |
| test    | current_late_direction_conflict |        15 |                3 |                  12 |
| test    | current_flat_late_event         |         6 |                0 |                   6 |
| test    | stable_flat                     |         1 |                0 |                   0 |
| train   | roughly_aligned                 |       474 |               47 |                   0 |
| train   | local_dominant_current          |        74 |               10 |                   0 |
| train   | late_dominant_context           |        64 |                5 |                  64 |
| train   | current_late_direction_conflict |        57 |                8 |                  42 |
| train   | current_flat_late_event         |        26 |                1 |                  26 |
| train   | stable_flat                     |         7 |                0 |                   0 |
| val     | roughly_aligned                 |       142 |               11 |                   0 |
| val     | local_dominant_current          |        31 |                4 |                   0 |
| val     | current_late_direction_conflict |        26 |                6 |                  18 |
| val     | current_flat_late_event         |        17 |                0 |                  17 |
| val     | late_dominant_context           |        15 |                3 |                  15 |
| val     | stable_flat                     |         2 |                0 |                   0 |

## severe 复核优先 Top 20

|   severe_rank |   screenshot_rank | event_uid                                         | coarse_scene_label_cn   | error_tags                                                          | local_0_2_motion_label     | late_2_6_context_label    | horizon_alignment_label         | recommended_label_action          |   late_over_local_abs_ratio |
|--------------:|------------------:|:--------------------------------------------------|:------------------------|:--------------------------------------------------------------------|:---------------------------|:--------------------------|:--------------------------------|:----------------------------------|----------------------------:|
|             8 |               nan | zx_Entity_Recording_2025_09_27_16_32_00_v108_040  | 平路过弯                | false_large_maneuver                                                | local_0_2_flat_hold        | late_2_6_extreme_positive | current_flat_late_event         | split_local_flat_and_late_context |                   35.5401   |
|             3 |                20 | zx_Entity_Recording_2025_09_27_17_14_07_v108_016  | 连续变道/连续左右修正   | false_large_maneuver;regression_vs_v300;shown_in_user_screenshot    | local_0_2_flat_hold        | late_2_6_extreme_positive | current_flat_late_event         | split_local_flat_and_late_context |                   28.5029   |
|            37 |               nan | gf_Entity_Recording_2025_09_26_10_30_12_v108_006  | 下坡过弯                | regression_vs_v300                                                  | local_0_2_flat_hold        | late_2_6_extreme_negative | current_flat_late_event         | split_local_flat_and_late_context |                   16.7977   |
|            13 |               nan | gzj_Entity_Recording_2025_09_27_11_53_25_v108_042 | 平路过弯                | opposite_peak_direction                                             | local_0_2_mild_positive    | late_2_6_extreme_positive | late_dominant_context           | keep_late_context_separate        |                    7.92327  |
|             4 |                23 | gzj_Entity_Recording_2025_09_27_11_41_47_v108_048 | 平路过弯                | opposite_peak_direction;regression_vs_v300;shown_in_user_screenshot | local_0_2_mild_negative    | late_2_6_extreme_negative | late_dominant_context           | keep_late_context_separate        |                    4.40844  |
|            33 |               nan | zdq_Entity_Recording_2025_09_26_16_03_48_v108_019 | 平路过弯                | regression_vs_v300                                                  | local_0_2_mild_positive    | late_2_6_extreme_positive | late_dominant_context           | keep_late_context_separate        |                    4.33609  |
|            25 |               nan | gzj_Entity_Recording_2025_09_27_11_41_47_v108_021 | 平路过弯                | regression_vs_v300                                                  | local_0_2_mild_negative    | late_2_6_extreme_positive | current_late_direction_conflict | split_current_and_late_direction  |                    4.00748  |
|            30 |               nan | lxy_Entity_Recording_2025_09_28_17_55_52_v108_021 | 紧急变道/猛打方向失稳   | regression_vs_v300                                                  | local_0_2_mild_positive    | late_2_6_strong_positive  | late_dominant_context           | keep_late_context_separate        |                    2.55156  |
|            27 |               nan | byx_Entity_Recording_2025_09_28_17_35_43_v108_032 | 下坡过弯                | regression_vs_v300                                                  | local_0_2_strong_positive  | late_2_6_strong_positive  | late_dominant_context           | keep_late_context_separate        |                    1.56814  |
|             7 |                14 | gzj_Entity_Recording_2025_09_27_12_28_14_v108_054 | 平路过弯                | opposite_peak_direction;shown_in_user_screenshot                    | local_0_2_mild_positive    | late_2_6_mild_negative    | current_late_direction_conflict | split_current_and_late_direction  |                    1.35357  |
|            24 |               nan | tyy_Entity_Recording_2025_09_28_14_23_43_v108_026 | 下坡过弯                | regression_vs_v300                                                  | local_0_2_extreme_negative | late_2_6_strong_positive  | current_late_direction_conflict | split_current_and_late_direction  |                    0.529607 |
|             5 |               nan | zdq_Entity_Recording_2025_09_26_15_14_51_v108_012 | 连续变道/连续左右修正   | false_large_maneuver;regression_vs_v300                             | local_0_2_flat_hold        | late_2_6_mild_positive    | roughly_aligned                 | no_anchor_change_needed           |                    4.94261  |
|            12 |               nan | gzj_Entity_Recording_2025_09_27_12_28_14_v108_052 | 平路过弯                | opposite_peak_direction                                             | local_0_2_mild_positive    | late_2_6_mild_positive    | roughly_aligned                 | no_anchor_change_needed           |                    1.53849  |
|            10 |               nan | hzh_Entity_Recording_2025_09_27_19_44_05_v108_020 | 下坡过弯                | opposite_peak_direction                                             | local_0_2_mild_positive    | late_2_6_mild_positive    | roughly_aligned                 | no_anchor_change_needed           |                    1.42225  |
|            34 |               nan | zx_Entity_Recording_2025_09_27_18_17_48_v108_031  | 连续变道/连续左右修正   | regression_vs_v300                                                  | local_0_2_strong_negative  | late_2_6_strong_negative  | roughly_aligned                 | no_anchor_change_needed           |                    1.21398  |
|            16 |               nan | byx_Entity_Recording_2025_09_28_17_25_18_v108_037 | 连续变道/连续左右修正   | missed_extreme_amplitude                                            | local_0_2_extreme_negative | late_2_6_extreme_negative | roughly_aligned                 | no_anchor_change_needed           |                    1.19619  |
|            18 |               nan | lxy_Entity_Recording_2025_09_28_18_19_35_v108_027 | 其他/不确定             | missed_extreme_amplitude                                            | local_0_2_extreme_negative | late_2_6_extreme_negative | roughly_aligned                 | no_anchor_change_needed           |                    1.18584  |
|            23 |               nan | byx_Entity_Recording_2025_09_28_17_35_43_v108_036 | 其他/不确定             | regression_vs_v300                                                  | local_0_2_extreme_negative | late_2_6_extreme_negative | roughly_aligned                 | no_anchor_change_needed           |                    1.14517  |
|            17 |               nan | zx_Entity_Recording_2025_09_27_16_46_13_v108_053  | 平路过弯                | missed_extreme_amplitude                                            | local_0_2_extreme_negative | late_2_6_extreme_negative | roughly_aligned                 | no_anchor_change_needed           |                    1.12592  |
|            15 |               nan | txj_Entity_Recording_2025_09_27_08_40_46_v108_033 | 紧急变道/猛打方向失稳   | missed_extreme_amplitude                                            | local_0_2_extreme_negative | late_2_6_extreme_negative | roughly_aligned                 | no_anchor_change_needed           |                    1.1218   |

## 粗标签与 local/late 标签组合 Top 25

| coarse_scene_label     | local_0_2_motion_label     | late_2_6_context_label    |   event_n |   mismatch_n |   local_flat_late_large_n |   direction_conflict_n |
|:-----------------------|:---------------------------|:--------------------------|----------:|-------------:|--------------------------:|-----------------------:|
| continuous_lane_change | local_0_2_strong_positive  | late_2_6_strong_positive  |        92 |            2 |                         0 |                      0 |
| continuous_lane_change | local_0_2_strong_negative  | late_2_6_strong_negative  |        77 |            3 |                         0 |                      0 |
| continuous_lane_change | local_0_2_mild_negative    | late_2_6_mild_negative    |        37 |            0 |                         0 |                      0 |
| continuous_lane_change | local_0_2_mild_positive    | late_2_6_strong_positive  |        21 |           12 |                         0 |                      0 |
| continuous_lane_change | local_0_2_strong_positive  | late_2_6_mild_positive    |        20 |            0 |                         0 |                      0 |
| continuous_lane_change | local_0_2_mild_positive    | late_2_6_mild_positive    |        19 |            0 |                         0 |                      0 |
| continuous_lane_change | local_0_2_strong_negative  | late_2_6_mild_negative    |        19 |            0 |                         0 |                      0 |
| continuous_lane_change | local_0_2_extreme_positive | late_2_6_extreme_positive |        17 |            2 |                         0 |                      0 |
| continuous_lane_change | local_0_2_mild_negative    | late_2_6_strong_negative  |        15 |            9 |                         0 |                      0 |
| continuous_lane_change | local_0_2_strong_negative  | late_2_6_extreme_negative |        13 |            6 |                         0 |                      0 |
| continuous_lane_change | local_0_2_strong_positive  | late_2_6_extreme_positive |        12 |            5 |                         0 |                      0 |
| continuous_lane_change | local_0_2_extreme_positive | late_2_6_strong_positive  |        11 |            0 |                         0 |                      0 |
| continuous_lane_change | local_0_2_extreme_negative | late_2_6_extreme_negative |        10 |            0 |                         0 |                      0 |
| continuous_lane_change | local_0_2_extreme_negative | late_2_6_strong_negative  |         7 |            0 |                         0 |                      0 |
| continuous_lane_change | local_0_2_flat_hold        | late_2_6_flat_hold        |         6 |            0 |                         0 |                      0 |
| continuous_lane_change | local_0_2_flat_hold        | late_2_6_extreme_negative |         5 |            5 |                         5 |                      0 |
| continuous_lane_change | local_0_2_strong_positive  | late_2_6_strong_negative  |         5 |            2 |                         0 |                      5 |
| continuous_lane_change | local_0_2_flat_hold        | late_2_6_extreme_positive |         4 |            4 |                         4 |                      0 |
| continuous_lane_change | local_0_2_flat_hold        | late_2_6_mild_negative    |         4 |            0 |                         0 |                      0 |
| continuous_lane_change | local_0_2_flat_hold        | late_2_6_mild_positive    |         3 |            0 |                         0 |                      0 |
| continuous_lane_change | local_0_2_mild_negative    | late_2_6_mild_positive    |         3 |            0 |                         0 |                      2 |
| continuous_lane_change | local_0_2_flat_hold        | late_2_6_strong_positive  |         2 |            2 |                         2 |                      0 |
| continuous_lane_change | local_0_2_strong_negative  | late_2_6_mild_positive    |         2 |            0 |                         0 |                      2 |
| continuous_lane_change | local_0_2_strong_negative  | late_2_6_strong_positive  |         2 |            0 |                         0 |                      2 |
| continuous_lane_change | local_0_2_extreme_positive | late_2_6_extreme_negative |         1 |            0 |                         0 |                      1 |

## 当前判断

- 下一轮不应把 `late_2_6_context_label` 当作 0-2s 预测输入。
- 对 `current_flat_late_event`，0-2s 训练应以 flat/hold 为局部标签，late event 只作为后续上下文。
- 对 `current_late_direction_conflict`，必须把当前方向和后续方向分开，否则模型会学到反向。
- 可以优先让用户人工复核 severe overlay 表中的 `split_local_flat_and_late_context` 和 `split_current_and_late_direction`。

## guardrail

```json
{
  "pass": true,
  "version": "v312_horizon_aligned_label_anchor_audit_20260704",
  "training_run": false,
  "event_n": 1167,
  "local_0_2_label_source": "true_target_curve_0_2s_diagnostic_not_deployable_input",
  "late_2_6_context_source": "raw_vehicle_future_2_6s_diagnostic_not_original_anchor_input",
  "uses_test_error_as_training_feature": false,
  "candidate_selection_uses_test": false,
  "deployable_without_manual_or_preanchor_label": false,
  "coarse_label_horizon_mismatch_n": 227,
  "local_flat_late_large_n": 49,
  "local_late_direction_conflict_n": 98,
  "severe_overlay_n": 37,
  "severe_coarse_label_horizon_mismatch_n": 11,
  "figure_paths": [
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v312_horizon_aligned_label_anchor_audit_20260704\\figures\\v312_local_0_2_label_distribution.png",
    "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\03_baselines\\v312_horizon_aligned_label_anchor_audit_20260704\\figures\\v312_horizon_alignment_by_split.png"
  ],
  "runtime_seconds": 59.78507399559021
}
```