# Feature Input Audit

## Scope

- Training script: `F:\data_set_process\data_process\02_code\final_code\model\training\future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
- Resolved data root: `F:\data_set_process\data_process\01_datasets\多模态数据\被试数据集合`
- Vehicle file scan mode: `exact_training_pattern`
- Total vehicle files audited: `91`

## Event Pair Coverage

- Missing paired `v312` event files: `5`
- Missing paired `v400` event files: `5`
- Missing `v312` basenames: `Entity_Recording_2025_09_28_18_16_26, Entity_Recording_2025_09_27_08_39_31, Entity_Recording_2025_09_27_17_40_06, Entity_Recording_2025_09_27_17_40_34, Entity_Recording_2025_09_28_16_25_33`
- Missing `v400` basenames: `Entity_Recording_2025_09_28_18_16_26, Entity_Recording_2025_09_27_08_39_31, Entity_Recording_2025_09_27_17_40_06, Entity_Recording_2025_09_27_17_40_34, Entity_Recording_2025_09_28_16_25_33`

## Lane Column Naming

- Exact lane-related column counts: `zx1|lateraldistance (91)`
- `zx1|lateraldistance` dominant naming: `yes`
- Lane distance observed in `91` files, but exact training lookup aliases match `0` files.

## Feature Presence Summary

Numeric columns report `global_min`, `median_of_file_medians`, and `global_max` across files.

| feature_key        | files_present | training_exact | observed_exact_columns   | global_min | median_of_file_medians | global_max |
| ------------------ | ------------- | -------------- | ------------------------ | ---------- | ---------------------- | ---------- |
| time_s             | 91            | 91             | t_s (91)                 | 0          | 285.765                | 903.475    |
| storage_time       | 91            | 91             | StorageTime (91)         |            |                        |            |
| roll               | 91            | 91             | zx|roll (91)             | -1.39071   | -0.001439              | 1.3798     |
| steering_wheel     | 91            | 91             | zx|SteeringWheel (91)    | -6.9609    | -0.129853              | 7.49043    |
| yaw_rate           | 91            | 91             | zx|vyaw (91)             | -3.18538   | -0.000265              | 4.10609    |
| speed_vx           | 91            | 91             | zx|vx (91)               | -26.181    | 30.8477                | 44.82      |
| speed_kmh          | 91            | 91             | zx1|v_km/h (91)          | 0          | 111.123                | 161.353    |
| z_position         | 91            | 91             | zx|z (91)                | -13.861    | -7.14501               | 0.477378   |
| lateral_accel      | 91            | 91             | zx|ay (91)               | -16.5414   | -0.003114              | 16.3379    |
| longitudinal_accel | 91            | 91             | zx|ax (91)               | -10.9443   | 0.197301               | 10.3123    |
| lane_distance      | 91            | 0              | zx1|lateraldistance (91) | -227.601   | 0.078597               | 345.125    |
| lane_curvature     | 91            | 91             | zx1|lanecurvatureXY (91) | -0.520512  | 0                      | 0.019863   |
| road_type_fixed    | 91            | 91             | road_type_fixed (91)     | 0          | 0                      | 1          |
| ref_nn_ok          | 91            | 91             | ref_nn_ok (91)           | 0          | 1                      | 1          |
| yaw                | 91            | 91             | zx|yaw (91)              | -28.7439   | -0.866122              | 21.198     |

## Speed Unit Check

- Files with both speed columns and positive finite ratio rows: `90`
- Files with zero positive finite ratio rows: `1`
- Files whose ratio median is within `+-0.01` of `3.6`: `88`
- Median of per-file ratio medians: `3.600221`
- Zero-ratio basenames: `Entity_Recording_2025_09_28_16_25_33`
- Outlier files with ratio median more than `0.05` away from `3.6` are shown below.

| basename                             | ratio_rows | ratio_min  | ratio_median | ratio_max   |
| ------------------------------------ | ---------- | ---------- | ------------ | ----------- |
| Entity_Recording_2025_09_27_17_40_06 | 4          | 117.641978 | 159.009601   | 810.771911  |
| Entity_Recording_2025_09_27_17_40_34 | 47         | 0.070459   | 30.048559    | 1663.050458 |

- Interpretation: when the ratio clusters near `3.6`, `zx|vx` behaves like m/s and `zx1|v_km/h` behaves like km/h.
