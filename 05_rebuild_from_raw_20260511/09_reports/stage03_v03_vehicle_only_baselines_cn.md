# v0.3 车辆-only 基线技术报告

## 数据摘要

```json
{
  "dataset_id": "v03_vehicle_only_pre2_label5_20hz",
  "source_episode_table": "F:\\data_set_process\\data_process\\05_rebuild_from_raw_20260511\\02_samples\\extreme_condition_episodes_v0_3\\tables\\extreme_condition_episodes_all_v0_3.csv",
  "usable_categories": [
    "delayed_or_no_steer",
    "normal_control",
    "strong_response",
    "weak_or_conservative"
  ],
  "input_time": [
    -2.0,
    -1.95,
    -1.9,
    -1.85,
    -1.8,
    -1.75,
    -1.7,
    -1.65,
    -1.6,
    -1.55,
    -1.5,
    -1.45,
    -1.4,
    -1.35,
    -1.3,
    -1.25,
    -1.2,
    -1.15,
    -1.1,
    -1.05,
    -1.0,
    -0.95,
    -0.9,
    -0.85,
    -0.8,
    -0.75,
    -0.7,
    -0.65,
    -0.6,
    -0.55,
    -0.5,
    -0.45,
    -0.4,
    -0.35,
    -0.3,
    -0.25,
    -0.2,
    -0.15,
    -0.1,
    -0.05,
    0.0
  ],
  "label_time": [
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
    1.05,
    1.1,
    1.15,
    1.2,
    1.25,
    1.3,
    1.35,
    1.4,
    1.45,
    1.5,
    1.55,
    1.6,
    1.65,
    1.7,
    1.75,
    1.8,
    1.85,
    1.9,
    1.95,
    2.0,
    2.05,
    2.1,
    2.15,
    2.2,
    2.25,
    2.3,
    2.35,
    2.4,
    2.45,
    2.5,
    2.55,
    2.6,
    2.65,
    2.7,
    2.75,
    2.8,
    2.85,
    2.9,
    2.95,
    3.0,
    3.05,
    3.1,
    3.15,
    3.2,
    3.25,
    3.3,
    3.35,
    3.4,
    3.45,
    3.5,
    3.55,
    3.6,
    3.65,
    3.7,
    3.75,
    3.8,
    3.85,
    3.9,
    3.95,
    4.0,
    4.05,
    4.1,
    4.15,
    4.2,
    4.25,
    4.3,
    4.35,
    4.4,
    4.45,
    4.5,
    4.55,
    4.6,
    4.65,
    4.7,
    4.75,
    4.8,
    4.85,
    4.9,
    4.95,
    5.0
  ],
  "feature_names": [
    "zx|SteeringWheel",
    "steer_rate",
    "zx1|v_km/h",
    "zx|BrakePedal",
    "zx|AcceleratorPedal",
    "zx|ax",
    "zx|ay",
    "zx|vyaw",
    "zx|vroll",
    "zx|roll",
    "lateral_distance_selected",
    "zx1|mu",
    "curvature_selected"
  ],
  "sample_count": 482,
  "dropped_count": 0,
  "split_counts": {
    "train": 280,
    "test": 114,
    "val": 88
  },
  "subject_counts": {
    "zx": 57,
    "txj": 49,
    "zdq": 38,
    "rjy": 34,
    "gf": 33,
    "zxy": 33,
    "hzh": 31,
    "yyl": 28,
    "gzj": 24,
    "yzy": 23,
    "lxy": 22,
    "byx": 20,
    "lx": 20,
    "tyy": 16,
    "cwh": 14,
    "xst": 14,
    "zt": 14,
    "jy": 12
  },
  "category_counts": {
    "weak_or_conservative": 208,
    "delayed_or_no_steer": 139,
    "normal_control": 86,
    "strong_response": 49
  },
  "standardization_scope": "all learned models fit preprocessing on train split only"
}
```

## 全部指标

| model_name | split | n | rmse_steer | primary_rmse_0_2s | tail_rmse_2_5s | peak_abs_mae | wrong_side_rate_large | severe_amp_under_rate_large | large_response_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rbf_kernel_vehicle_context_alpha0.1_g2 | test | 114 | 0.797252 | 0.600911 | 0.905429 | 0.649123 | 0.314286 | 0.742857 | 0.428571 |
| zero_delta | test | 114 | 0.811135 | 0.638187 | 0.909279 | 0.982288 | 1 | 1 | 0 |
| knn_vehicle_history_context_k5 | test | 114 | 0.819036 | 0.609355 | 0.933327 | 0.629376 | 0.4 | 0.742857 | 0.628571 |
| ridge_vehicle_history_context_alpha1000 | test | 114 | 0.821677 | 0.647704 | 0.920137 | 0.656143 | 0.314286 | 0.8 | 0.571429 |
| train_global_mean | test | 114 | 0.822071 | 0.636862 | 0.926055 | 0.829358 | 0.542857 | 1 | 0 |
| train_category_mean | test | 114 | 0.825545 | 0.615827 | 0.940586 | 0.766296 | 0.571429 | 0.942857 | 0.257143 |
| train_context_mean | test | 114 | 0.832098 | 0.646957 | 0.936043 | 0.785911 | 0.542857 | 1 | 0 |
| linear_trend_from_last_rate | test | 114 | 4.39105 | 1.79402 | 5.46983 | 3.42279 | 0.828571 | 0.742857 | 0.257143 |
| knn_vehicle_history_context_k5 | train | 280 | 0.00138754 | 0.000788493 | 0.00166906 | 0.00103227 | 0 | 0 | 1 |
| rbf_kernel_vehicle_context_alpha0.1_g2 | train | 280 | 0.126679 | 0.0803945 | 0.149851 | 0.139138 | 0.0571429 | 0.0428571 | 1 |
| ridge_vehicle_history_context_alpha1000 | train | 280 | 0.571285 | 0.377082 | 0.669851 | 0.49494 | 0.242857 | 0.742857 | 0.542857 |
| train_category_mean | train | 280 | 0.697796 | 0.480997 | 0.810572 | 0.602247 | 0.314286 | 0.857143 | 0.285714 |
| train_context_mean | train | 280 | 0.711925 | 0.495316 | 0.825217 | 0.623844 | 0.314286 | 1 | 0 |
| train_global_mean | train | 280 | 0.718617 | 0.499754 | 0.833049 | 0.672764 | 0.314286 | 1 | 0 |
| zero_delta | train | 280 | 0.72861 | 0.502075 | 0.846477 | 0.797652 | 1 | 1 | 0 |
| linear_trend_from_last_rate | train | 280 | 4.77005 | 1.90906 | 5.95003 | 3.14812 | 0.8 | 0.628571 | 0.442857 |
| knn_vehicle_history_context_k5 | val | 88 | 0.442626 | 0.333553 | 0.50231 | 0.445337 | 0.217391 | 0.652174 | 0.521739 |
| rbf_kernel_vehicle_context_alpha0.1_g2 | val | 88 | 0.447658 | 0.310667 | 0.519206 | 0.343106 | 0.130435 | 0.521739 | 0.608696 |
| ridge_vehicle_history_context_alpha1000 | val | 88 | 0.494095 | 0.371409 | 0.561505 | 0.411182 | 0.347826 | 0.695652 | 0.521739 |
| train_category_mean | val | 88 | 0.545422 | 0.417621 | 0.616086 | 0.528496 | 0.347826 | 0.956522 | 0.0434783 |
| zero_delta | val | 88 | 0.552138 | 0.406969 | 0.630572 | 0.710512 | 1 | 1 | 0 |
| train_global_mean | val | 88 | 0.554441 | 0.408293 | 0.633413 | 0.570371 | 0.347826 | 1 | 0 |
| train_context_mean | val | 88 | 0.558133 | 0.414964 | 0.635934 | 0.522532 | 0.347826 | 1 | 0 |
| linear_trend_from_last_rate | val | 88 | 5.95943 | 2.40429 | 7.42963 | 4.23066 | 0.826087 | 0.478261 | 0.565217 |

## 最好模型分样本类型结果

| v0_3_category_cn | n | large_n | rmse_steer_approx | peak_abs_mae | mean_gt_peak_abs | mean_pred_peak_abs | wrong_side_rate_large | severe_amp_under_rate_large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 正常驾驶/普通弯道对照 | 19 | 1 | 0.19208 | 0.154396 | 0.243749 | 0.161511 | 0 | 1 |
| 弱响应/保守响应 | 46 | 11 | 0.665565 | 0.6862 | 0.932864 | 0.361303 | 0.454545 | 1 |
| 延迟或无明显转向响应 | 35 | 14 | 0.808566 | 0.528055 | 1.109 | 0.658116 | 0.214286 | 0.428571 |
| 强响应型极限工况 | 14 | 9 | 1.42677 | 1.50139 | 1.83021 | 0.348994 | 0.333333 | 0.888889 |

## 最好模型分被试结果

| subject | n | large_n | rmse_steer_approx | peak_abs_mae | mean_gt_peak_abs | mean_pred_peak_abs | wrong_side_rate_large | severe_amp_under_rate_large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jy | 2 | 0 | 0.0433604 | 0.062839 | 0 | 0.062839 | NA | NA |
| lx | 13 | 2 | 0.378921 | 0.286202 | 0.700025 | 0.430498 | 0 | 0.5 |
| zdq | 7 | 1 | 0.50251 | 0.195834 | 0.521005 | 0.508843 | 1 | 0 |
| gf | 9 | 4 | 0.56614 | 0.461459 | 0.853185 | 0.449527 | 0.25 | 1 |
| byx | 10 | 2 | 0.601897 | 0.406911 | 0.728423 | 0.557966 | 0 | 0.5 |
| zx | 28 | 11 | 0.784332 | 0.66591 | 1.1123 | 0.525867 | 0.363636 | 0.545455 |
| xst | 14 | 6 | 0.978213 | 0.746189 | 1.01013 | 0.27193 | 0.5 | 0.833333 |
| txj | 31 | 9 | 1.01111 | 1.01511 | 1.25756 | 0.327996 | 0.222222 | 1 |

## 最好模型分工况上下文结果

| condition_context_cn | n | large_n | rmse_steer_approx | peak_abs_mae | mean_gt_peak_abs | mean_pred_peak_abs | wrong_side_rate_large | severe_amp_under_rate_large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 普通驾驶对照 | 18 | 0 | 0.120108 | 0.0854759 | 0.170412 | 0.161104 | NA | NA |
| 横向动态 | 1 | 0 | 0.22861 | 0.372976 | 0.187705 | 0.560681 | NA | NA |
| 低附着 | 80 | 24 | 0.663512 | 0.611106 | 0.935895 | 0.408066 | 0.291667 | 0.791667 |
| 横滚/姿态 | 4 | 1 | 0.686204 | 0.896143 | 1.09903 | 0.20289 | 0 | 1 |
| 弯道/曲率 | 11 | 10 | 1.78489 | 1.78322 | 2.678 | 0.971936 | 0.4 | 0.6 |