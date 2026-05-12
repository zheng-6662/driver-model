# 阶段 3 v0.4 候选强车辆基线模型卡：RBF KRR 无被试 ID

更新时间：2026-05-12

## 2026-05-12 补充说明

用户追问事件锚点来源后，本模型卡降级为“候选锚点上的车辆诊断材料”。它不能作为最终强车辆基线结论，也不能支撑进入连续风格或生理有效性验证。正式流程需要先完成人工事件标注审查包，并在 `manual_verified` 锚点上重跑车辆基线。

## 模型定位

`rbf_krr_vehicle_no_subject` 是当前阶段 3 最干净的强车辆候选：只使用车辆历史统计、道路曲率/事件元信息，不使用 `subject`，不使用连续风格、生理或脑电，也不使用 old v400/raw dynamic 作为主锚点。

## 关键测试结果

pre2 + session-level test：

    window_config_id      split_strategy split                 model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse
pre2_label2_old_main session_level_split  test rbf_krr_vehicle_no_subject         67    0.382337                 0.820896         0.179104               0.545455                           1.08385               0.283582         0.543657             0.379007                         0.029851              0.642092

pre3 + session-level test：

             window_config_id      split_strategy split                 model_name  n_samples  rmse_steer  peak_direction_accuracy  wrong_side_rate  large_response_recall  peak_amp_ratio_pred_over_gt_mean  severe_amp_under_rate  peak_time_mae_s  tail_abs_error_mean  reversal_count_exact_match_rate  difficult_top20_rmse
pre3_label3_response_coverage session_level_split  test rbf_krr_vehicle_no_subject         67    0.466957                 0.791045         0.208955               0.470588                          0.741498               0.358209         0.745224             0.392508                              0.0              0.832563

## 分被试风险

    window_config_id subject  n_samples  sample_rmse_mean  sample_rmse_median  gt_peak_abs_mean  wrong_side_rate  severe_under_rate  difficult_rate  tail_abs_error_mean  reversal_exact_rate
pre2_label2_old_main     zdq          5          0.460148            0.234952          0.720124         0.400000           0.400000        0.200000             0.647621             0.000000
pre2_label2_old_main      gf          5          0.400224            0.198316          0.836396         0.200000           0.600000        0.400000             0.480156             0.000000
pre2_label2_old_main     yyl          5          0.375660            0.398027          0.675967         0.400000           0.200000        0.400000             0.528910             0.000000
pre2_label2_old_main     yzy          5          0.318205            0.287381          0.735935         0.200000           0.200000        0.600000             0.475178             0.000000
pre2_label2_old_main     byx         15          0.305639            0.249005          0.664657         0.133333           0.266667        0.333333             0.282104             0.000000
pre2_label2_old_main     hzh          9          0.259464            0.215232          0.665746         0.111111           0.555556        0.222222             0.411238             0.222222
pre2_label2_old_main     lxy          5          0.235466            0.216717          0.697992         0.000000           0.400000        0.200000             0.410413             0.000000
pre2_label2_old_main     txj          5          0.224055            0.168166          0.456053         0.200000           0.200000        0.200000             0.383133             0.000000
pre2_label2_old_main      zx         10          0.223966            0.159649          0.485288         0.200000           0.000000        0.200000             0.244583             0.000000
pre2_label2_old_main     zxy          3          0.099688            0.090249          0.634253         0.000000           0.000000        0.000000             0.129300             0.000000

## pre2 幅值桶

    window_config_id      group_name group_value  n_samples  sample_rmse_mean  sample_rmse_median  wrong_side_rate  severe_under_rate  large_response_recall  peak_amp_ratio_mean  tail_abs_error_mean  reversal_exact_rate
pre2_label2_old_main gt_peak_abs_bin      0-0.25         18          0.178247            0.147334         0.277778           0.000000               0.000000             2.237193             0.298482             0.000000
pre2_label2_old_main gt_peak_abs_bin    0.25-0.5         17          0.175364            0.168166         0.294118           0.294118               0.000000             0.780405             0.225316             0.058824
pre2_label2_old_main gt_peak_abs_bin     0.5-1.0         17          0.245417            0.241272         0.058824           0.352941               0.294118             0.645683             0.191045             0.058824
pre2_label2_old_main gt_peak_abs_bin     1.0-2.0         13          0.507067            0.514690         0.000000           0.461538               0.538462             0.577677             0.722606             0.000000
pre2_label2_old_main gt_peak_abs_bin       >=2.0          2          1.286833            1.286833         0.500000           1.000000               0.000000             0.297589             1.774405             0.000000

## pre3 幅值桶

             window_config_id      group_name group_value  n_samples  sample_rmse_mean  sample_rmse_median  wrong_side_rate  severe_under_rate  large_response_recall  peak_amp_ratio_mean  tail_abs_error_mean  reversal_exact_rate
pre3_label3_response_coverage gt_peak_abs_bin      0-0.25          9          0.159061            0.153808         0.222222           0.111111               0.000000             1.465500             0.165950                  0.0
pre3_label3_response_coverage gt_peak_abs_bin    0.25-0.5         20          0.194153            0.164282         0.300000           0.150000               0.000000             0.809267             0.188344                  0.0
pre3_label3_response_coverage gt_peak_abs_bin     0.5-1.0         21          0.281988            0.247177         0.047619           0.523810               0.000000             0.549529             0.335056                  0.0
pre3_label3_response_coverage gt_peak_abs_bin     1.0-2.0         15          0.630930            0.583973         0.266667           0.533333               0.466667             0.515540             0.630613                  0.0
pre3_label3_response_coverage gt_peak_abs_bin       >=2.0          2          1.559186            1.559186         0.500000           0.500000               0.500000             0.516138             2.271106                  0.0

## 图表

- 固定样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre2_session_v0_4.png`
- 坏样本轨迹图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre2_session_v0_4.png`
- 长窗口固定样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_fixed_predictions_pre3_session_v0_4.png`
- 长窗口坏样本图：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/03_baselines/stage03_rbf_krr_model_card_v0_4/figures/stage03_rbf_krr_bad_samples_pre3_session_v0_4.png`

## 当前判断

这个模型可以暂时作为阶段 3 的强车辆参照，但仍不能开启风格/生理有效性结论。原因是：当前样本只覆盖低泄漏道路曲率事件；pre3 长窗口仍需要确认尾段和大幅响应；反向修正 exact rate 很低，说明结构化响应问题还没有解决。下一步应继续在阶段 3 内完成长窗口和物理错误复核，或者构建更明确的响应关键点/分解车辆模型。
