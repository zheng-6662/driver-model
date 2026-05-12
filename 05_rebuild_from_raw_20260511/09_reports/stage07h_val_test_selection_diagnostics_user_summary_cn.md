# Stage 7h 用户查看版：val/test 选择不稳定诊断 v0.1

## 这个阶段为什么做

Stage 7g 出现了一个必须解释的现象：val gate 选择的 `segment_abs_rf_blend_25` 在 test 上没有超过 RBF/KNN，但另一个 test-only 候选 `rbf_resid_keypoint_scaled` 明显更好。这个阶段不训练新模型，只诊断“为什么 val 选不中 test 上好的候选”。

## 这个阶段检查了什么

- 候选在 train/val/test 的 RMSE delta 和排名是否稳定。
- val 和 test 的响应类型、道路/事件上下文、候选置信度分布是否有偏移。
- selected 候选和 test-only 最好候选在逐样本、分响应类型和分道路 bucket 上的收益是否一致。
- 关键点回归的 val/test 误差是否一致。

## 目前发现了什么

- RBF/KNN test RMSE=0.533667。
- val gate selected=`segment_abs_rf_blend_25`，selected test RMSE=0.536176，delta=+0.002509，不能升级。
- test-only 最好非 oracle 候选=`rbf_resid_keypoint_scaled`，test delta=-0.025129；该结果只能作为诊断，因为它没有被 val gate 选中。
- keypoint/segment oracle test RMSE=0.462003，说明候选空间仍有上限，但选择/校准还没解决。

## 候选稳定性摘要

```text
                         model_name  rmse_delta_vs_rbf_val  rmse_delta_vs_rbf_test  val_rmse_rank  test_rmse_rank  val_test_delta_swing         diagnostic_status
          rbf_resid_keypoint_scaled               0.017149               -0.025129            8.0             1.0             -0.042278 test_best_diagnostic_only
 rbf_resid_keypoint_scaled_blend_50               0.004254               -0.018202            6.0             2.0             -0.022456           other_candidate
          segment_resid_rf_blend_25              -0.006778               -0.005620            3.0             4.0              0.001157           other_candidate
rbf_kernel_ridge_context_no_subject               0.000000                0.000000            5.0             5.0              0.000000           other_candidate
            segment_abs_rf_blend_25              -0.010339                0.002509            1.0             7.0              0.012848      selected_by_val_gate
```

## val/test 分布偏移摘要

```text
                feature  n_values  js_val_test  js_train_val  js_train_test          largest_val_test_shift_value  largest_test_minus_val_prop
        response_family        15     0.073070      0.037517       0.053480 medium|return_near_zero|multi_segment                     0.117857
road_design_module_name         7     0.039354      0.031620       0.022907                        middle_section                     0.070238
            event_level         7     0.039044      0.024849       0.021206                        extreme_active                    -0.130952
        correction_mode         4     0.027077      0.005836       0.016434                          single_sweep                     0.025000
         direction_mode         2     0.015523      0.007730       0.001355                              negative                     0.146429
         amplitude_mode         4     0.011975      0.006924       0.012955                                medium                     0.117857
 road_design_risk_class         4     0.009984      0.018516       0.009072                   design_regular_road                     0.095238
            top1_branch         3     0.007958      0.007723       0.027499                                     0                     0.082143
```

```text
                feature  train_n  train_mean  train_std  train_median  val_n   val_mean    val_std  val_median  test_n  test_mean   test_std  test_median  test_minus_val_mean  test_minus_val_median  std_mean_diff_val_test_by_train_std  ks_val_test
           prob_entropy      188    1.085875   0.009229      1.086212     42   1.083432   0.036054    1.090041      40   1.084205   0.009307     1.084231             0.000774              -0.005810                             0.083846     0.300000
        gt_onset_time_s      188    0.481809   0.463901      0.290000     42   0.461429   0.551690    0.245000      40   0.277500   0.286666     0.152500            -0.183929              -0.092500                            -0.396483     0.261905
 branch_peak_abs_spread      188    0.567431   0.251750      0.629370     42   0.533378   0.243883    0.501264      40   0.550262   0.188346     0.572696             0.016884               0.071432                             0.067068     0.251190
topk_branch_spread_peak      188    0.365917   0.078207      0.376520     42   0.350743   0.064876    0.354894      40   0.374994   0.077232     0.389934             0.024251               0.035040                             0.310091     0.247619
      anchor_time_rel_s      188  307.768830 171.769480    294.930000     42 336.062857 161.428960  326.695000      40 271.373375 163.428803   234.570000           -64.689482             -92.125000                            -0.376606     0.240476
            prob_margin      188    0.035518   0.028349      0.029844     42   0.037386   0.040611    0.031431      40   0.029753   0.025892     0.021660            -0.007633              -0.009772                            -0.269248     0.239286
              top1_prob      188    0.382759   0.022386      0.381931     42   0.381065   0.038501    0.378996      40   0.383178   0.020273     0.380909             0.002113               0.001913                             0.094367     0.226190
topk_branch_spread_mean      188    0.208208   0.046367      0.214166     42   0.201948   0.039793    0.207770      40   0.210640   0.043099     0.223290             0.008692               0.015520                             0.187463     0.216667
```

## 逐样本收益摘要

```text
          candidate_model split  n_samples  mean_gain  positive_gain_rate
rbf_resid_keypoint_scaled  test         40   0.025016            0.650000
rbf_resid_keypoint_scaled train        188   0.022059            0.585106
rbf_resid_keypoint_scaled   val         42  -0.013395            0.404762
  segment_abs_rf_blend_25  test         40  -0.002277            0.525000
  segment_abs_rf_blend_25 train        188  -0.006574            0.494681
  segment_abs_rf_blend_25   val         42   0.006409            0.619048
```

## 哪些结果可信

可信的是：Stage 7h 没有训练新模型，也没有使用生理、脑电、连续风格、subject ID 或服务器凭据；它只复核 Stage 7g 已有候选在不同 split 上的稳定性。

## 哪些结果还不能下结论

不能把 `rbf_resid_keypoint_scaled` 当成新主线，因为它是按 test 表现事后发现的。只有未来用 train/val 规则稳定选中它或同类策略，并在 test 上仍超过 RBF/KNN，才能升级。

## 下一阶段是否可以继续

下一步应先做候选选择校准或验证集重构，例如多折 session validation、按 response bucket/道路模块分层的 val gate、关键点不确定性评分。仍不应进入生理/EEG。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_candidate_val_test_stability.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_val_test_categorical_shift.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_candidate_gain_by_split.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/figures/stage07h_keypoint_target_rmse_by_split.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07h_val_test_selection_diagnostics_v0_1/tables/stage07h_gate_table.csv`
