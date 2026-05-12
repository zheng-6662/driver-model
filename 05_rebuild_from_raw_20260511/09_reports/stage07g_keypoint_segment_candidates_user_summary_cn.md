# Stage 7g 用户查看版：keypoint/segment 车辆-only 候选 v0.1

## 这个阶段为什么做

Stage 7f 的响应类型原型只证明了 oracle 空间存在，但 validation gate 没有批准升级。Stage 7g 尝试更直接地预测响应关键点：主峰方向/幅值、峰值时间、启动时间和尾段值，再用这些关键点生成分段轨迹或校正 RBF/KNN 轨迹。

## 这个阶段检查了什么

- 只使用事件前车辆、道路/事件上下文和已有候选预测自身形态特征。
- 不使用 subject ID、session ID、test 标签、生理、脑电和连续风格。
- 关键点回归模型只在 train split 拟合。
- val 选择候选，test 只报告一次。

## 目前发现了什么

- val 选择策略：`segment_abs_rf_blend_25`。
- test 上 selected RMSE=0.536176，RBF/KNN RMSE=0.533667，delta=+0.002509。
- keypoint/segment oracle RMSE=0.462003，只作为诊断上限。
- gate=no_upgrade。
- test 上事后最好的非 oracle 候选是 `rbf_resid_keypoint_scaled`，RMSE=0.508538，delta=-0.025129；但它不是 val gate 选中的策略，不能作为可部署升级结论。

## 关键点预测质量

```text
model_prefix       target split     rmse      mae      bias      corr
         abs  peak_signed   val 0.757952 0.547474 -0.203269  0.845982
         abs  peak_signed  test 0.858810 0.660271 -0.349729  0.796070
         abs  peak_time_s   val 0.544119 0.417799  0.079468  0.065754
         abs  peak_time_s  test 0.421546 0.322936  0.074206  0.195029
         abs onset_time_s   val 0.570218 0.371245  0.029750 -0.058195
         abs onset_time_s  test 0.361616 0.314167  0.209032  0.238896
         abs  tail_signed   val 0.261202 0.200247  0.000715  0.092358
         abs  tail_signed  test 0.242938 0.196777  0.007812 -0.204303
       resid  peak_signed   val 0.715985 0.486161 -0.102149  0.855030
       resid  peak_signed  test 0.789667 0.543640 -0.270118  0.820470
       resid  peak_time_s   val 0.655622 0.517787  0.107814  0.124052
       resid  peak_time_s  test 0.557853 0.426311  0.040161  0.072001
       resid onset_time_s   val 0.604058 0.385686  0.130072  0.366844
       resid onset_time_s  test 0.366992 0.275653  0.168456  0.313355
       resid  tail_signed   val 0.271793 0.210021  0.006594  0.066353
       resid  tail_signed  test 0.237278 0.189408 -0.015358  0.040017
```

## val 策略选择表

```text
                                      model_name  rmse_steer  rmse_delta_vs_rbf  wrong_side_rate  large_response_recall  selected_by_val_gate
                         segment_abs_rf_blend_25    0.561143          -0.010339         0.119048                    0.5                     1
                         segment_abs_rf_blend_50    0.563179          -0.008304         0.119048                    0.6                     0
                       segment_resid_rf_blend_25    0.564705          -0.006778         0.071429                    0.5                     0
                       segment_resid_rf_blend_50    0.568026          -0.003456         0.071429                    0.5                     0
              rbf_resid_keypoint_scaled_blend_50    0.575736           0.004254         0.095238                    0.5                     0
                rbf_abs_keypoint_scaled_blend_50    0.585841           0.014358         0.095238                    0.6                     0
                       rbf_resid_keypoint_scaled    0.588631           0.017149         0.095238                    0.6                     0
keypoint_residual_vehicle_transformer_no_subject    0.598300           0.026818         0.095238                    0.5                     0
                        segment_abs_rf_piecewise    0.603113           0.031630         0.119048                    0.7                     0
                      segment_resid_rf_piecewise    0.603792           0.032310         0.071429                    0.6                     0
                         rbf_abs_keypoint_scaled    0.612114           0.040631         0.095238                    0.6                     0
     topk_vehicle_transformer_branch0_no_subject    0.621018           0.049536         0.119048                    0.3                     0
```

## 哪些结果可信

可信的是：这一轮严格使用 train 拟合关键点，val 选择候选，test 最终报告，没有引入生理/脑电/连续风格或服务器信息。它可以判断“关键点/分段候选”是否比 Stage 7f 的纯响应类型原型更有前景。

## 哪些结果还不能下结论

不能把 keypoint/segment oracle 当成可部署模型；如果 validation 选择仍退回 RBF/KNN 或 test 没有稳定提升，就不能进入生理/EEG 有效性结论。

## 下一阶段是否可以继续

如果 gate 仍是 no_upgrade，下一步应复核关键点回归误差和候选生成形态，而不是继续堆 selector。只有车辆-only 候选生成和非 oracle 选择稳定后，才适合重新评估生理/EEG。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_metric_summary_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_keypoint_target_scatter.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_fixed_predictions_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/figures/stage07g_oracle_gain_predictions_test.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07g_keypoint_segment_candidates_v0_1/tables/stage07g_gate_table.csv`
