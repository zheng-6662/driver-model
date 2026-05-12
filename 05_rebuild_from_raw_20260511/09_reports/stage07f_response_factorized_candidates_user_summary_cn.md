# Stage 7f 用户查看版：response-factorized 车辆-only 原型候选 v0.1

## 这个阶段为什么做

Stage 7e 判断不能继续只调 selector，而要让候选生成本身覆盖方向、幅值、峰值时间、尾段和反向/多段修正。这个阶段先做一个轻量版本：用 train split 的真实响应类型建立原型轨迹，再用事件前特征预测响应类型，生成车辆-only 候选。

## 这个阶段检查了什么

- 输入仍然只用事件前车辆/道路/事件上下文和已有候选预测形态特征。
- 禁止使用 subject ID、session ID、test 标签、生理、脑电、连续风格。
- 原型轨迹只从 train split 估计。
- val 选择策略，test 只报告。

## 目前发现了什么

- val 选择策略：`rbf_kernel_ridge_context_no_subject`。
- test 上该策略 RMSE=0.533667，RBF/KNN RMSE=0.533667，delta=+0.000000。
- response-factorized oracle RMSE=0.440217。
- response-factorized + existing candidates oracle RMSE=0.388119。
- gate=no_upgrade。

## 响应类型预测质量

```text
         factor split  accuracy  balanced_accuracy  mean_confidence
 direction_mode  test  0.925000           0.919437         0.849657
 direction_mode   val  0.904762           0.916667         0.879187
 amplitude_mode  test  0.475000           0.495951         0.466405
 amplitude_mode   val  0.500000           0.449020         0.491323
    peak_timing  test  0.650000           0.661616         0.637007
    peak_timing   val  0.595238           0.576389         0.615503
      tail_mode  test  0.825000           0.380392         0.814633
      tail_mode   val  0.785714           0.305556         0.786491
correction_mode  test  0.950000           0.657895         0.725166
correction_mode   val  0.880952           0.316239         0.695280
```

## val 策略选择表

```text
                                      model_name  rmse_steer  rmse_delta_vs_rbf  wrong_side_rate  large_response_recall  selected_by_val_gate
 proto_amplitude__fallback_rbf_conf_prod_lt_0.25    0.571482                0.0         0.119048                    0.5                     0
 proto_amplitude__fallback_rbf_conf_prod_lt_0.32    0.571482                0.0         0.119048                    0.5                     0
 proto_amplitude__fallback_rbf_conf_prod_lt_0.40    0.571482                0.0         0.119048                    0.5                     0
 proto_amplitude__fallback_rbf_conf_prod_lt_0.50    0.571482                0.0         0.119048                    0.5                     0
proto_combo_full__fallback_rbf_conf_prod_lt_0.25    0.571482                0.0         0.119048                    0.5                     0
proto_combo_full__fallback_rbf_conf_prod_lt_0.32    0.571482                0.0         0.119048                    0.5                     0
proto_combo_full__fallback_rbf_conf_prod_lt_0.40    0.571482                0.0         0.119048                    0.5                     0
proto_combo_full__fallback_rbf_conf_prod_lt_0.50    0.571482                0.0         0.119048                    0.5                     0
proto_correction__fallback_rbf_conf_prod_lt_0.25    0.571482                0.0         0.119048                    0.5                     0
proto_correction__fallback_rbf_conf_prod_lt_0.32    0.571482                0.0         0.119048                    0.5                     0
proto_correction__fallback_rbf_conf_prod_lt_0.40    0.571482                0.0         0.119048                    0.5                     0
proto_correction__fallback_rbf_conf_prod_lt_0.50    0.571482                0.0         0.119048                    0.5                     0
```

## 哪些结果可信

可信的是：这一版严格 train-only 建原型、val 选策略、test 报告；没有用生理/脑电/风格，也没有读取服务器凭据。它能判断“响应类型原型候选”这一方向是否值得继续。

## 哪些结果还不能下结论

不能把 oracle 当作可部署性能；如果 selected 策略没有超过 RBF/KNN，就不能说多假设已经解决。即便 oracle 好，也只能说明下一版需要更强的非 oracle 选择和候选生成。

## 下一阶段是否可以继续

如果 gate 仍是 no_upgrade，下一步不要进入生理/EEG；应把 response-factorized 原型升级成可训练关键点/分段候选模型，重点提升响应类型预测与候选选择。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/figures/stage07f_metric_summary_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/figures/stage07f_fixed_predictions_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/figures/stage07f_oracle_gain_predictions_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_gate_table.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07f_response_factorized_candidates_v0_1/tables/stage07f_factor_prediction_metrics.csv`
