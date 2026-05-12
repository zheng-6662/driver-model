# Stage 7i 用户查看版：稳定性校准候选选择 v0.1

## 这个阶段为什么做

Stage 7h 发现 Stage 7g 的问题不是候选完全没用，而是 validation 按最小 RMSE 选到了 test 上退化的候选。Stage 7i 不训练新模型，只用 train/val 重新设计更保守的选择规则，检查能否选出更稳定的车辆-only 候选。

## 这个阶段检查了什么

- 原始 Stage 7g val-best 规则。
- train/val 稳定性惩罚规则：`val_delta + 0.5 * abs(train_delta - val_delta)`。
- 更强惩罚、困难样本和物理指标加权规则。
- 每个规则只能用 train/val 信息选候选，test 只做最终报告。

## 目前发现了什么

- 当前推荐继续观察的规则：`stability_penalty_l05`。
- 该规则选中的候选：`segment_resid_rf_blend_25`。
- test RMSE=0.528046，相对 RBF/KNN delta=-0.005620。
- difficult RMSE delta=-0.029588，wrong-side=0.225，large recall=0.750。
- gate=weak_candidate_continue。

## 规则对照

```text
              policy_name            selected_model  val_rmse_delta_vs_rbf  rmse_delta_vs_rbf  wrong_side_rate  large_response_recall  difficult_rmse_delta_vs_rbf
    stability_penalty_l05 segment_resid_rf_blend_25              -0.006778          -0.005620            0.225                   0.75                    -0.029588
    stability_penalty_l10 segment_resid_rf_blend_25              -0.006778          -0.005620            0.225                   0.75                    -0.029588
stage7g_original_val_best   segment_abs_rf_blend_25              -0.010339           0.002509            0.225                   0.75                    -0.022623
       val_plus_difficult   segment_abs_rf_blend_50              -0.008304           0.017537            0.225                   0.75                    -0.035294
        val_plus_physical   segment_abs_rf_blend_50              -0.008304           0.017537            0.225                   0.75                    -0.035294
```

## 稳定性分数前十

```text
                                 model_name  rmse_delta_vs_rbf_train  rmse_delta_vs_rbf_val  score_stability_l05  rmse_delta_vs_rbf_test  difficult_rmse_delta_vs_rbf_test
                  segment_resid_rf_blend_25                 0.007625              -0.006778             0.000424               -0.005620                         -0.029588
                    segment_abs_rf_blend_25                 0.012571              -0.010339             0.001116                0.002509                         -0.022623
                  segment_resid_rf_blend_50                 0.037223              -0.003456             0.016883                0.000349                         -0.046434
         rbf_resid_keypoint_scaled_blend_50                -0.022878               0.004254             0.017820               -0.018202                         -0.046660
                    segment_abs_rf_blend_50                 0.046127              -0.008304             0.018912                0.017537                         -0.035294
           rbf_abs_keypoint_scaled_blend_50                -0.015284               0.014358             0.029180               -0.007153                         -0.029840
                  rbf_resid_keypoint_scaled                -0.018835               0.017149             0.035141               -0.025129                         -0.077768
                    rbf_abs_keypoint_scaled                -0.000449               0.040631             0.061172                0.004112                         -0.043671
topk_vehicle_transformer_branch0_no_subject                 0.098065               0.049536             0.073801                0.021422                          0.019291
                 segment_resid_rf_piecewise                 0.143176               0.032310             0.087743                0.044930                         -0.038628
```

## 逐样本收益

```text
split  n_samples  mean_gain  median_gain  positive_gain_rate
 test         40   0.005366     0.003297            0.550000
train        188  -0.005491     0.000215            0.505319
  val         42   0.006947     0.007961            0.595238
```

## 哪些结果可信

可信的是：这个规则没有看 test 标签来选候选，选择只来自 train/val 的稳定性分数；它比原始 val-best 规则更符合 Stage 7h 暴露出的风险。

## 哪些结果还不能下结论

这还不能直接升为最终主线。原因是目前只有一个固定 session-level split，没有多折验证；收益主要是 RMSE 和困难样本 RMSE，错侧率和大幅响应召回没有进一步改善。它只能作为“弱候选继续验证”。

## 下一阶段是否可以继续

可以继续做 Stage 7j：对该稳定性校准规则做多折 session validation 或重新构建分层 validation，再决定是否把它冻结为车辆-only 主候选。生理/EEG 仍不进入。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/figures/stage07i_policy_summary.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/figures/stage07i_stability_score_components.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/figures/stage07i_selected_gain_distribution.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_policy_test_summary.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07i_stability_calibrated_selection_v0_1/tables/stage07i_gate_table.csv`
