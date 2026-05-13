# Stage 7j 用户查看版：session 多折稳定性验证 v0.1

## 这个阶段为什么做

Stage 7i 只在一个固定 split 上显示 `segment_resid_rf_blend_25` 有弱收益。这个结果不能直接升级，因为可能只是当前 validation/test 划分偶然有利。Stage 7j 用 session 分组做 5 折复核，检查稳定选择规则是否能跨 session 复现。

## 这个阶段检查了什么

- 每一折重新训练 RBF/KNN 基座，避免把固定 split 的 RBF 预测直接搬到新折里。
- 只允许事件前车辆/道路上下文，以及该折重训 RBF 得到的预测形态特征。
- 明确排除固定 split 训练出来的 top-K/Transformer/keypoint 预测特征，因为它们在新折里会有训练信息泄漏风险。
- 用 `stability_penalty_l05` 和原始 Stage 7g val gate 做对照，test 只做最终报告。

## 目前发现了什么

- gate=no_upgrade。
- `stability_penalty_l05` 平均 test delta vs fold RBF=+0.000329。
- improved fold rate=0.600。
- difficult improved fold rate=0.800。
- 选中的模型集合：rbf_abs_keypoint_scaled_blend_50, rbf_resid_keypoint_scaled, segment_abs_rf_blend_25, segment_resid_rf_blend_25。

## 多折汇总

```text
              policy_name  n_folds  mean_test_rmse  mean_test_delta_vs_rbf  median_test_delta_vs_rbf  std_test_delta_vs_rbf  improved_fold_count  improved_fold_rate  mean_wrong_side_delta  mean_large_recall_delta  mean_difficult_delta  difficult_improved_fold_rate                                                                                                 selected_models
stage7g_original_val_gate        5        0.624985               -0.004170                 -0.009860               0.009537                    3                 0.6               0.014815                 0.154762             -0.019418                           0.8                          rbf_resid_keypoint_scaled, rbf_resid_keypoint_scaled_blend_50, segment_abs_rf_blend_25
     always_rbf_reference        5        0.629156                0.000000                  0.000000               0.000000                    0                 0.0               0.000000                 0.000000              0.000000                           0.0                                                                             rbf_kernel_ridge_context_no_subject
    stability_penalty_l05        5        0.629485                0.000329                 -0.011193               0.016275                    3                 0.6               0.011111                 0.126190             -0.014632                           0.8 rbf_abs_keypoint_scaled_blend_50, rbf_resid_keypoint_scaled, segment_abs_rf_blend_25, segment_resid_rf_blend_25
```

## gate

```text
                 gate_item                            status                                                                                                                        evidence
       cv_feature_protocol strict_retrained_rbf_context_only          RBF was retrained per fold; fixed-split topK/Transformer candidate-prediction features were excluded to avoid leakage.
stability_policy_cv_result                        no_upgrade                                         mean test delta=+0.000329; improved fold rate=0.600; difficult improved fold rate=0.800
          mainline_upgrade                         not_final Even a positive CV result would still need full upstream candidate retraining and fixed-plot review before freezing a mainline.
stage08_physio_eeg_allowed                           blocked                           Vehicle-only candidate stability is still under validation; no physio/EEG evidence is evaluated here.
               server_used                                no                                                                        Local CPU diagnostic run only; credential file not read.
```

## 哪些结果可信

可信的是：这轮没有用生理、脑电、连续风格、驾驶员 ID，也没有用固定 split 的 top-K/Transformer 预测特征；RBF 基座每折重训，标准化和特征选择只用对应 train split。

## 哪些结果还不能下结论

这仍然不是完整最终主线验证。因为 Stage7i 原模型使用过固定 split 的候选预测特征，而本轮为了避免泄漏只保留了重训 RBF 形态特征；因此 Stage7j 更像严格稳定性审计，而不是完整复刻所有上游候选模型。

## 下一阶段是否可以继续

如果 gate 仍是 `no_upgrade`，应回到候选生成或 split 设计，不进入生理/EEG。如果 gate 是 `weak_candidate_continue`，也只能进入更完整的上游重训/固定图复核，不能直接宣称最终升级。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/figures/stage07j_policy_fold_deltas.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/figures/stage07j_selected_model_counts.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/figures/stage07j_candidate_val_test_delta_scatter.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_policy_aggregate.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07j_session_cv_stability_v0_1/tables/stage07j_gate_table.csv`
