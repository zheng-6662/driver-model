# Stage 7d 用户查看版：非 oracle 候选选择器 v0.2

## 这个阶段为什么做

Stage 7c 已经证明候选池有 oracle 上限，但 oracle 不能部署。这个阶段专门检查：只用事件前信息和候选预测本身的特征，能不能在不看 test 标签的情况下，学会什么时候不要用 RBF/KNN。

## 这个阶段检查了什么

- 候选：RBF/KNN、keypoint residual、top-K branch0/1/2。
- 输入特征：道路/事件上下文、top-K 概率、候选轨迹自身的峰值/方向/反向修正/分散度等。
- 禁止输入：sample RMSE、真实标签、oracle winner、错侧率、困难标签、subject ID、session ID、生理、脑电、连续风格。
- 训练/选择：train 训练 classifier，val 选择策略，test 只报告。

## 目前发现了什么

- val 选择策略：`always_rbf_reference`。
- val 上该策略 RMSE=0.571482，RBF/KNN val RMSE=0.571482。
- test 上该策略 RMSE=0.533667，RBF/KNN test RMSE=0.533667，delta=+0.000000。
- test 上该策略选择 RBF/KNN 的比例=1.000。
- gate=no_upgrade。val gate 没有发现比 RBF/KNN 更可靠的非 oracle 策略。

## val 选择表

```text
                                     model_name  rmse_steer  rmse_delta_vs_rbf  wrong_side_rate  large_response_recall  rbf_selected_rate  oracle_match_rate  selected_by_val_gate
  rf_depth3_balanced__fallback_rbf_conf_lt_0.35    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth3_balanced__fallback_rbf_conf_lt_0.45    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth3_balanced__fallback_rbf_conf_lt_0.55    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth3_balanced__fallback_rbf_conf_lt_0.65    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth3_balanced__fallback_rbf_conf_lt_0.75    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth4_balanced__fallback_rbf_conf_lt_0.35    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth4_balanced__fallback_rbf_conf_lt_0.45    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth4_balanced__fallback_rbf_conf_lt_0.55    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth4_balanced__fallback_rbf_conf_lt_0.65    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
  rf_depth4_balanced__fallback_rbf_conf_lt_0.75    0.571482           0.000000         0.119048                    0.5           1.000000           0.357143                     0
logreg_balanced_c0_2__fallback_rbf_conf_lt_0.35    0.575926           0.004444         0.095238                    0.5           0.547619           0.261905                     0
logreg_balanced_c0_2__fallback_rbf_conf_lt_0.45    0.579446           0.007964         0.119048                    0.5           0.833333           0.309524                     0
```

## test 对照表

```text
              model_name  rmse_steer  rmse_delta_vs_rbf  wrong_side_rate  large_response_recall  difficult_top20_rmse
    always_rbf_reference    0.533667           0.000000            0.225                  0.750              0.678907
    topk_top1_non_oracle    0.587865           0.054198            0.100                  0.750              0.717094
broad_oracle_upper_bound    0.410957          -0.122710            0.075                  0.875              0.604369
```

## 哪些结果可信

可信的是：这个选择器没有使用 test 标签做选择，也没有使用 subject ID、生理、脑电或连续风格。它回答的是“候选池的 oracle 上限能否被当前非 oracle 特征转化为可部署收益”。

## 哪些结果还不能下结论

如果 gate 仍然是 no_upgrade，就不能说多假设路线已经超过 RBF/KNN；也不能据此进入生理/EEG 有效性结论。oracle 上限仍然只能说明潜力，不是部署性能。

## 下一阶段是否可以继续

可以继续，但如果本轮仍没有超过 RBF/KNN，下一步应转向候选生成方式本身：让候选显式覆盖方向、幅值、峰值时间、尾段和多段修正，而不是继续只堆选择器。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_policy_metrics_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_validation_rmse_delta.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/figures/stage07d_selected_choice_counts.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_gate_table.csv`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07d_non_oracle_selector_v0_2/tables/stage07d_selected_policy_decisions.csv`
