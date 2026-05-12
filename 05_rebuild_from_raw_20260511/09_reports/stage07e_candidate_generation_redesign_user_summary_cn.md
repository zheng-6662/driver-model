# Stage 7e 用户查看版：候选生成重设计审计 v0.1

## 这个阶段为什么做

Stage 7c 说明候选池有 oracle 上限，Stage 7d 说明当前非 oracle selector 学不会稳定选择。因此下一步不能继续只堆 selector，而要先检查候选本身应该怎样生成，才能覆盖真实失稳响应。

## 这个阶段检查了什么

- 从真实方向盘标签里提取响应类型：方向、幅值、峰值时间、尾段模式、反向修正/多段修正。
- 用已有候选轨迹计算每个响应类型下的 RBF/KNN 误差、候选 oracle 误差、oracle gain 和候选胜出比例。
- 把缺口分成两类：候选池有信号但 selector 不会选；候选生成本身还不够。
- 输出下一版候选生成蓝图，不训练新模型。

## 目前发现了什么

- test RBF/KNN RMSE=0.533667。
- test deployable candidate oracle RMSE=0.410957，平均样本 gain=0.108576。
- test 中 oracle 选择非 RBF/KNN 候选的比例=0.700。
- 这说明当前候选池不是完全无效，但 Stage 7d 已经证明当前 selector 不能可靠使用它。

test 覆盖状态统计：

```text
                       coverage_status  n
selector_gap_candidate_pool_has_signal 16
             low_n_interpret_carefully 15
                             mixed_gap  1
```

## 下一版候选生成优先级

```text
           bucket_type                         bucket_value  n_samples  rbf_rmse  deployable_oracle_rmse  mean_gain_over_rbf  non_rbf_oracle_rate                        coverage_status  priority_score                                                               recommended_action
road_design_risk_class                 design_curve_context          9  0.676857                0.387491            0.252579             0.777778 selector_gap_candidate_pool_has_signal        0.534424          use response-factorized candidate generation and non-oracle calibration
       response_family small|return_near_zero|multi_segment          7  0.519285                0.361176            0.152559             0.857143 selector_gap_candidate_pool_has_signal        0.400267                                 add reversal/multi-segment candidate constructor
        direction_mode                             positive         17  0.605298                0.431250            0.144638             0.823529 selector_gap_candidate_pool_has_signal        0.381663                       add direction-conditioned candidates with wrong-side guard
road_design_risk_class             design_high_risk_surface          5  0.449456                0.323927            0.120189             1.000000 selector_gap_candidate_pool_has_signal        0.380283          use response-factorized candidate generation and non-oracle calibration
        amplitude_mode                                small         13  0.422745                0.291643            0.120952             0.769231 selector_gap_candidate_pool_has_signal        0.335275    add amplitude-quantile candidates and explicit severe-underprediction penalty
road_design_risk_class         design_special_event_segment          6  0.427812                0.361793            0.074407             1.000000 selector_gap_candidate_pool_has_signal        0.311611          use response-factorized candidate generation and non-oracle calibration
             tail_mode                       same_side_tail          5  0.287361                0.184273            0.100894             0.800000 selector_gap_candidate_pool_has_signal        0.311341 add tail-mode candidates: return, same-side persistence, opposite-side overshoot
       correction_mode                        multi_segment         38  0.545413                0.419443            0.112605             0.710526 selector_gap_candidate_pool_has_signal        0.311013                                 add reversal/multi-segment candidate constructor
```

## 哪些结果可信

可信的是：这一步只使用 Stage 7c 已导出的候选轨迹和真实标签做离线审计，没有训练模型，没有用生理/脑电/连续风格，也没有读取服务器凭据。它给出的不是性能提升，而是下一版车辆-only 多候选模型应该覆盖哪些物理响应类型。

## 哪些结果还不能下结论

不能说多假设已经可部署有效；不能说生理或 EEG 可以进入；不能把候选 oracle 当作模型结果。Stage 7e 只说明“下一步该怎样重新生成候选”。

## 下一阶段是否可以继续

可以继续 Stage 7，但下一步应按候选生成蓝图实现 response-factorized candidates：方向/幅值、峰值时间、尾段模式、反向修正/多段修正、可靠性门控。RBF/KNN 必须继续作为固定主参照。

## 推荐优先查看

1. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_oracle_gain_by_response_family_test.png`
2. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_oracle_winner_distribution_test.png`
3. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_candidate_gap_scatter_test.png`
4. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/figures/stage07e_candidate_generation_blueprint.png`
5. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_candidate_generation_blueprint.csv`
6. `F:/data_set_process/data_process/05_rebuild_from_raw_20260511/07_multihypothesis/stage07e_candidate_generation_redesign_v0_1/tables/stage07e_gate_table.csv`
