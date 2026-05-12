# 阶段 4：连续风格路线收口决策 v0.1

## 输入证据

- 协议 gate：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_continuous_style_protocol_v0_1/tables/style_protocol_gate_table.csv`
- session-level 探索指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_increment_exploratory_v0_1/tables/style_increment_metrics.csv`
- cross-split 指标：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_metrics.csv`
- cross-split gate：`F:/data_set_process/data_process/05_rebuild_from_raw_20260511/04_style/stage04_style_cross_split_validation_v0_1/tables/style_cross_split_gate_table.csv`

## 证据摘要

```text
     split_strategy                                          model_name              role_cn  n_test_samples  rbf_rmse  model_rmse  delta_rmse_vs_rbf  relative_rmse_change_pct_vs_rbf  rbf_wrong_side_rate  model_wrong_side_rate  delta_wrong_side_rate  rbf_large_response_recall  model_large_response_recall  delta_large_response_recall  rbf_difficult_top20_rmse  model_difficult_top20_rmse  delta_difficult_top20_rmse  rbf_reversal_exact  model_reversal_exact  delta_reversal_exact                                interpretation_cn
session_level_split         rbf_plus_style_last60_guard3_residual_ridge          连续风格 last60              40  0.533667    0.534559           0.000892                         0.167116             0.225000               0.225000               0.000000                   0.750000                     0.750000                          0.0                  0.678907                    0.680891                    0.001984                 0.0                   0.0                   0.0                               未超过 RBF，不能作为有效性证据。
session_level_split           rbf_plus_style_all_windows_residual_ridge             连续风格全部窗口              40  0.533667    0.564143           0.030476                         5.710717             0.225000               0.175000              -0.050000                   0.750000                     0.750000                          0.0                  0.678907                    0.702179                    0.023273                 0.0                   0.0                   0.0            全部窗口在 session-level 明显变差，说明堆更多风格特征不稳。
session_level_split                   rbf_plus_driver_id_residual_ridge            驾驶员 ID 对照              40  0.533667    0.533661          -0.000006                        -0.001083             0.225000               0.225000               0.000000                   0.750000                     0.750000                          0.0                  0.678907                    0.679859                    0.000952                 0.0                   0.0                   0.0                      驾驶员 ID 对照没有实质增益；用于排除身份代理风险。
session_level_split rbf_plus_style_last60_with_driver_id_residual_ridge 连续风格 last60 + 驾驶员 ID              40  0.533667    0.534558           0.000891                         0.166943             0.225000               0.225000               0.000000                   0.750000                     0.750000                          0.0                  0.678907                    0.680895                    0.001988                 0.0                   0.0                   0.0                风格加 ID 仍没有形成稳定收益，说明当前融合方式不足以支撑结论。
subject_level_split         rbf_plus_style_last60_guard3_residual_ridge          连续风格 last60              68  0.484847    0.483510          -0.001337                        -0.275760             0.147059               0.147059               0.000000                   0.666667                     0.666667                          0.0                  0.658887                    0.659204                    0.000317                 0.0                   0.0                   0.0                  只有很小 RMSE 改善，关键物理指标没有稳定改善，不能升级。
subject_level_split           rbf_plus_style_all_windows_residual_ridge             连续风格全部窗口              68  0.484847    0.482109          -0.002738                        -0.564699             0.147059               0.117647              -0.029412                   0.666667                     0.666667                          0.0                  0.658887                    0.655899                   -0.002989                 0.0                   0.0                   0.0 subject-level 有小改善，但 session-level 明显变差，不能算稳定路线。
subject_level_split                   rbf_plus_driver_id_residual_ridge            驾驶员 ID 对照              68  0.484847    0.484992           0.000146                         0.030024             0.147059               0.147059               0.000000                   0.666667                     0.666667                          0.0                  0.658887                    0.659192                    0.000304                 0.0                   0.0                   0.0                      驾驶员 ID 对照没有实质增益；用于排除身份代理风险。
subject_level_split rbf_plus_style_last60_with_driver_id_residual_ridge 连续风格 last60 + 驾驶员 ID              68  0.484847    0.483511          -0.001335                        -0.275411             0.147059               0.147059               0.000000                   0.666667                     0.666667                          0.0                  0.658887                    0.659211                    0.000323                 0.0                   0.0                   0.0                风格加 ID 仍没有形成稳定收益，说明当前融合方式不足以支撑结论。
```

## gate

```text
                         gate_item                status                                                                                                                       evidence                       decision_cn
         no_leakage_style_protocol         pass_protocol                                           stage04_continuous_style_protocol_v0_1 passed direct-input and label-overlap checks.                 事件前连续风格候选的来源协议可用。
         style_two_split_rmse_gain                  fail                                                                                session delta=0.000892; subject delta=-0.001337    必须两类切分都超过 RBF 才能进入有效性候选；当前不满足。
        style_physical_metric_gain                  fail session wrong/large/difficult delta=0.000000/0.000000/0.001984; subject wrong/large/difficult delta=0.000000/0.000000/0.000317        没有稳定改善错侧、大幅响应召回或困难样本，不能升级。
         style_not_driver_id_proxy             weak_pass            driver ID control is near RBF and does not explain a large style gain; however style itself also lacks stable gain.          当前不是强身份代理问题，而是风格增量本身不稳定。
  style_route_continue_as_mainline    no_go_current_form                                session-level fails; subject-level only tiny RMSE gain; physical metrics do not stably improve.             当前连续风格直接残差融合路线不升级为主线。
physio_eeg_role_validation_allowed               blocked                                                                      vehicle+style fair reference is not strong/stable enough.                 生理/EEG 有效性验证继续阻塞。
                        next_route go_vehicle_structured                                               RBF still has wrong-side, reversal, multi-segment and difficult-sample failures. 回到车辆-only 结构化轨迹建模，优先响应分解/关键点/多假设。
```

## 下一步

```text
 priority                 task                                         why_cn  allowed_now
        1 阶段 6 车辆-only 结构化轨迹建模 连续风格当前没有稳定增量，先解决车辆-only 的错侧、幅值、反向修正、多段修正和困难样本。         True
        2         固定坏样本图人工复核摘要         确认 RBF 失败到底来自事件语义、物理不可预测、多段响应还是模型结构不足。         True
        3           连续风格更强表示探索              只可作为后备探索；当前统计特征 + 残差 Ridge 不支持主线。        False
        4         生理/EEG 有效性验证                   尚未形成稳定车辆+风格公平参照，不能把生理增量归因干净。        False
```

## 决策

当前连续风格直接残差融合路线不升级为主线；生理/EEG 有效性验证继续阻塞；下一步回到车辆-only 结构化轨迹建模。