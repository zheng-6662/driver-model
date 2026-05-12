# Stage 7a：非 oracle 多候选选择协议 v0.1

生成时间：2026-05-13 06:31

本轮不训练模型，不使用生理、EEG、连续风格或被试 ID。目标是把 Stage 7 的可部署选择规则和禁止使用信息固定下来。

## Gate

| gate                       | status                      | evidence                                                        | decision                                   |
| -------------------------- | --------------------------- | --------------------------------------------------------------- | ------------------------------------------ |
| stage07_protocol_ready     | ready_for_non_oracle_design | 候选池、允许特征、禁止信息、评价指标和固定图协议已定义。                                    | 可以进入 Stage 7 非 oracle 选择器设计，但不能直接训练生理/EEG。 |
| oracle_gap_status          | large_oracle_gap            | RBF RMSE=0.533667; broad oracle RMSE=0.375182; delta=-0.158484. | 只说明多候选上限存在，不能作为可部署结果。                      |
| deployable_selector_status | blocked                     | best deployable selector RMSE=0.533912; delta=+0.000245.        | 必须先解决选择策略和校准。                              |
| stage05_physio_eeg_allowed | blocked                     | 车辆-only 多候选选择协议刚建立，尚未完成可部署 selector。                            | 继续阻塞生理/EEG有效性结论。                           |

## Candidate Pool

| model_name                                       | candidate_role                 | rmse_steer | wrong_side_rate | large_response_recall |
| ------------------------------------------------ | ------------------------------ | ---------- | --------------- | --------------------- |
| keypoint_residual_vehicle_transformer_no_subject | keypoint_candidate             | 0.548994   | 0.125000        | 0.875000              |
| ridge_rich_context_no_subject                    | low_variance_vehicle_candidate | 0.536450   | 0.175000        | 0.500000              |
| ridge_rich_history_no_subject                    | low_variance_vehicle_candidate | 0.538776   | 0.175000        | 0.500000              |
| knn_template_context_no_subject                  | template_candidate             | 0.625829   | 0.175000        | 0.750000              |
| peak_scaled_template_context_no_subject          | template_candidate             | 0.688597   | 0.200000        | 0.875000              |
| direction_gated_knn_template_no_subject          | template_candidate             | 0.693497   | 0.200000        | 0.875000              |
| topk_vehicle_transformer_branch0_no_subject      | topk_branch_candidate          | 0.555098   | 0.050000        | 0.750000              |
| topk_vehicle_transformer_branch1_no_subject      | topk_branch_candidate          | 0.589604   | 0.025000        | 0.750000              |
| topk_vehicle_transformer_branch2_no_subject      | topk_branch_candidate          | 0.685254   | 0.150000        | 0.625000              |
| rbf_kernel_ridge_context_no_subject              | current_primary_reference      | 0.533667   | 0.225000        | 0.750000              |

## Feature Guard

| feature_group              | status                     | rule_cn                                      |
| -------------------------- | -------------------------- | -------------------------------------------- |
| event_context              | allowed                    | 事件/道路上下文来自样本 manifest，可用于 train/val/test 推理。 |
| vehicle_history_summary    | allowed                    | 只能来自事件锚点之前输入窗口；禁止使用标签窗口。                     |
| candidate_prediction_shape | allowed_with_train_val_fit | 可由候选预测自身计算；选择器训练和阈值只允许看 train/val。           |
| candidate_disagreement     | allowed_with_train_val_fit | 不需要真实标签，适合作为不确定性和候选多样性特征。                    |
| calibration_prior          | allowed_train_val_only     | 只能用 train/val 统计，测试集不能参与可靠性表或标准化。            |
| oracle_winner              | forbidden                  | 只能用于分析上限，禁止作为训练标签以外的测试决策依据。                  |
| test_sample_rmse           | forbidden                  | 禁止进入选择器输入、阈值选择或可部署决策。                        |
| physio_eeg_style           | blocked                    | 车辆-only Stage 7 选择策略未闭环前继续阻塞。                |
| subject_id                 | blocked                    | 当前不允许用驾驶员 ID 解决选择问题。                         |

## Selection Steps

| step_id | step_name                   | status                   | requirement_cn                                                                   |
| ------- | --------------------------- | ------------------------ | -------------------------------------------------------------------------------- |
| S7A-0   | freeze_inputs               | required_before_training | 固定 Stage 6e 候选池、B_response3s_strict_core split 和 RBF/KNN 主参照。                    |
| S7A-1   | candidate_prediction_export | pending                  | 为每个候选保存同一 sample_id 的预测轨迹、候选形态特征和候选间差异。                                          |
| S7A-2   | train_val_selector          | pending                  | 只用 train 拟合选择器，只用 val 选模型/阈值/温度缩放；test 只最终评估一次。                                  |
| S7A-3   | calibration                 | pending                  | 报告选择置信度分桶、ECE/Brier、coverage-risk 曲线和 abstain/fallback 到 RBF 的策略。                |
| S7A-4   | top1_vs_fallback            | pending                  | 同时报告 top-1 selector、RBF fallback、abstain-on-low-confidence 和 oracle upper bound。 |
| S7A-5   | fixed_plots                 | pending                  | 固定样本图、bad samples 图、oracle-gap 样本图、selector-regret 样本图必须全部输出。                    |
| S7A-6   | promotion_gate              | pending                  | 只有 test RMSE 不劣于 RBF，且至少一个物理指标或困难样本改善，才允许升级。                                     |
