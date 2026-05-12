# 阶段 4 连续驾驶风格协议与候选特征处理 v0.1

生成时间：2026-05-13 05:05:54

## 输入

- 样本清单：`F:\data_set_process\data_process\05_rebuild_from_raw_20260511\02_samples\vehicle_instability_response_task_decision_v0_1\tables\sample_response_task_manifest.csv`
- 样本筛选：`window_config_id == pre3_label3_response_coverage` 且 `task_sample_role == response3s_strict_core_candidate`
- 原始车辆文件：来自 manifest 的 `vehicle_raw_absolute_path`

## 无泄漏定义

- 车辆直接输入窗口：`[-3, 0]` 秒，属于车辆-only 基线输入，不作为连续风格；
- 标签窗口：`[0, 3]` 秒，完全禁止进入输入、风格、标准化和任何特征拟合；
- 连续风格候选窗口：`prefix_until_guard3`、`last120_guard3`、`last60_guard3`、`last30_guard3`，全部要求 `end <= anchor - 3s`；
- 标准化：只用 `session_level_split=train` 的候选特征拟合均值和标准差。

## 产物

- `tables/style_feature_candidate_long.csv`：一行一个样本-风格窗口；
- `tables/style_feature_candidate_wide.csv`：一行一个样本，适合后续建模；
- `tables/style_feature_candidate_wide_trainz_session_split.csv`：按 session train-only 统计标准化后的候选特征；
- `tables/style_train_only_scaler_session_split.csv`：训练集均值/标准差；
- `tables/style_source_protocol_table.csv`：风格来源允许/阻塞规则；
- `tables/style_leakage_guard_table.csv`：泄漏边界检查；
- `tables/style_permutation_plan.csv`：置乱和 ID 对照协议；
- `tables/style_subject_road_coupling_audit.csv`：被试-道路耦合审计；
- `tables/style_protocol_gate_table.csv`：是否允许进入下一步的 gate。

## 关键数量

- B 轨道严格核心样本数：270
- long 表行数：1080
- wide 表列数：480
- train-only 可标准化数值特征：436/440

## Gate 结论

| gate | status | evidence | decision_cn |
| --- | --- | --- | --- |
| style_feature_source_defined | pass_protocol | source_protocol_table + leakage_guard_table | 已定义事件前车辆历史风格来源，且排除直接输入和标签未来。 |
| style_candidate_features_extracted | pass | style_feature_candidate_long/wide | 已生成候选风格特征表；样本可用性见 split/subject feasibility。 |
| train_only_standardization_ready | pass_protocol | style_train_only_scaler_session_split.csv | 标准化只允许用 session split 的训练集拟合。 |
| permutation_controls_defined | pass_protocol | style_permutation_plan.csv | 已定义被试内、跨被试、跨 session、道路平衡和 ID 对照。 |
| style_effectiveness_claim_allowed | blocked | no model/permutation result yet | 还没有与 RBF 固定参照、置乱和分被试验证比较，不能说风格有效。 |
| stage05_physio_eeg_allowed | blocked | style baseline not validated yet | 生理/脑电继续阻塞，直到车辆+风格参照完成公平验证。 |

## 当前限制

本轮没有训练模型，也没有评估风格增量。连续风格是否有效，必须在下一步接入固定 RBF 参照后，通过原始风格、置乱风格、驾驶员 ID 对照、分被试/分 session 和物理指标共同判断。
