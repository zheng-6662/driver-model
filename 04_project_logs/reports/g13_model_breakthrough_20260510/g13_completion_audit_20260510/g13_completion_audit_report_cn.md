# G13 完成审计报告

## 目标拆解

G13 的具体交付标准被拆成以下几类：

- 恢复旧设置和核心 checkpoint，保证旧结果可复验。
- 实现并验证响应类型辅助学习、响应类型影响轨迹预测、方向/幅值物理约束、脑电教师与肌电推理输入组合。
- 使用服务器完成正式训练，并拉回 checkpoint、指标和预测图。
- 每个候选版本记录整体误差、尾段误差、综合选择指标、幅值不足、错侧、后段漂移、G11 困难样本、分响应类型和分被试结果。
- 输出中文实验报告、版本表、统一对照表、预测图索引和是否形成论文主线的建议。

## 逐项核对

| 要求 | 证据 | 状态 |
| --- | --- | --- |
| 旧设置和核心 checkpoint 恢复 | 恢复报告、恢复索引、核心运行目录和 best_model.pt | 完成 |
| 响应类型辅助学习已实现并验证 | 代码含响应类型头；G13A/B/H/I 完整训练；seed2026 诊断和三种子报告 | 完成 |
| 响应类型影响轨迹预测已实现并验证 | 条件化预测头；G13C/F/I 完整训练 | 完成 |
| 方向/幅值/尾段物理约束已实现并验证 | 训练参数含幅值/方向损失；G13F/I 完整训练；物理风险表 | 完成 |
| 脑电教师与肌电推理输入的选择性融合已验证 | G13H/G13I 三种子，命令含 EEG teacher checkpoint 和 raw_emg_only | 完成 |
| 服务器高效训练且未留下训练进程 | 服务器启动记录、并行训练日志、最终 GPU 空闲状态 | 完成 |
| 每个版本记录整体误差、尾段、综合指标和预测图 | g13_seed2026_full_index、g13_hi_seed_wise_metrics、每个 run 的 overview.png/plot_index.csv | 完成 |
| 记录幅值不足、错侧、后段漂移、G11、分响应类型、分被试 | 物理风险表、G11 困难样本表、分响应类型表、分被试表、逐样本明细 | 完成 |
| 中文实验报告、版本表、统一对照表、预测图索引、论文主线建议 | seed2026 诊断报告、三种子复验报告、本审计报告、预测图索引 | 完成 |
| 代码编译检查 | g13_code_compile_check_20260510.txt | 完成 |

## 运行产物完整性

| 版本 | seed | 运行目录 | 必要文件 |
| --- | ---: | --- | --- |
| G13A | 2026 | `F:\data_set_process\data_process\tmp\event_conditioned_runs\G13A_连续风格 + 响应类型辅助学习_seed2026_20260510_185506` | 完整 |
| G13B | 2026 | `F:\data_set_process\data_process\tmp\event_conditioned_runs\G13B_连续风格 + 肌电 + 响应类型辅助学习_seed2026_20260510_185506` | 完整 |
| G13C | 2026 | `F:\data_set_process\data_process\tmp\event_conditioned_runs\G13C_连续风格 + 肌电 + 响应类型影响轨迹预测_seed2026_20260510_185506` | 完整 |
| G13F | 2026 | `F:\data_set_process\data_process\tmp\event_conditioned_runs\G13F_肌电 + 响应类型 + 幅值方向物理约束_seed2026_20260510_185506` | 完整 |
| G13H | 2026 | `F:\data_set_process\data_process\tmp\event_conditioned_runs\G13H_脑电教师 + 肌电学生 + 响应类型辅助学习_seed2026_20260510_185506` | 完整 |
| G13I | 2026 | `F:\data_set_process\data_process\tmp\event_conditioned_runs\G13I_脑电教师 + 肌电学生 + 困难响应加权 + 物理约束_seed2026_20260510_191647` | 完整 |
| G13H | 2027 | `F:\data_set_process\data_process\tmp\event_conditioned_runs\G13H_脑电教师 + 肌电学生 + 响应类型辅助学习_seed2027_20260510_194141` | 完整 |
| G13H | 2028 | `F:\data_set_process\data_process\tmp\event_conditioned_runs\G13H_脑电教师 + 肌电学生 + 响应类型辅助学习_seed2028_20260510_194148` | 完整 |
| G13I | 2027 | `F:\data_set_process\data_process\tmp\event_conditioned_runs\G13I_脑电教师 + 肌电学生 + 困难响应加权 + 物理约束_seed2027_20260510_194157` | 完整 |
| G13I | 2028 | `F:\data_set_process\data_process\tmp\event_conditioned_runs\G13I_脑电教师 + 肌电学生 + 困难响应加权 + 物理约束_seed2028_20260510_194205` | 完整 |

## 代码证据

| 内容 | 文件 | 状态 |
| --- | --- | --- |
| 响应类型标签/辅助头 | `F:\data_set_process\data_process\02_code\final_code\model\training\event_conditioned_baseline_model.py` | 存在 |
| 响应类型条件化预测头 | `F:\data_set_process\data_process\02_code\final_code\model\training\conditioned_trajectory_head.py` | 存在 |
| 训练参数、物理损失、蒸馏权重 | `F:\data_set_process\data_process\02_code\final_code\model\training\run_event_conditioned_trajectory_baseline.py` | 存在 |
| G13 候选版本运行器 | `F:\data_set_process\data_process\02_code\final_code\model\training\fair_vehicle_event_comparison_20260427\run_g13_breakthrough_candidates.py` | 存在 |
| G13 seed2026 诊断脚本 | `F:\data_set_process\data_process\02_code\final_code\model\training\fair_vehicle_event_comparison_20260427\summarize_g13_seed2026_diagnostics.py` | 存在 |
| G13 三种子汇总脚本 | `F:\data_set_process\data_process\02_code\final_code\model\training\fair_vehicle_event_comparison_20260427\summarize_g13_hi_multiseed.py` | 存在 |

## 审计结论

- G13 的执行型交付已经完整：代码、训练、恢复、三种子复验、诊断表、预测图索引和中文报告均已落地。
- 结果结论不是“形成新主线”，而是“G13H/G13I 不能替代 E5A/E6/E10C”。
- 因此 G13 这一阶段可以关闭；后续若继续，应开新阶段，重点研究 seed2027 回落、幅值不足和 G11 困难样本仍无法超过 E6 的原因。

## 主要结论摘要

- G13H：三种子 test RMSE `0.4503±0.0109`，没有超过 E5A/E6/E10C。
- G13I：三种子 test RMSE `0.4546±0.0072`，物理风险更均衡但整体和尾段更弱。
- 当前保留 E5A/E6/E10C 作为主候选，G13H/G13I 作为负面边界和诊断证据。
