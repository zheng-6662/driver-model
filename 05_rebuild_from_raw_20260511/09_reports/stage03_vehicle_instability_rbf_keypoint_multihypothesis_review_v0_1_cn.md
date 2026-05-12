# 阶段 3 技术报告：RBF/keypoint 多候选车辆-only 复盘 v0.1

## 范围

- 轨道：`B_response3s_strict_core`。
- 候选：`rbf_kernel_ridge_context_no_subject` 与 `keypoint_residual_vehicle_transformer_no_subject`。
- 可部署策略：复用上一轮 train/val logistic selector，test 只最终评价。
- 上限策略：oracle best-of-two，仅用于诊断，不作为可部署结果。
- 未使用：subject ID、生理、脑电、连续风格、服务器、服务器密码文件。

## 主要 test 指标

| 模型 | RMSE | 错侧率 | 大幅召回 | 困难 top20 RMSE |
|---|---:|---:|---:|---:|
| RBF | 0.533667 | 0.225 | 0.750 | 0.678907 |
| keypoint | 0.548993 | 0.125 | 0.875 | 0.728858 |
| selector | 0.533912 | 0.200 | 0.875 | 0.648365 |
| oracle | 0.475095 | 0.200 | 0.875 | 0.648365 |

## 选择器诊断

- test selector choice accuracy=0.550000
- test selector keypoint rate=0.275000
- test oracle keypoint rate=0.425000
- test mean selector regret=0.059123
- test mean oracle gain over RBF=0.052177

## 结论

RBF/keypoint 的 oracle 上限较明显，证明两者在样本层有互补；但 train/val selector 还不能把这个上限稳定转化为整体 RMSE 收益。当前证据支持继续做车辆-only 多假设/可靠性路线，不支持进入连续风格、生理或 EEG 有效性结论。
