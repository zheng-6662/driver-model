# 阶段 3 技术报告：top-K top1/bestK 差距复盘 v0.1

## 输入

- `topk_vehicle_transformer_branch_diagnostics.csv`
- `topk_vehicle_transformer_per_sample_metrics.csv`
- `sample_response_task_manifest.csv`
- `response_decomposition_sample_labels.csv`

未训练新模型，未使用 subject ID 作为训练特征，未使用生理、脑电、连续风格、服务器或服务器密码文件。

## 诊断规则

- `top1_bestk_gap = rmse_top1_branch - rmse_bestk`
- 高 gap 阈值来自 train split 的 75 分位。
- 简单风险分数使用 train split 标准化：`-z(top1_prob) - z(prob_margin) + z(branch_spread_mean) + z(branch_spread_peak)`。
- 高风险阈值来自 train split 的 75 分位。

## test 摘要

| 指标 | 数值 |
|---|---:|
| top1 与 bestK 一致率 | 0.300000 |
| 平均 top1-bestK gap | 0.110531 |
| 平均 top1-RBF RMSE 差 | 0.068514 |
| 平均 bestK over RBF 收益 | 0.042018 |
| 高风险捕捉高 gap 比例 | 0.545455 |
| 低置信规则捕捉高 gap 比例 | 0.636364 |

## 结论

top-K v0.1 的下一步应集中在选择机制和可靠性估计，而不是继续把 best-of-K 上限当成模型效果。当前结果仍然阻塞连续风格、生理和 EEG 增量验证。
