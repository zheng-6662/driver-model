# 阶段 3 技术报告：top-K 车辆-only Transformer v0.1

## 范围

- 轨道：`B_response3s_strict_core`。
- 输入：事件前车辆时序 + 因果可得道路/事件上下文。
- 输出：K=3 条轨迹 + 分支 logits。
- 训练：min-of-K 轨迹损失 + 分支选择交叉熵 + 平滑项 + 轻量多样性项。
- checkpoint 选择：validation top-1 RMSE。
- 未使用：subject ID、生理、脑电、连续风格、服务器、服务器密码文件。

## test 指标

| 模型 | RMSE | 错侧率 | 大幅召回 | 困难 top20 RMSE |
|---|---:|---:|---:|---:|
| RBF | 0.533667 | 0.225 | 0.750 | 0.678907 |
| top-1 | 0.587883 | 0.100 | 0.750 | 0.717094 |
| best-of-3 | 0.477534 | 0.025 | 0.875 | 0.634191 |

## 可靠性诊断

- test top1_matches_best_rate=0.300000
- test mean_top1_prob=0.383176
- test mean_prob_margin=0.029751
- test mean_branch_spread=0.210648

## 结论

本轮用于判断真正 top-K 车辆-only 是否比 RBF/keypoint 事后二选一更适合继续。是否升级主线必须以 top-1 指标、固定图和坏样本图为准；best-of-3 只能作为候选覆盖上限。
