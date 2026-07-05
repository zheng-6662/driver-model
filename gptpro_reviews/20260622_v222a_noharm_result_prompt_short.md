GPTPro，上一条长报告发送后你没有给出可执行回复。这里是压缩版，请只给下一步指令，要求有边界、有产物、有 stop condition。

Codex 已按你上一轮要求完成：

`v222a_gain_harm_decomposition -> oracle safe gate upper bound -> binary validation-only no-harm gate`

执行边界：
- 未训练 v222b/v223；
- 未做多候选 router；
- gate 只用 v222a cache 的 `feature_matrix`；
- 预测器 train-only 拟合；
- 阈值 validation-only 选择；
- test 只在 gate 固定后 locked report 一次。

验证：
- `py_compile` 通过；
- ZIP `bad_file=None`；
- leakage guard 全 pass；
- feature schema fail=0；
- 禁用项未命中 `W3_B4_original_soft / oracle / fallback / true_label`。

Validation-selected gate：
- loose_main_pool：val RMSE delta `-0.018917`，tail delta `-0.013437`，under reduction `0.064725`，formal pass=True。
- strict_main_pool：val RMSE delta `-0.010182`，tail delta `-0.008429`，under reduction `0.003704`，formal pass=True。

Locked test：
- loose_main_pool：formal pass=False；under reduction `0.043478` 保住，但 RMSE delta `+0.010559`、tail delta `+0.027764`，伤 RMSE/tail。
- strict_main_pool：formal pass=False；RMSE delta `-0.008975`、tail delta `-0.005264` 守住，但 under reduction `-0.017241`、strong-under reduction `-0.023438`，低估变差。

Oracle safe gate test 上限仍好：
- loose oracle RMSE `0.520273`，tail `0.597736`，under `0.119565`；
- strict oracle RMSE `0.538076`，tail `0.618740`，under `0.120690`。

Codex 当前判断：
- residual 局部有价值；
- learned no-harm gate validation 过关但 locked test 失败；
- v222a 暂不应作为 formal headline；
- 不应自动进入 v222b/v223。

请你裁决一个方向：
1. 是否停止 v222a 主线，只保留 diagnostic/case study？
2. 若允许继续，只能是什么最小诊断？必须输出哪些文件？stop condition 是什么？
3. 是否继续禁止 v222b/v223？如果不禁止，请说明 locked test 失败为什么不是停止理由。
