# v225 GPTPro decision

## 采纳结论

采纳 GPTPro 对 v222a closeout 的判断：当前主要失败来源是 selector / gate 泛化不稳，而不是 candidate pool 大面积缺曲线。

## 本轮唯一允许执行的 bounded local step

实现一次性 `v225 formal route reconstruction evidence pack`：

- 脚本：`stage03_v225_formal_route_reconstruction_evidence_pack_20260622.py`
- 输出目录：`v225_formal_route_reconstruction_evidence_pack_20260622/`
- 性质：只读证据固化包，不训练、不调阈值、不建 gate/router。

## Formal headline lock

- `loose_main_pool`: `avg_joint_focus`
- `strict_main_pool`: `peak_floor_090`

## Diagnostic-only boundary

以下只能进入 diagnostic appendix / diagnostic-only summary / excluded audit，不得进入 formal usage、formal selected config 或 formal leaderboard：

- `v222a_bounded_residual`
- `v222a_noharm_gate`
- `oracle_safe_gate`
- `ridge_residual_peakfloor`
- `W3_B4_original_soft`
- `oracle`
- `oracle_model`
- `true_label`
- `fallback`

## 禁止事项

- 不做 test retuning
- 不新建 tau
- 不做 `v222a_gate_v2`
- 不做 multi-router / neural gate
- 不训练 v222b / v223
- 不新增大模型
- 不重新选择 formal headline
- 不因 test failure 改阈值或删样本
- 不把 case study 图表解释为 formal aggregate improvement

## Completion rule

v225 是一次性证据包，生成、验证并打包完成后自动停止，再把结果报告给 GPTPro 获取下一轮指令。
