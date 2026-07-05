GPTPro你好。Codex 已按你上一轮指令完成：

`v222a_gain_harm_decomposition -> oracle safe gate upper bound -> binary validation-only no-harm gate -> 再决定是否停止 v222a`

请你基于下面的本地执行结果给下一步指令。请保持范围有边界，避免陷入局部困境；如果你判断应停止 v222a，请明确停止，并给出下一条可执行路线、必交产物和 stop condition。

## 本轮执行边界

- 未训练 v222b/v223。
- 未做多候选 router。
- formal baseline 固定：
  - loose_main_pool baseline = `avg_joint_focus`
  - strict_main_pool baseline = `peak_floor_090`
- gate 模型只用 v222a cache 的 `feature_matrix` 做 inference features。
- safe/useful/tail-harm 预测器只在 train 拟合。
- tau_safe / tau_useful / tau_tail_harm 只按 validation tradeoff 选择。
- test 只在 gate 固定后写入 locked report。

## 已生成产物

- 脚本：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v222a_noharm_gate_diagnostic_20260622.py`
- 输出目录：
  - `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_noharm_gate_diagnostic_20260622`
- 必交表：
  - `tables/gain_harm_decomposition.csv`
  - `tables/oracle_safe_gate_report.csv`
  - `tables/val_gate_tradeoff_table.csv`
  - `tables/test_locked_gate_report.csv`
  - `tables/per_sample_gate_decisions.csv`
  - `logs/selected_gate_manifest.json`
- ZIP：
  - `v222a_noharm_gate_diagnostic_pack.zip`

## 验证结果

- `python -m py_compile` 通过。
- ZIP 校验 `bad_file=None`，包含 11 个文件。
- leakage guard 全部 pass：
  - `feature_schema_forbidden_tokens`
  - `selection_uses_validation_only`
  - `train_only_fit`
  - `test_locked_once`
  - `no_v222b_or_v223`
- feature schema audit fail=0。
- 禁用项检查未命中 `W3_B4_original_soft / oracle_model / fallback / true_label`。

## validation-selected gate 结果

`loose_main_pool`

- 选择阈值：tau_safe=0.4, tau_useful=0.2, tau_tail_harm=0.0
- validation formal gate pass=True
- validation RMSE delta vs baseline = `-0.018917`
- validation tail delta = `-0.013437`
- validation under reduction = `0.064725`
- validation strong-under reduction = `0.069959`
- coverage = `0.944984`

`strict_main_pool`

- 选择阈值：tau_safe=0.6, tau_useful=0.2, tau_tail_harm=0.0
- validation formal gate pass=True
- validation RMSE delta vs baseline = `-0.010182`
- validation tail delta = `-0.008429`
- validation under reduction = `0.003704`
- validation strong-under reduction = `0.004651`
- coverage = `0.359259`

## locked test 结果

`loose_main_pool`

- formal_gate_pass_vs_baseline=False
- aggregate_noharm_pass_vs_baseline=False
- under_improved_vs_baseline=True
- test RMSE delta vs baseline = `+0.010559`
- test tail delta = `+0.027764`
- test under reduction = `0.043478`
- test strong-under reduction = `0.037313`
- coverage = `0.842391`
- 解释：仍能减少低估，但伤害 RMSE/tail。

`strict_main_pool`

- formal_gate_pass_vs_baseline=False
- aggregate_noharm_pass_vs_baseline=True
- under_improved_vs_baseline=False
- test RMSE delta vs baseline = `-0.008975`
- test tail delta = `-0.005264`
- test under reduction = `-0.017241`
- test strong-under reduction = `-0.023438`
- coverage = `0.373563`
- 解释：守住 RMSE/tail，但 under 指标反而变差。

## oracle safe gate 上限

`loose_main_pool` test oracle:

- RMSE = `0.520273`
- tail RMSE = `0.597736`
- under = `0.119565`
- strong-under = `0.156716`
- coverage = `0.423913`
- safe_under_fix_coverage = `0.300`

`strict_main_pool` test oracle:

- RMSE = `0.538076`
- tail RMSE = `0.618740`
- under = `0.120690`
- strong-under = `0.156250`
- coverage = `0.436782`
- safe_under_fix_coverage = `0.125`

## Codex 当前判断

- residual 局部有价值，oracle safe gate 上限明显好于 baseline。
- 但 learned no-harm gate 在 validation 过关、locked test 失败，且两池失败模式相反：
  - loose：under 改善保住了，但 RMSE/tail 被伤害；
  - strict：RMSE/tail 保住了，但 under 改善消失并反向变差。
- 因此 v222a 暂不应作为 formal headline，也不应自动进入 v222b/v223。

## 需要你裁决

请给出下一步指令，只选一个方向并写清产物/停止线：

1. 是否正式停止 v222a bounded residual/no-harm gate 主线，只保留 diagnostic/case study？
2. 是否允许一轮更窄的 gate-feature 诊断？如果允许，请明确不得看 test 调阈值、不得扩大为 router、必须输出哪些表，并给 stop condition。
3. 如果你认为应换方向，请给出下一条最小可执行路线，以及它为什么不是继续在 v222a 局部打转。
4. 是否继续禁止 v222b/v223？如果不禁止，请说明 locked test 失败为什么不构成停止理由，并给出严格的验证纪律。
