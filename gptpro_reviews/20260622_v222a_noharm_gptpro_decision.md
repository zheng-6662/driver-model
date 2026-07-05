# GPTPro Reply Decision: v222a no-harm gate

## Accepted

- 接受 `v222a_gain_harm_decomposition -> oracle safe gate upper bound -> binary validation-only no-harm gate` 作为下一步。
  - 原因：本地 v222a 结果已经显示 residual 可以降低 loose pool 低估率，但会伤害 RMSE/tail；这正好需要 no-harm gate 诊断，而不是继续加复杂模型。
- 接受 “暂不进入 v222b/v223”。
  - 原因：当前 v222a 没有同时守住 RMSE、tail 和 under，直接加复杂度会扩大过拟合风险。
- 接受 `test` 只在 validation-selected gate 固定后报告。
  - 原因：符合本项目 validation-only 选择纪律。
- 接受输出四件核心产物：
  - `selected_gate_manifest.json`
  - `val_gate_tradeoff_table.csv`
  - `test_locked_gate_report.csv`
  - `per_sample_gate_decisions.csv`

## Rejected

- 拒绝此轮直接做 v222b neural soft fusion 或 v223 mechanism Transformer。
  - 原因：v222a 尚未证明可在 no-harm 前提下保留 under reduction。
- 拒绝做 14 candidates multi-router、continuous softmax fusion 或 neural gate。
  - 原因：validation 只有 108 行，复杂 selector 很容易过拟合。

## Deferred

- 是否恢复 v222b/v223：等本轮 no-harm gate 诊断之后再决定。
- 是否写成论文主结果：只有 no-harm gate 在 validation 选定后 test 也能守住 RMSE/tail 并保留 under 改善，才考虑。

## Evidence Links

- v222a cache: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_candidate_curve_cache_20260622`
- v222a light residual: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622`
- selected metrics: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_selected_metrics.csv`
- baseline metrics: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v222a_light_fusion_residual_20260622\tables\v222a_reference_baseline_metrics.csv`

