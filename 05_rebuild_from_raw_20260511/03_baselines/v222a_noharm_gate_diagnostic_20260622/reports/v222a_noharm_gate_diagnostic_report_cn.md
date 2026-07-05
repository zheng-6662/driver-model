# v222a no-harm gate 诊断报告

## 结论

- loose_main_pool: validation no-harm gate 通过，locked test 未通过。test RMSE delta=0.010559, tail delta=0.027764, under reduction=0.043478, coverage=0.842391。
- strict_main_pool: validation no-harm gate 通过，locked test 未通过。test RMSE delta=-0.008975, tail delta=-0.005264, under reduction=-0.017241, coverage=0.373563。

## GPTPro 指令执行情况

- 已完成 gain/harm decomposition。
- 已完成 diagnostic-only oracle safe gate upper bound。
- 已完成 binary validation-only no-harm gate。
- test 只在 validation-selected gate 固定后报告一次。
- 本轮未训练 v222b/v223，也未做多候选 router。

## 关键文件

- `tables/gain_harm_decomposition.csv`
- `tables/oracle_safe_gate_report.csv`
- `tables/val_gate_tradeoff_table.csv`
- `tables/test_locked_gate_report.csv`
- `tables/per_sample_gate_decisions.csv`
- `logs/selected_gate_manifest.json`
- `v222a_noharm_gate_diagnostic_pack.zip`
