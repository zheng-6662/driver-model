# GPTPro Action Items: v222a no-harm gate

1. Implement `stage03_v222a_noharm_gate_diagnostic_20260622.py`.
2. Load v222a cache and selected residual predictions.
3. For each pool, compare fixed baseline `B` with selected residual `M`.
4. Export gain/harm decomposition table.
5. Build diagnostic-only oracle safe gate upper bound.
6. Train lightweight binary validation-only no-harm gate:
   - safe classifier
   - useful classifier
   - tail-harm predictor
7. Select thresholds on validation only.
8. Lock selected gate and report test once.
9. Export:
   - `selected_gate_manifest.json`
   - `val_gate_tradeoff_table.csv`
   - `test_locked_gate_report.csv`
   - `per_sample_gate_decisions.csv`
10. Verify py_compile, leakage guard, forbidden-name exclusion, and ZIP.
11. Update project notes and daily log.
12. Package the result and send the concise evidence summary back to GPTPro for the next instruction.
