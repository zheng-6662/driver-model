# Codex Handoff: Task 2 鈥?Approximate Timestep & Boundary Analysis Using Existing CSVs

## Task Type
Low risk, read-only analysis.

## Objective
Without prediction sequences (which are missing), use existing sample-level metrics CSVs to approximate:
1. Whether Q1_fast degradation concentrates in tail vs front
2. Whether boundary_shift worsening is "slope flattening (smoothing)" or "boundary time-shift"

## Background
- Step 1 (per-timestep error curves) and Step 3 (boundary slope comparison) from the 5-step plan both need raw prediction sequences, which do not exist
- But the formal_eval CSVs already contain pre-computed front/tail RMSE, boundary_slope_abs_err, tail_slope_abs_err, etc.
- This task uses those existing metrics to answer the same questions approximately

## Input Files
1. **Attribution master table**: `F:\data_set_process\data_process\04_project_logs\reports\attribution_master_table.csv` (749 rows 脳 100 cols)
2. **Conditioned trajectory sample metrics**: `F:\data_set_process\data_process\04_project_logs\reports\v3_selection_conditioned_interaction_pilot_20260327\task_2_conditioned_v2\formal_eval\conditioned_trajectory_sample_metrics.csv`
3. **Baseline trajectory sample metrics**: `F:\data_set_process\data_process\04_project_logs\reports\v3_selection_conditioned_interaction_pilot_20260327\task_2_conditioned_v2\formal_eval\baseline_trajectory_sample_metrics.csv`
4. **Sample level comparison**: `F:\data_set_process\data_process\04_project_logs\reports\v3_selection_conditioned_interaction_pilot_20260327\task_2_conditioned_v2\formal_eval\sample_level_comparison.csv`

## What To Do

### Part A: Front vs Tail Error Concentration (Approximate Step 1)
1. For each sample, compute `tail_to_front_ratio = rmse_tail_abs_steer / rmse_pre_tail_abs_steer` for both baseline and conditioned
2. Compute the delta: `conditioned_ratio - baseline_ratio`
3. Group by `latency_proxy_bucket` (Q1_fast vs others):
   - Mean tail_to_front_ratio for baseline and conditioned
   - If Q1_fast's conditioned ratio is significantly higher than baseline ratio 鈫?degradation concentrates in tail
4. Also group by `eval_morphology_label` 脳 `latency_proxy_bucket` for interaction
5. Make a summary table showing: for each group, baseline front RMSE, baseline tail RMSE, conditioned front RMSE, conditioned tail RMSE, and whether degradation is front-driven or tail-driven

### Part B: Boundary Smoothing vs Shifting (Approximate Step 3)
1. Use `boundary_slope_abs_err` (both `_baseline` and `_conditioned` from attribution table)
2. Also use `tail_slope_abs_err` if available in the sample-level CSVs
3. Compare baseline vs conditioned boundary_slope_abs_err grouped by `eval_morphology_label`:
   - If conditioned has HIGHER boundary_slope_abs_err 鈫?boundary slope mismatch (flattening/smoothing)
   - If conditioned has SIMILAR boundary_slope_abs_err but higher boundary_shift_abs_err 鈫?pure time-shift
4. For single_lobe and reverse_correction separately, compute:
   - Mean/median boundary_slope_abs_err (baseline vs conditioned)
   - Mean/median boundary_shift_abs_err (baseline vs conditioned)
   - Scatter: boundary_slope_abs_err_conditioned vs boundary_shift_abs_err_conditioned
5. Cross-check with `peak_abs_amp_err`: if peak amplitude error is dominant, boundary issues may be secondary to amplitude mismatch

### Part C: Q1_fast 脳 single_lobe Scatter
1. For the intersection of Q1_fast AND single_lobe samples:
   - Scatter plot: `peak_abs_amp_err_conditioned` (x) vs `boundary_shift_abs_err_conditioned` (y)
   - Color by `subj`
   - Also annotate `shape_corr_conditioned` as point size
2. This reveals whether the worst cases are amplitude-driven or boundary-driven

## Output
- Script: `F:\data_set_process\data_process\02_code\tools\attribution_approx_timestep_boundary_analysis.py`
- CSV: `F:\data_set_process\data_process\04_project_logs\reports\approx_timestep_boundary_analysis_20260408.csv`
- Report: `F:\data_set_process\data_process\04_project_logs\reports\approx_timestep_boundary_analysis_20260408.md`
- Figures (optional but helpful): scatter PNGs in `F:\data_set_process\data_process\04_project_logs\reports\`

## Constraints
- Read-only: DO NOT modify any training code, protocol config, or split files
- Script goes to `tools/`, reports go to `reports/`
- Use `F:\python3.11\python.exe` to run
- After completion, append progress to `F:\data_set_process\data_process\04_project_logs\reports\project_progress_master.md`

## Key Questions This Should Answer
1. Is Q1_fast degradation MOSTLY in the tail segment, or also in the front?
2. Is boundary_shift worsening in single_lobe mainly "slope flattening" or "time shift"?
3. In the worst cases (Q1_fast 脳 single_lobe), is the dominant error amplitude or boundary?

