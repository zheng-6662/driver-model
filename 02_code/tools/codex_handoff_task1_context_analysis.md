# Codex Handoff: Task 1 鈥?Context / Anchor Signal Value Range Analysis

## Task Type
Low risk, read-only analysis.

## Objective
Verify whether Q1_fast samples have systematically higher anchor-point signal values (steer, steer_rate, ay, yawrate), which would explain why uniform context broadcast causes tail amplitude mismatch.

## Background
- Current main line: `allphase_control_v2_context_full2s` + deterministic conditioned v2
- Q1_fast (fast-reacting) samples show tail RMSE degradation (+0.0155 delta vs baseline)
- Hypothesis: Q1_fast samples have higher steer_rate/amplitude at anchor, and uniform broadcast of context to all 400 future timesteps amplifies tail mismatch
- The 5-dim context in the maintained training script is: `[steer_anchor, steer_rate, ay, yawrate, style_id]`
- The conditioned v2 formal run used an extended 7-dim context with `structured_v2` conditioning (event-level embedding), but the base anchor signals are still the primary input

## Input Files
1. **Sample manifest**: `F:\data_set_process\data_process\02_code\final_code\model\training\protocol_allphase_control_v2_context_full2s\sample_manifest.csv`
   - Contains `sample_key`, `subj`, `split`, `anchor_s`, `recording_id`, `phase_type`, etc.
2. **Attribution master table**: `F:\data_set_process\data_process\04_project_logs\reports\attribution_master_table.csv`
   - Contains `latency_proxy_bucket` (Q1_fast / Q2 / Q3 / Q4_slow), `eval_morphology_label`, `mechanism_tag`, all delta metrics
3. **Raw vehicle data**: `F:\data_set_process\data_process\01_datasets\?????\??????\{subj}\vehicle\*_vehicle_aligned_cleaned.csv`
   - Contains time-series with `t_s`, `zx|SteeringWheel` (steer), steer_rate columns, `zx|vyaw` (yaw_rate), lateral acceleration
4. **Training script** (reference only, DO NOT modify): `F:\data_set_process\data_process\02_code\final_code\model\training\future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
   - Lines ~723-729 show how ctx is built at anchor point

## What To Do
1. Read sample_manifest to get `sample_key`, `subj`, `recording_id`, `anchor_s` (the anchor timestamp in seconds)
2. For each test-split sample, load the corresponding vehicle CSV and extract signal values at the anchor timestamp:
   - `steer_anchor` (absolute steering wheel angle)
   - `steer_rate` (steering wheel angular velocity, may need to compute from diff if not directly available)
   - `ay` (lateral acceleration)
   - `yawrate` (yaw rate)
3. Join with `attribution_master_table.csv` on `sample_key` to get `latency_proxy_bucket`
4. Group by `latency_proxy_bucket` (Q1_fast / Q2 / Q3 / Q4_slow) and compute:
   - Mean, std, min, max, median of each signal
   - Particularly compare Q1_fast vs non-Q1_fast
5. Also do secondary grouping by `eval_morphology_label` (single_lobe / reverse_correction / multi_correction)
6. Compute Pearson correlation between each anchor signal and `delta_rmse_tail_abs_steer`

## Output
- Script: `F:\data_set_process\data_process\02_code\tools\attribution_context_value_range_analysis.py`
- CSV: `F:\data_set_process\data_process\04_project_logs\reports\context_value_range_by_latency_bucket_20260408.csv`
- Report: `F:\data_set_process\data_process\04_project_logs\reports\context_value_range_by_latency_bucket_20260408.md`

## Constraints
- Read-only: DO NOT modify any training code, protocol config, or split files
- DO NOT modify files in `02_code/final_code/` or `datasetprocess/澶氭ā鎬佹暟鎹?绋嬪簭杩愯缁撴灉/`
- Script goes to `02_code/tools/`, reports go to `04_project_logs/reports/`
- Use `F:\python3.11\python.exe` to run
- After completion, append progress to `F:\data_set_process\data_process\04_project_logs\reports\project_progress_master.md`

## Expected Findings
- If Q1_fast has systematically higher steer_rate / amplitude at anchor, it supports the "uniform broadcast amplifies tail error" hypothesis
- If no systematic difference, the mechanism is more likely related to the structured_v2 event conditioning layer rather than raw anchor signals

