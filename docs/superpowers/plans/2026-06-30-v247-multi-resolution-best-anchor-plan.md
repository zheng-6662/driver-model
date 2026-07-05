# v247 Multi-Resolution Best Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build v247 as a multi-resolution best-anchor discovery audit: reconstruct fine-grid anchors, score them with the locked v241 trajectory model, derive offline best-anchor labels, and train an input-only selector to approximate those labels.

**Architecture:** A single focused experiment script will reuse existing v236 raw-vehicle sampling and v241/v239 model inference utilities. It will not train a new trajectory model; it will only run locked v241 inference on newly sampled fine-grid anchors and train lightweight selector baselines on train split. Outputs follow the existing `03_baselines/vNNN_*` table/figure/report/log/zip pattern.

**Tech Stack:** Python, pandas, numpy, scikit-learn, PyTorch, matplotlib, existing v236/v238/v239/v241 experiment modules.

---

### Task 1: Create v247 Script Skeleton And Plan Integration

**Files:**
- Create: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v247_multi_resolution_best_anchor_discovery_20260630.py`
- Read: `F:\data_set_process\data_process\docs\superpowers\specs\2026-06-30-v247-multi-resolution-best-anchor-design.md`

- [ ] **Step 1: Create imports, constants, output folders, and shared helpers**

Implement a script header with Chinese comments explaining that v247 is best-anchor discovery, not anchor shifting. Include constants:

```python
FINE_DELAY_MS = list(range(0, 1001, 50))
COARSE_DELAY_MS = [0, 200, 400, 600, 800, 1000]
SCORE_CONFIGS = [
    ("error_only", 0.00, 0.00),
    ("delay_l03", 0.03, 0.00),
    ("delay_l05", 0.05, 0.00),
    ("delay_l10", 0.10, 0.00),
    ("delay_l05_unstable_m03", 0.05, 0.03),
    ("delay_l05_unstable_m05", 0.05, 0.05),
    ("delay_l10_unstable_m05", 0.10, 0.05),
]
PRIMARY_SCORE_NAME = "delay_l05_unstable_m05"
```

Add helpers: `import_module_from_path`, `ensure_clean_output`, `write_csv`, `file_sha256`, `finite_rmse`, `safe_mode_int`, `zip_outputs`.

- [ ] **Step 2: Verify script compiles**

Run:

```powershell
python -m py_compile "F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v247_multi_resolution_best_anchor_discovery_20260630.py"
```

Expected: exit code 0.

### Task 2: Build Fine-Grid Rolling Dataset

**Files:**
- Modify: `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v247_multi_resolution_best_anchor_discovery_20260630.py`
- Reuse: `stage03_v236_rolling_reanchor_dataset_and_baseline_20260624.py`

- [ ] **Step 1: Reuse v236 raw sampler with 50ms delays**

Add `build_fine_grid_dataset()`:

```python
def build_fine_grid_dataset() -> tuple[object, pd.DataFrame, list[str], pd.DataFrame]:
    V236.DELAY_MS = FINE_DELAY_MS
    event_df = V236.load_event_manifest()
    x_hist, x_road, x_phase, y_future, manifest, dropped = V236.build_rolling_dataset(event_df)
    x_design, feature_names = V236.build_design_matrix(x_hist, x_road, x_phase)
    data = V238.RollingData(
        manifest=manifest.reset_index(drop=True),
        x_hist=x_hist.astype(np.float32),
        x_road=x_road.astype(np.float32),
        x_phase=x_phase.astype(np.float32),
        y_future=y_future.astype(np.float32),
        pred_v236=np.full_like(y_future, np.nan, dtype=np.float32),
        feature_names=feature_names,
        target_names=V236.TARGET_NAMES,
    )
    return data, manifest.reset_index(drop=True), feature_names, dropped
```

- [ ] **Step 2: Add sampling support audit**

Add `build_sampling_audit(manifest, dropped)` that reports:

```text
requested_delay_step_ms = 50
actual_delay_values
n_events
n_expected_rows = n_events * 21
n_generated_rows
n_dropped_rows
max_abs_nearest_time_error_ms_p95
max_abs_nearest_time_error_ms_max
fine_grid_sampling_checked = True
fine_grid_supported = generated delay count includes all 21 requested delays for most events
```

- [ ] **Step 3: Smoke-test dataset construction on live data**

Run a temporary import-and-call snippet:

```powershell
@'
import importlib.util
from pathlib import Path
p = Path(r"F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v247_multi_resolution_best_anchor_discovery_20260630.py")
spec = importlib.util.spec_from_file_location("v247", p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.ensure_clean_output()
data, manifest, feature_names, dropped = m.build_fine_grid_dataset()
print(data.x_hist.shape, data.x_road.shape, data.x_phase.shape, data.y_future.shape, manifest["delay_ms"].nunique())
print(sorted(manifest["delay_ms"].unique().tolist())[:5], sorted(manifest["delay_ms"].unique().tolist())[-5:])
'@ | python -X utf8 -u -
```

Expected: delay count should be 21 if 50ms fine grid is supported.

### Task 3: Run Locked v241 Inference On Fine Anchors

**Files:**
- Modify: `stage03_v247_multi_resolution_best_anchor_discovery_20260630.py`
- Reuse: `stage03_v241_stronger_temporal_model_20260626.py`
- Reuse: `stage03_v239_light_attention_noharm_20260626.py`

- [ ] **Step 1: Add v241 checkpoint loader**

Implement `load_v241_model_for_data(data, device)`:

```python
checkpoint = torch.load(V241_MODEL, map_location="cpu", weights_only=False)
config = dict(checkpoint["config"])
model = V241.StrongerTemporalQueryAttention(
    hist_dim=data.x_hist.shape[-1],
    road_dim=data.x_road.shape[-1],
    phase_dim=data.x_phase.shape[-1],
    point_dim=len(V238.POINT_EXTRA_FEATURE_NAMES),
    hist_len=data.x_hist.shape[1],
    road_len=data.x_road.shape[1],
    hidden_dim=int(config["hidden_dim"]),
    n_heads=int(config["n_heads"]),
    n_layers=int(config["n_layers"]),
    mlp_hidden=int(config["mlp_hidden"]),
    dropout=0.0,
).to(device)
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()
```

- [ ] **Step 2: Add scaler conversion**

Implement `scalers_from_checkpoint_payload(payload)` returning `V239.SequenceScalers(...)` from the checkpoint `scalers` dict. Do not refit scalers on fine-grid data.

- [ ] **Step 3: Predict fine-grid curves**

Implement:

```python
x_base = V238.build_base_design_matrix(data)
point_data = V238.build_point_dataset(data, x_base)
scalers = scalers_from_checkpoint_payload(checkpoint["scalers"])
arrays = V239.standardize_arrays(data, point_data, scalers)
pred_curve = V239.predict_all_points(model, arrays, point_data, scalers, device, batch_size=8192)
```

- [ ] **Step 4: Validate against existing coarse predictions**

For coarse delay rows present in v236/v241, compare fine-grid v241 inference to the saved v241 prediction. Output `tables/v247_coarse_replay_alignment.csv` with RMSE/MAE by split and delay. This is a sanity check that checkpoint inference is aligned with existing artifacts.

### Task 4: Build Candidate Score Table And Offline Best Anchors

**Files:**
- Modify: `stage03_v247_multi_resolution_best_anchor_discovery_20260630.py`

- [ ] **Step 1: Compute candidate error metrics**

For each fine anchor sample compute:

```text
candidate_tail_rmse_v241
candidate_original_remaining_rmse_v241
candidate_tail_point_n
candidate_original_remaining_point_n
candidate_delay_ms
candidate_delay_s
```

Use steering delta target `data.y_future[:, :, 0]` and predicted curve. Tail error should evaluate points whose `candidate_delay_s + future_grid_s` falls within `[1.0, 2.0]`.

- [ ] **Step 2: Compute instability features**

Use `data.x_hist` around the candidate anchor:

```text
abs_steer_slope_last05 = abs(steer[-1] - steer[-6])
abs_steer_second_diff_last05 = abs((steer[-1] - steer[-6]) - (steer[-6] - steer[-11]))
abs_yaw_change_last05 = abs(yaw_rate[-1] - yaw_rate[-6])
abs_lat_change_last05 = abs(lateral_distance[-1] - lateral_distance[-6])
```

Normalize each component by train split median absolute scale, then compute `instability_penalty`.

- [ ] **Step 3: Score each candidate**

For each `SCORE_CONFIGS` entry:

```python
candidate[f"score_{name}"] = (
    candidate_tail_rmse_v241
    + lambda_wait * (candidate_delay_ms / 1000.0)
    + mu_unstable * instability_penalty
)
```

- [ ] **Step 4: Select best anchor per event**

For each `event_uid` and score name, choose minimum score; tie-break by earlier `candidate_delay_ms`. Output:

```text
tables/v247_best_anchor_by_event.csv
tables/v247_best_anchor_distribution.csv
tables/v247_score_weight_sweep_summary.csv
```

### Task 5: Train Input-Only Selector

**Files:**
- Modify: `stage03_v247_multi_resolution_best_anchor_discovery_20260630.py`

- [ ] **Step 1: Build selector feature table**

Allowed inputs:

```text
candidate_delay_ms
nearest_coarse_delay_ms
residual_offset_ms
scene_type
pool_key
history statistics
road statistics
phase features
instability visible components
```

Forbidden inputs:

```text
event_uid
recording
subject
candidate_tail_rmse_v241
score_* as features
future true curve
oracle/best labels as features
```

- [ ] **Step 2: Train selector baselines**

Train on train split only:

```text
selector_ridge_score
selector_random_forest_score
```

Target is `score_{PRIMARY_SCORE_NAME}`. Predict candidate score for all splits, rank candidates within event, and select the lowest predicted score.

- [ ] **Step 3: Add fixed policies**

Add:

```text
policy_keep_0ms_anchor
policy_wait_to_latest_anchor
policy_nearest_coarse_oracle_proxy
```

The first two are non-oracle baselines; the oracle proxy is diagnostic only and should be marked as not deployable if included.

### Task 6: Evaluate Selector And Generate Figures

**Files:**
- Modify: `stage03_v247_multi_resolution_best_anchor_discovery_20260630.py`

- [ ] **Step 1: Selector summary metrics**

Output `tables/v247_selector_policy_summary.csv` with:

```text
exact_50ms_match_rate
within_50ms_rate
within_100ms_rate
within_200ms_rate
mean_selected_error_v241
mean_best_error_v241
mean_current_0ms_error_v241
selected_error_delta_vs_current
gain_capture_rate
mean_selected_delay_ms
mean_best_delay_ms
```

Group by:

```text
all
normal
bad_top10
very_bad_top5
early_best_after_400
observe_later_like
strong_steer
reverse
```

- [ ] **Step 2: Figures**

Generate:

```text
figures/v247_best_anchor_distribution_by_group.png
figures/v247_selector_vs_oracle_error.png
figures/v247_selected_delay_distribution.png
figures/v247_error_delay_score_curves_examples.png
figures/v247_signal_anchor_alignment.png
```

- [ ] **Step 3: Report**

Generate `reports/v247_multi_resolution_best_anchor_discovery_cn.md` with conclusions in Chinese:

```text
50ms fine grid 是否支持
error-only 是否偏向最晚
等待代价/不稳定惩罚是否改变 best anchor 分布
差样本和 normal 的 best anchor 是否不同
selector 是否超过 wait-latest
下一步是否值得进入模型训练
```

### Task 7: Logs, ZIP, Notes, Verification

**Files:**
- Modify: `PROJECT_STATUS_CN.md`
- Modify: `TASK_QUEUE_CN.md`
- Modify: `ARTIFACT_INDEX_CN.md`
- Modify/Create: `daily_logs/2026-06-30.md`

- [ ] **Step 1: Guardrail logs**

Write `logs/guardrail_check.json` with:

```text
pass
stage
no_trajectory_model_training
input_only_selector
oracle_best_anchor_upper_bound_only
no_test_based_retuning
no_event_uid_or_recording_as_features
fine_grid_sampling_checked
score_weights_declared_before_test_summary
zip_testzip
```

- [ ] **Step 2: Run verification**

Run:

```powershell
python -m py_compile "F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v247_multi_resolution_best_anchor_discovery_20260630.py"
python -X utf8 -u "F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\scripts\stage03_v247_multi_resolution_best_anchor_discovery_20260630.py"
```

Expected:

```text
guardrail_check.pass=True
ZIP testzip=None
report exists
figures exist
```

- [ ] **Step 3: Sync project notes**

Append v247 summary to the four note-layer files with paths, key results, and next decision. Do not remove previous v245/v246 entries.

