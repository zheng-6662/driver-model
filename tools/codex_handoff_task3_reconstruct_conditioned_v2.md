# Codex Handoff: Task 3 — Reconstruct Conditioned V2 Training + Save Prediction Sequences

## Task Type
Medium risk. Involves restoring deleted training script files and retraining model (same config, same data, same split).

## Objective
1. Restore the conditioned v2 training pipeline source files from git history
2. Retrain conditioned v2 model using the EXACT same config as the original formal run
3. Modify the eval script to ALSO save per-sample prediction sequences (pred + GT, 400 timesteps)
4. Run eval and save prediction sequences as `.npz`

## Critical Background
- The conditioned v2 formal run (`EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL_20260327_000432`) has only summary/metrics CSV, no checkpoint or prediction sequences
- The training scripts were committed in git commit `418f869` ("Add v3 conditioned trajectory and interaction pilot") but later deleted from working tree
- The `.py` source files are gone, but `.pyc` files remain in `__pycache__`, AND full source is retrievable from `git show 418f869:<path>`
- The training script `run_event_conditioned_trajectory_baseline.py` supports BOTH `--conditioning-mode baseline` and `--conditioning-mode structured_v2` (the latter is conditioned v2)

## Source Recovery — Required Files from git commit 418f869

Restore these files from git history to their original locations under `datasetprocess/final_code/model/training/`:

```
git show 418f869:datasetprocess/final_code/model/training/conditioned_trajectory_head.py > conditioned_trajectory_head.py
git show 418f869:datasetprocess/final_code/model/training/event_conditioned_baseline_model.py > event_conditioned_baseline_model.py
git show 418f869:datasetprocess/final_code/model/training/run_event_conditioned_trajectory_baseline.py > run_event_conditioned_trajectory_baseline.py
git show 418f869:datasetprocess/final_code/model/training/eval_event_conditioned_trajectory.py > eval_event_conditioned_trajectory.py
git show 418f869:datasetprocess/final_code/model/training/event_conditioned_eval_support.py > event_conditioned_eval_support.py
git show 418f869:datasetprocess/final_code/model/training/event_head.py > event_head.py
git show 418f869:datasetprocess/final_code/model/training/event_targets.py > event_targets.py
git show 418f869:datasetprocess/final_code/model/training/event_target_export.py > event_target_export.py
git show 418f869:datasetprocess/final_code/model/training/plot_event_conditioned_trajectory.py > plot_event_conditioned_trajectory.py
```

## Training Config — From Original run_summary.json

These are the EXACT parameters used for the conditioned v2 formal run. Use them verbatim:

```json
{
  "manifest": "F:\\data_set_process\\data_process\\datasetprocess\\final_code\\model\\training\\protocol_allphase_control_v2_context_full2s\\sample_manifest.csv",
  "run_prefix": "EXP_EVENT_CONDITIONED_TRAJECTORY_V2_FORMAL",
  "seed": 2026,
  "device": "cuda",
  "init_checkpoint": null,
  "epochs": 3,
  "min_epochs": 3,
  "patience": 2,
  "batch_size": 64,
  "lr": 0.0003,
  "weight_decay": 0.0,
  "grad_clip": 1.0,
  "event_loss_weight": 0.5,
  "teacher_forcing_ratio": 0.75,
  "selection_mode": "structure_aware_primary",
  "d_model": 128,
  "nhead": 2,
  "enc_layers": 2,
  "dec_layers": 2,
  "ffn_dim": 256,
  "dropout": 0.1,
  "event_embed_dim": 96,
  "event_bin_size": 20,
  "conditioning_mode": "structured_v2",
  "structure_width": 0.065,
  "gate_temperature": 0.04,
  "event_residual_scale": 1.2,
  "use_privileged_teacher": false
}
```

**NOTE**: The original run used `device: "cpu"` and `init_checkpoint` pointing to the baseline formal run's best_model.pt. Both the checkpoint and CPU setting need adjustment:
- Use `device: "cuda"` (user has GPU available, will be much faster)
- The init_checkpoint is deleted, so you have two options:
1. First retrain baseline (with `--conditioning-mode baseline`, same manifest), then use its checkpoint as init for v2
2. Or train v2 from scratch (results may differ slightly from original but should be structurally similar)

Option 1 is more faithful. The original baseline run config was also 3 epochs, same manifest, same seed. Check `tmp/event_conditioned_runs/EXP_EVENT_CONDITIONED_TRAJECTORY_BASELINE_FORMAL_20260326_211235/run_summary.json` for baseline config details.

## Prediction Sequence Export — Modification to Eval Script

After training completes, modify `eval_event_conditioned_trajectory.py` to:
1. During inference (where `preds` and `trues` lists are collected), also save the raw arrays
2. Save as `np.savez_compressed(path, pred=pred_array, true=true_array, sample_keys=sample_key_array)`
3. Shape should be: `pred` and `true` as `(N_samples, 400, 3)` (400 timesteps, 3 channels: steer, yawrate, ay)
4. `sample_keys` as `(N_samples,)` string array for joining with attribution table

Save to: `F:\data_set_process\data_process\reports\conditioned_v2_prediction_sequences.npz`
Also save baseline's: `F:\data_set_process\data_process\reports\baseline_prediction_sequences.npz`

## Output Files
- Restored source files in `datasetprocess/final_code/model/training/`
- Trained checkpoints in the new run directory under `tmp/event_conditioned_runs/`
- `reports/conditioned_v2_prediction_sequences.npz`
- `reports/baseline_prediction_sequences.npz`

## Constraints
- DO NOT modify `protocol_config.json`, `sample_manifest.csv`, split definitions, or any existing run outputs
- DO NOT change hyperparameters from the original config — the goal is faithful reproduction
- Use `F:\python3.11\python.exe`
- The training runs on GPU (`device: "cuda"`), expect ~5-10 min per 3-epoch run
- After completion, append progress to `F:\data_set_process\data_process\reports\project_progress_master.md`
- The restored `.py` files should be placed back in their original locations but DO NOT `git add` them — leave as untracked

## Risk Assessment
- **No split/protocol risk**: using EXACT same manifest and split
- **Reproducibility note**: Results may not be bit-identical to original (no saved checkpoint to resume from), but structural conclusions should be consistent
- **Disk space**: ~50 MB for two sets of prediction sequences + checkpoints
