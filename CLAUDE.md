# CLAUDE.md

## Project Scope

This repository is a driver-response / driver-model research workspace focused on extreme driving conditions and multimodal data:

- vehicle signals
- physiological signals
- EEG signals
- event-level multimodal dataset construction
- future steering / trajectory prediction
- training diagnostics and experiment comparison

The recommended maintained code path is:

- `F:\data_set_process\data_process\02_code\final_code`

Do not treat the whole repository as a single clean training repo. It is a mixed workspace containing:

- current maintained code
- historical scripts
- many archived experiment outputs
- temporary runs
- reports and repair utilities

## Primary Code Areas

When working on the main pipeline, prioritize these files and directories first:

- `F:\data_set_process\data_process\02_code\final_code\processing\vehicle\preprocess_vehicle_v14.py`
- `F:\data_set_process\data_process\02_code\final_code\processing\physio\鐢熺悊鏁版嵁澶勭悊.py`
- `F:\data_set_process\data_process\02_code\final_code\processing\physio\杩涗竴姝ュ鐞嗙敓鐞嗘暟鎹?py`
- `F:\data_set_process\data_process\02_code\final_code\processing\eeg\鑴戠數鏁版嵁澶勭悊.py`
- `F:\data_set_process\data_process\02_code\final_code\processing\eeg\finally.py`
- `F:\data_set_process\data_process\02_code\final_code\processing\eeg\鑴戠數鏁版嵁涓庤溅杈嗘暟鎹榻愭椂闂存埑.py`
- `F:\data_set_process\data_process\02_code\final_code\dataset\build_event_dataset_v2_pad_mask_multipeak.py`
- `F:\data_set_process\data_process\02_code\final_code\model\training\future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
- `F:\data_set_process\data_process\02_code\final_code\model\diagnostics\future_steer_event_rollpeak_transformer_v5_8_diag_eval.py`

Protocol and split references that should be checked before changing training behavior:

- `F:\data_set_process\data_process\02_code\final_code\model\training\protocol_primary_control_v2_context_full2s\protocol_config.json`
- `F:\data_set_process\data_process\02_code\final_code\model\training\protocol_primary_control_v1\protocol_config.json`
- `F:\data_set_process\data_process\02_code\final_code\model\training\protocol_d2_response_aligned_core_v1\protocol_config.json`
- `F:\data_set_process\data_process\02_code\final_code\model\training\protocol_d3_response_aligned_extended_v1\protocol_config.json`

## Repository Guardrails

Prefer editing only active code and lightweight documentation.

Avoid editing or bulk-refactoring these areas unless the task explicitly requires it:

- `F:\data_set_process\data_process\02_code\legacy_multimodal\01_历史入口归档`
- `F:\data_set_process\data_process\03_results\多模态数据\程序运行结果`
- `F:\data_set_process\data_process\03_results\tmp`
- `F:\data_set_process\data_process\03_results\artifacts`
- backup folders such as `*_backup_*`
- one-off repair outputs and generated reports at the repo root

Treat generated outputs as data, not source:

- run summaries
- metrics JSON
- figures
- copied training scripts inside run folders
- exported spreadsheets / reports

## Working Style For This Project

When helping in this repository:

- start from `02_code/final_code` unless the user explicitly asks for historical comparison
- preserve subject-level split policy unless the user explicitly requests a new split
- verify whether a script uses hard-coded paths before changing behavior
- prefer minimal, targeted edits over broad refactors
- call out risks of data leakage, time leakage, label leakage, and split contamination
- be careful with files that mix active code and historical commented-out blocks
- when comparing experiments, use the saved `run_summary.json`, `config.json`, metrics JSON, and protocol config before making claims
- before giving a compressed chat summary for any substantial work, append a detailed progress entry to `F:\data_set_process\data_process\04_project_logs\reports\project_progress_master.md`
- detailed progress entries should capture who did the work, what was done, why it was done, what was found, and the recommended next step
- command execution, file inspection, script edits, experiment analysis, and literature / plan鏁寸悊 all count as progress when they materially advance the project
- when the log gets longer, keep the file's top-level current-status summary usable so a later session does not need to reread the full history first
- when appropriate, update the file's date index and topic index so later sessions can quickly find what was done on a given day or under a given theme

## Goal-Driven Autonomous Mode

If the user frames a task as 鈥淚 will give the goal, you keep pushing until it is achieved,鈥?treat that as a supported default mode for this repository rather than as an unusual request.

In that mode:

- ask for or infer `goal`, `acceptance criteria`, and `red lines`
- do not pause for routine micro-instructions between normal implementation steps
- keep executing through inspect -> plan -> patch -> validate -> log cycles until the acceptance criteria are met or a real evidence-based blocker is reached
- use the repository's existing Claude -> Codex collaboration workflow rather than inventing a parallel process
- before every compressed summary for substantial work, ensure the detailed progress has already been appended to `F:\data_set_process\data_process\04_project_logs\reports\project_progress_master.md`
- treat 鈥渄o not delete my files鈥?as the default guardrail unless the user explicitly overrides it
- only stop to ask the user when a decision has non-obvious consequences, hidden risk, materially changes protocol / compute cost / research direction, or risks drifting away from the core goal of predicting driver behavior and vehicle-state trends under extreme conditions
- in this repository, avoid tunnel vision: if the same narrow issue has already been tried `3-4` times without meaningful improvement, switch direction unless there is strong new evidence
- do not reduce every experiment to tiny smoke runs by habit; if the target path, risks, and outputs are already well understood, it is acceptable to go directly to full training / full evaluation

The formal repository reference for this mode is:

- `F:\data_set_process\data_process\04_project_logs\reports\goal_driven_autonomous_workflow.md`

The user-facing task template is:

- `F:\data_set_process\data_process\04_project_logs\reports\goal_driven_target_template.md`

## High-Risk Failure Modes

Always watch for these:

- train / val / test subject leakage
- accidental changes to event anchor definitions
- mismatched future horizon length
- changing online-only inputs into look-ahead inputs
- hidden changes to label definitions such as `primary`, `response_aligned`, or `full_future_2s_only`
- mixing historical scripts with current final pipeline
- editing generated code copies inside experiment result folders instead of the real source

## Preferred Analysis Order

For any new modeling task, inspect in this order:

1. root `README.md`
2. `02_code/final_code/README.md`
3. relevant `protocol_config.json`
4. active preprocessing script
5. dataset build script
6. active training script
7. diagnostics script
8. recent `run_summary.json` or metrics files

## Experiment Conventions

Important current conventions inferred from the repo:

- many experiments use a 3.0 second history window
- many future prediction tasks use a 2.0 second future horizon
- common sampling rate is 200 Hz
- subject-level fixed split is important and should not be changed casually
- current work includes primary-control, D2 response-aligned, and D3 extended protocol families
- experiment outputs are often saved under timestamped directories with copied configs and summaries

Before proposing a training change, identify:

- which protocol family it belongs to
- whether it changes labels, anchors, targets, or just optimization
- whether comparison against old runs will still be fair

## Common Tasks Claude Should Help With

Good tasks for Claude in this repo:

- inspect preprocessing or training scripts safely
- trace a metric back to the generating script
- compare two protocol configs
- summarize differences across experiment runs
- detect possible leakage or evaluation inconsistencies
- prepare ablation plans
- generate experiment notes or result summaries
- build small analysis helpers in `F:\data_set_process\data_process\02_code\tools`
- for literature search, paper screening, or Zotero import in this repo, prefer the restored ScholarAIO/Zotero workflow described in `F:\data_set_process\data_process\.claude\commands\literature-workflow.md` and `F:\data_set_process\data_process\04_project_logs\reports\codex_academic_zotero_workflow.md` instead of falling back to generic web search first

## Output Locations

Prefer keeping new helper outputs in clearly scoped locations instead of scattering files:

- code utilities: `F:\data_set_process\data_process\02_code\tools`
- human-readable reports: `F:\data_set_process\data_process\04_project_logs\reports`
- temporary scratch outputs: `F:\data_set_process\data_process\03_results\tmp`

If a change is meant to become part of the maintained pipeline, place it under:

- `F:\data_set_process\data_process\02_code\final_code`

## Commands And Execution Notes

This repo appears to rely heavily on script-level hard-coded paths. Before running any long job:

- inspect `ROOT`, `RESULT_ROOT`, and similar constants
- confirm whether the script writes into `绋嬪簭杩愯缁撴灉`, `tmp`, or another output root
- confirm GPU assumptions and seed configuration
- choose smoke tests or full training based on the current decision value and risk level; do not force smoke-first by habit

When asked to run experiments, prefer:

- right-sized validation for the current risk level; small smoke runs are useful, but do not use them mechanically when the path is already clear enough for a full run
- preserving the original full run script
- adding a small helper or wrapper rather than rewriting the core training file unless necessary

Environment defaults for this project:

- when Claude runs Python, prefer `conda run -n predict2 python ...`
- when preparing Python commands or execution handoffs for Codex, default to the `predict2` conda environment
- when running model training / evaluation / diagnostics, prefer GPU by default unless the user explicitly requests CPU or the script/environment requires otherwise
- before long GPU jobs, still confirm output paths, device assumptions, and whether a smoke test should run first
- if a smoke test is no longer the most decision-useful step, it is acceptable to go straight to a full run after documenting why

## Codex Collaboration Defaults

When Claude is asked to call local Codex in this project:

- default to the verified bridge under `D:\ClaudeCode\codex-bridge` rather than assuming `codex` exists in PATH
- prefer `D:\ClaudeCode\codex-bridge\codex.exe`, `D:\ClaudeCode\codex-bridge\invoke-codex.ps1`, or `D:\ClaudeCode\codex-bridge\run-codex.cmd` as the first local Codex entrypoints in this repository
- only fall back to checking other install locations or PATH-first discovery if the user explicitly asks or the bridge path fails
- treat this bridge-first behavior as the default for Claude -> Codex delegation in this workspace

## Future Claude Extensions To Add

The user wants Claude features added later. Good next additions for this project are:

- a repo-specific `/data-check` workflow for split, path, and schema validation
- a `/summarize-run` workflow that reads `run_summary.json`, config, and metrics
- a `/compare-runs` workflow for two experiment folders
- a `/failure-analysis` workflow for bad cases or low-performing subjects
- hooks that warn before editing archived run outputs
- subagents for data audit, training audit, and experiment reporting

Until those are implemented, Claude should still behave as if these priorities exist.
## Future Claude Extensions To Add

The user wants Claude features added later. Good next additions for this project are:

- a repo-specific `/data-check` workflow for split, path, and schema validation
- a `/summarize-run` workflow that reads `run_summary.json`, config, and metrics
- a `/compare-runs` workflow for two experiment folders
- a `/failure-analysis` workflow for bad cases or low-performing subjects
- hooks that warn before editing archived run outputs
- subagents for data audit, training audit, and experiment reporting

Until those are implemented, Claude should still behave as if these priorities exist.



