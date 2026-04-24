---
name: gptpro-packager
description: Package the current repo state into a GPT Pro handoff or codegen pack, including current-state logs, selected evidence, exact configs, active source files, and a direct next-step prompt. Use when the user asks to bundle current project background, source code, latest experiment results, or next-step instructions for GPT Pro, especially in F:\data_set_process\data_process where packs should be written under 04_project_logs/reports/ and zipped for sharing.
---

Build a GPT Pro pack for this repo.

## Workflow

1. Read the current anchors before packaging:
   - `04_project_logs/references/current-state.md`
   - latest daily log
   - `04_project_logs/reports/progress/decision_log.md`
   - `04_project_logs/reports/progress/experiment_registry.md`
2. Read the latest closure report that defines the current no-go branch and the next approved direction.
3. If the decision boundary changed in the latest work, update the project logs first. Do not package stale state.
4. Choose a new pack folder under `04_project_logs/reports/`. Use a date-stamped, purpose-specific name such as `gptpro_codegen_pack_YYYYMMDD_target`.
5. Create or update these top-level files in the pack root before the final zip:
   - `README.md`
   - `PROJECT_STATUS_AND_CODEGEN_BRIEF_CN.md`
   - `GPTPRO_PROMPT_CN.md`
6. Run `scripts/build_gptpro_pack.ps1` to populate:
   - `context/`
   - `evidence/`
   - `configs/`
   - `protocol/`
   - `code/`
   and to refresh the zip.
7. Validate that the zip contains the three top-level docs plus the copied subdirectories.
8. Return both the folder path and the zip path to the user.

## Pack Rules

- Keep the pack narrow. Include the current keeper/anchor, the blocked branch, the latest closure report, and the next approved target. Do not dump the whole repo history.
- Prefer the current `v58_modular/` tree plus the thin wrapper entrypoint over stale monolithic snapshots.
- For this repo, treat these as the default code payload unless the user asks otherwise:
  - `02_code/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
  - `02_code/final_code/model/training/v58_modular/`
  - `02_code/tools/recalc_v58_checkpoint_with_current_metrics.py`
  - `02_code/tools/run_effectiveness_followup.py`
  - `02_code/tools/summarize_effectiveness_followup.py`
- If the current closure depends on a temporary helper such as a recalc shim, include it in `code/`.
- Rename copied evidence and config files so they remain understandable out of context. Use `source|dest-name` specs when calling the script.

## Script

- Use `scripts/build_gptpro_pack.ps1` for the repetitive filesystem work.
- The script automatically copies the standard context, protocol files, wrapper, modular code tree, and tool scripts.
- Add run-specific evidence, configs, and extra code via `source|dest-name` specs.
- Read `references/pack-checklist.md` for:
  - the standard file groups
  - naming rules
  - an example command that recreates the current `H15_LATE_RESIDUAL_HEAD_v1` codegen pack pattern

## Final Check

- Confirm the latest project-state files in the pack match the actual current repo state.
- Confirm the prompt inside `GPTPRO_PROMPT_CN.md` matches the current next approved move rather than an older plan.
- Confirm the zip is refreshed after the top-level docs were written.
