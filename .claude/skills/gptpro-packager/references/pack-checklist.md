# GPT Pro Pack Checklist

## Standard Top-Level Docs

Create these in the pack root:

- `README.md`
- `PROJECT_STATUS_AND_CODEGEN_BRIEF_CN.md`
- `GPTPRO_PROMPT_CN.md`

Recommended roles:

- `README.md`
  - one-line status
  - read order
  - pack contents
  - what GPT Pro should focus on
- `PROJECT_STATUS_AND_CODEGEN_BRIEF_CN.md`
  - current keeper / anchor
  - latest closed run
  - why the branch failed or succeeded
  - exact next approved direction
  - engineering constraints
- `GPTPRO_PROMPT_CN.md`
  - direct instruction for GPT Pro
  - concrete output requirements
  - red lines
  - desired code locations and run config style

## Standard File Groups

### Context

- `04_project_logs/references/current-state.md`
- latest daily log
- `04_project_logs/reports/progress/decision_log.md`
- `04_project_logs/reports/progress/experiment_registry.md`

### Protocol

- `02_code/final_code/model/training/protocol_primary_control_v2_context_full2s/protocol_config.json`
- `02_code/final_code/model/training/protocol_primary_control_v2_context_full2s/frozen_subject_split.json`

### Default Code

- wrapper entrypoint
- `v58_modular/`
- recalc tool
- current follow-up runner and summarizer when relevant
- temporary helpers only if the current closure truly depends on them

### Evidence

Prefer only the files GPT Pro actually needs to judge or implement the next step:

- latest review / closure note
- key recalc summaries for the live keeper, failed branch, and latest follow-up
- manual review artifacts when they materially changed the conclusion
- comparison tables when they save GPT Pro from reopening old debates

### Configs

Include exact `run_config.json` snapshots for:

- the live keeper
- the failed branch you are trying to repair or replace
- the latest follow-up run that sets the new boundary

## Naming Rules

- Put packs under `04_project_logs/reports/`
- Use a date plus purpose plus target, for example:
  - `gptpro_codegen_pack_20260423_h15_late_residual_head_v1`
- Rename generic evidence filenames when copying:
  - `recalc_best_by_loss_summary.json` is too ambiguous on its own
  - prefer names like `h15_ac_cf_hlf_v1_recalc_best_by_loss_summary.json`

## Example Invocation

This pattern matches the current codegen-pack workflow:

```powershell
powershell -ExecutionPolicy Bypass -File ".claude/skills/gptpro-packager/scripts/build_gptpro_pack.ps1" `
  -PackName "gptpro_codegen_pack_20260423_h15_late_residual_head_v1" `
  -EvidenceFiles @(
    "04_project_logs/reports/gptpro_effectiveness_review_20260423.md",
    "04_project_logs/reports/effectiveness_followup_20260423/h15_ac_cf_hlf_v1_summary.md",
    "03_results/tmp/input_group_ablation_20260421/baseline_fixed_input/TRAIN_V5_4_STATECOND_REV_20260421_223235/figures/recalc_best_by_structured_summary.json|baseline_fixed_input_recalc_best_by_structured_summary.json",
    "03_results/tmp/effectiveness_followup_20260422/h15_full/TRAIN_V5_4_STATECOND_REV_20260422_134929/figures/recalc_best_by_structured_summary.json|h15_recalc_best_by_structured_summary.json",
    "03_results/tmp/effectiveness_followup_20260423/h15_ac_cf_hlf_v1/TRAIN_V5_4_STATECOND_REV_20260423_131956/figures/recalc_best_by_loss_summary.json|h15_ac_cf_hlf_v1_recalc_best_by_loss_summary.json",
    "03_results/tmp/effectiveness_followup_20260423/h15_ac_cf_hlf_v1/TRAIN_V5_4_STATECOND_REV_20260423_131956/figures/recalc_best_by_structured_summary.json|h15_ac_cf_hlf_v1_recalc_best_by_structured_summary.json",
    "03_results/tmp/effectiveness_followup_20260423/h15_ac_cf_hlf_v1/TRAIN_V5_4_STATECOND_REV_20260423_131956/figures/strong_pos_review_best_by_loss/strong_pos_review_index.csv|strong_pos_review/strong_pos_review_index.csv",
    "03_results/tmp/effectiveness_followup_20260423/h15_ac_cf_hlf_v1/TRAIN_V5_4_STATECOND_REV_20260423_131956/figures/strong_pos_review_best_by_loss/strong_pos_01_idx225.png|strong_pos_review/strong_pos_01_idx225.png"
  ) `
  -ConfigFiles @(
    "03_results/tmp/input_group_ablation_20260421/baseline_fixed_input/TRAIN_V5_4_STATECOND_REV_20260421_223235/run_config.json|baseline_fixed_input_run_config.json",
    "03_results/tmp/effectiveness_followup_20260422/h15_full/TRAIN_V5_4_STATECOND_REV_20260422_134929/run_config.json|h15_run_config.json",
    "03_results/tmp/effectiveness_followup_20260423/h15_ac_cf_hlf_v1/TRAIN_V5_4_STATECOND_REV_20260423_131956/run_config.json|h15_ac_cf_hlf_v1_run_config.json"
  ) `
  -CodeFiles @(
    "tmp/recalc_v58_metrics_shim_20260423.py"
  )
```

## Validation

Check these after running the script:

- top-level docs still exist in the pack root
- `context/`, `evidence/`, `configs/`, `protocol/`, and `code/` are populated
- the zip timestamp is newer than the last edit to the top-level docs
- the zip contains the top-level docs and the copied directories
