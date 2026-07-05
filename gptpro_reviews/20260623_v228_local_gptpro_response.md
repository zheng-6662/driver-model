# GPTPro local desktop response: v228 final paper artifact freeze

Source: local ChatGPT Desktop app, clean ASCII handoff / retry after previous mojibake issue.

Status: valid GPTPro response obtained.

## Response

1. accept_or_reject_v227_closeout:
   Accept v227 as a valid reporting-only fallback. Reason: it uses only existing v225/v226 evidence, keeps locked formal models `loose=avg_joint_focus` and `strict=peak_floor_090`, performs no training, no tau/threshold search, no gate/router/selector, no v222b/v223, no formal headline change, and produces the required claim/readiness package with verified ZIP and guardrail compliance.

2. next_local_task_and_version:
   `stage03_v228_final_paper_artifact_freeze_20260623.py`
   Task type: reporting / packaging / manuscript-readiness only. No model work.

3. allowed_inputs:
   Only these existing artifacts may be read:

```text
05_rebuild_from_raw_20260511/03_baselines/v225_formal_route_reconstruction_evidence_pack_20260622/
05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/
05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_*/
```

Allowed files include only existing tables, reports, logs, figure manifests, and selected formal figures from v225/v226/v227.
Forbidden inputs remain: diagnostic-only rows as formal inputs, oracle/true-label/fallback rows, v222a gate/residual as formal model, v222b/v223 outputs, new predictions, new training cache, new tau/gate/router configs.

4. required_outputs:

```text
05_rebuild_from_raw_20260511/03_baselines/v228_final_paper_artifact_freeze_20260623/
    tables/
        final_formal_model_lock.csv
        final_main_result_table.csv
        final_ci_table.csv
        final_claim_lock_table.csv
        final_limitations_table.csv
        final_figure_selection_table.csv
        final_artifact_manifest.csv
        final_guardrail_summary.csv

    reports/
        v228_final_paper_artifact_freeze_cn.md
        manuscript_results_section_draft_cn.md
        manuscript_claim_boundary_notes_cn.md

    figures/
        selected_main_figures/
        selected_appendix_figures/

    logs/
        run_manifest.json
        input_file_hashes.json
        consistency_check.json
        forbidden_scan_report.json
        guardrail_check.json
        file_inventory.json

    v228_final_paper_artifact_freeze_pack.zip
```

`final_formal_model_lock.csv` must contain only:

```text
loose_main_pool,avg_joint_focus
strict_main_pool,peak_floor_090
```

5. stop_condition:
   After v228 pack is generated and validation checks pass, stop.
   If any locked metric, formal model name, CI value, guardrail status, or claim statement conflicts with v225/v226/v227, stop with a failure report only. Do not repair by training, retuning, changing headline models, adding gates/routers/selectors, changing thresholds, removing samples, or reinterpreting diagnostic-only outputs as formal evidence.

6. validation_checks:

```text
python -m py_compile stage03_v228_final_paper_artifact_freeze_20260623.py
full script run passes
ZIP testzip() == None
required files missing == []
formal model lock exactly matches loose=avg_joint_focus and strict=peak_floor_090
main test metrics exactly match v225/v226 within 1e-5
CI values exactly match v226 exported CI tables
v227 claim matrix entries are preserved or narrowed, never expanded
forbidden scan hits == []
no diagnostic-only model appears in final formal tables
no oracle / true-label / fallback row appears in final formal tables
no v222b / v223 / new tau / new gate / new router / new selector artifacts are created
selected figure files exist and are non-empty
file_inventory.json lists all required outputs
guardrail_check.json pass == true
consistency_check.json pass == true
```
