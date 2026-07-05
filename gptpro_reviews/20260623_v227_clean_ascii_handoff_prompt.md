# GPTPro clean ASCII handoff: v226 audit complete and v227 reporting fallback complete

Important correction:

The previous Chinese handoff message may have appeared as mojibake / garbled
text in ChatGPT Desktop. Please ignore that garbled message. This message is
ASCII-only and fully self-contained.

Your role:

Act as an external reviewer. Give exactly one bounded next instruction for
Codex. Do not brainstorm broadly.

Local status:

- v225 formal route reconstruction evidence pack completed.
- v226 formal robustness / confidence-interval audit completed.
- v227 paper / claim readiness package completed as a reporting-only fallback.
- v227 was created only from existing v225 and v226 evidence.
- No model was trained in v226 or v227.
- No threshold or tau was searched.
- No gate, router, or selector was created.
- No v222b or v223 was run.
- No formal headline changed.

Formal locked result:

- loose_main_pool formal model: avg_joint_focus
- strict_main_pool formal model: peak_floor_090

Key v226 verification:

- py_compile: pass
- full script run: pass
- zip testzip: None
- required files missing: []
- formal lock reproduction: pass
- metric reproduction tolerance: within 1e-5
- leakage guard: pass
- forbidden scan: pass
- table alignment: pass
- no diagnostic-only row in formal output: pass

Key v226 test metrics:

- loose_main_pool / avg_joint_focus:
  - n: 184
  - RMSE: 0.544884
  - tail RMSE: 0.629752
  - sample bootstrap RMSE CI: 0.496066 to 0.593811
  - subject-block bootstrap RMSE CI: 0.428783 to 0.599684
  - top 20 pct tail SSE share: 0.659320

- strict_main_pool / peak_floor_090:
  - n: 174
  - RMSE: 0.571770
  - tail RMSE: 0.658306
  - sample bootstrap RMSE CI: 0.511036 to 0.635521
  - subject-block bootstrap RMSE CI: 0.473689 to 0.615000
  - top 20 pct tail SSE share: 0.672493

v226 readiness decision:

- accepted_for_paper_main_result: true
- needs_new_model: false
- needs_gate_or_router: false
- needs_more_diagnostic_only: false

v227 reporting-only outputs:

- paper_main_result_table.csv
- paper_claim_support_matrix.csv
- paper_limitation_table.csv
- formal_guardrail_summary.csv
- formal_artifact_manifest.csv
- figure_selection_index.csv
- v227_paper_claim_readiness_cn.md
- v227_paper_claim_readiness_pack.zip

Current guardrails:

- Do not request v222b or v223.
- Do not request new model training.
- Do not request a new threshold or tau search.
- Do not request a new gate, router, or selector.
- Do not request formal leaderboard or headline changes.
- Do not request test-based retuning.
- If you believe any forbidden direction is now unlocked, you must explicitly
  state which stop condition was overturned and what leakage / validation /
  test-discipline guardrails make it safe. Otherwise keep the next step within
  reporting, writing, claim audit, table polish, or final package readiness.

Please answer with exactly these six items:

1. Accept or reject v227 as a valid reporting-only fallback, with exact reason.
2. The next local task name and version.
3. Allowed input files or directories.
4. Required output directory and required files.
5. Exact stop condition.
6. Validation checks required before Codex reports back to you.

