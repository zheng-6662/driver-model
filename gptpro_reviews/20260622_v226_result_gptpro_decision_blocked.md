# GPTPro decision record: v226 result report blocked

## Accepted

- The project accepts that v226 local execution is complete based on local
  evidence:
  - `py_compile` passed.
  - Full v226 script run passed.
  - ZIP integrity passed.
  - Required files missing list is empty.
  - Formal model lock is exact:
    - `loose_main_pool = avg_joint_focus`
    - `strict_main_pool = peak_floor_090`
  - Metric reproduction is within `1e-5`.
  - Leakage guard, forbidden scan, and table alignment passed.
  - Required figure counts passed.

## Rejected or deferred

- No new formal experiment, model training, threshold search, router, gate,
  `v222b`, or `v223` is accepted without a new valid GPTPro instruction.
- The garbled desktop prompt and the empty `已停止思考` outputs are rejected as
  valid GPTPro advice.
- The Chrome path is deferred because it requires user login/account action.

## Operating boundary until GPTPro is reachable

- The last valid GPTPro instruction says v226 should stop after the pack passes.
- Any local fallback work must therefore be non-model, non-selection, and
  reporting-only.
- Acceptable local fallback scope:
  - organize v225+v226 evidence for writing;
  - create claim/readiness tables from existing formal outputs;
  - package a paper-facing report from already locked evidence.
- Forbidden local fallback scope:
  - train or tune any model;
  - run `v222b` or `v223`;
  - change tau, threshold, gate, router, or formal headline;
  - use test performance to choose a new configuration.

## Evidence links

- v226 report:
  `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/reports/v226_formal_robustness_ci_audit_cn.md`
- v226 ZIP:
  `05_rebuild_from_raw_20260511/03_baselines/v226_formal_robustness_ci_audit_20260622/v226_formal_robustness_ci_audit_pack.zip`
- last valid GPTPro v226 decision:
  `gptpro_reviews/20260622_v226_formal_robustness_ci_gptpro_decision.md`
*** Add File: gptpro_reviews/20260622_v226_result_gptpro_action_items_blocked.md
# GPTPro action items: blocked handoff fallback

## Immediate action items

- Preserve the failed GPTPro handoff record.
- Do not execute new modeling work while no new GPTPro instruction exists.
- Continue only with reporting-only material that uses existing v225+v226
  evidence.

## Safe fallback work item

Create a bounded `v227` paper/claim readiness package from existing v225 and
v226 outputs only.

Allowed scope:

- Read v225 formal evidence pack outputs.
- Read v226 robustness / CI audit outputs.
- Produce writing-facing summaries, claim tables, limitation tables,
  reproducibility checklists, and artifact indexes.
- Reuse existing figures/tables by reference or copied package inclusion.

Forbidden scope:

- No model training.
- No new thresholds, tau, gate, router, or selector.
- No formal headline changes.
- No test-based retuning.
- No diagnostic-only row promotion.

Stop condition:

- Stop when the v227 reporting package is generated, locally validated, zipped,
  and note-layer state is updated.
