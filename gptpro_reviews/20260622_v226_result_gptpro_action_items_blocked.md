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
