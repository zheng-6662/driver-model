# 2026-06-22 v227 GPTPro handoff decision

## Decision

GPTPro did not provide a usable reply for the v227 handoff. The project should
treat this round as externally blocked, not as reviewed or approved.

## Accepted locally

- Keep v227 as a reporting-only package built from locked v225 and v226
  evidence.
- Preserve the formal headline:
  - `loose_main_pool=avg_joint_focus`
  - `strict_main_pool=peak_floor_090`
- Preserve the v226 robustness / confidence-interval audit as the current
  strongest formal evidence for paper-readiness.
- Use the v227 prompt file as the next handoff prompt when the GPTPro channel is
  reachable.

## Rejected / not authorized

- Do not treat v227 as GPTPro approval for any new experiment.
- Do not start v222b/v223.
- Do not tune a new tau, threshold, checkpoint, gate, router, or selector.
- Do not use test results for selection.
- Do not change the formal candidate pool or formal leaderboard.

## Evidence

- v226 output:
  `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v226_formal_robustness_ci_audit_20260622`
- v227 output:
  `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622`
- v227 next GPTPro prompt:
  `F:\data_set_process\data_process\05_rebuild_from_raw_20260511\03_baselines\v227_paper_claim_readiness_pack_20260622\reports\v227_next_gptpro_prompt_ascii.md`
- Bridge failure:
  `Could not verify Pro/进阶 mode. Refusing to send.`

## Next action

When the user restores the GPTPro / ChatGPT Pro browser session, resend the
v227 prompt and wait for one bounded writing/claim/reporting instruction.

