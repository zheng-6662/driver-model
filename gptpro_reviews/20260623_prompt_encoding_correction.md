# 2026-06-23 prompt encoding correction

## Correction

The earlier v226 handoff problem was not only a GPTPro response failure. A user
screenshot showed that the Chinese handoff message appeared in ChatGPT Desktop
as mojibake / garbled text. That message was therefore not a valid GPTPro review
prompt.

## Local finding

- Source file `gptpro_reviews/20260622_v226_robustness_ci_result_prompt.md`
  is valid UTF-8 and contains readable Chinese locally.
- The garbling happened during the Desktop send / paste path or display path.
- ASCII-only files are safe locally:
  - `gptpro_reviews/20260622_v226_robustness_ci_result_prompt_ascii.md`
  - `05_rebuild_from_raw_20260511/03_baselines/v227_paper_claim_readiness_pack_20260622/reports/v227_next_gptpro_prompt_ascii.md`
  - `gptpro_reviews/20260623_v227_clean_ascii_handoff_prompt.md`

## Operational change

Future GPTPro handoff prompts should be self-contained and ASCII-only unless the
browser/desktop paste path has been verified to preserve UTF-8 Chinese.

## Superseded interpretation

Previous blocked records that blamed only empty stopped-thinking output or Pro
mode verification are incomplete. The valid interpretation is:

- v226/v227 local work remains valid.
- External GPTPro review was not validly obtained.
- At least one Desktop handoff attempt used a prompt that became garbled in the
GPTPro UI.
- The next handoff should use
  `gptpro_reviews/20260623_v227_clean_ascii_handoff_prompt.md`.

