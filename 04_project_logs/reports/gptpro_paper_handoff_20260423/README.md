# GPT Pro Paper Handoff Pack

Date: `2026-04-23`

This pack is meant to let GPT Pro think about the small-paper direction quickly without re-reading the whole repository history from scratch.

## One-line purpose

Use this pack to design a publishable small-paper story around the current extreme-condition driver-model project, with emphasis on:

- driver-style conditioning
- training-time physio/EEG teacher supervision
- response-fidelity evaluation instead of RMSE-only judgment

## Current safest paper positioning

The safest current story is not:

- "we already have a single globally optimal model"

The safest current story is:

- a multimodal extreme-condition driver-modeling framework
- style-aware conditioning plus training-time physiological teacher guidance
- pooled post-trigger steering-response prediction
- response-structure-aware evaluation and selection protocol

## Recommended read order

1. `PAPER_CONTEXT_SUMMARY_CN.md`
2. `GPTPRO_PROMPT_CN.md`
3. `SCI_WRITING_NOTES_CN.md`
4. `proposal/OPENING_REPORT_PAPER_RELEVANT_SUMMARY_CN.md`
5. `proposal/OPENING_REPORT_KEY_SECTIONS_EXTRACT_CN.md`
6. `context/current-state.md`
7. `context/INPUT_ROLE_AUDIT.md`
8. `context/THESIS_DEFENSE_TABLES.md`
9. `evidence/input_ablation_summary.md`
10. `evidence/bridge_summary.md`
11. `evidence/effectiveness_summary.md`
12. `code/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py`
13. `code/v58_modular/README.md`
14. `tools/recalc_v58_checkpoint_with_current_metrics.py`
15. `tools/run_input_group_ablation.py`
16. `tools/run_effectiveness_followup.py`

## Package contents

- `PAPER_CONTEXT_SUMMARY_CN.md`
  - paper-focused summary of the current project state, evidence chain, and safest claim boundary
- `GPTPRO_PROMPT_CN.md`
  - direct prompt asking GPT Pro to propose a full SCI-level small-paper plan
- `SCI_WRITING_NOTES_CN.md`
  - what can be claimed, what cannot be claimed, likely reviewer concerns, and the minimum missing evidence
- `proposal/`
  - the original opening report plus a paper-relevant summary and section extract
- `context/`
  - current-state, progress logs, decision log, experiment registry, and defense-boundary notes
- `code/`
  - active wrapper and full `v58_modular` training package
- `tools/`
  - recalc, ablation, effectiveness, and audit tools used to produce the current evidence chain
- `configs/`
  - representative run configs for the main keepers and non-promoted branches
- `evidence/`
  - key comparison tables, summaries, and per-run structured summaries
- `assets/`
  - style-cluster result files and selected supporting artifacts

## Key interpretation boundaries

- `style_id` is part of the deployed forward-pass context and can be described as a driver-style prior.
- physio and EEG are not required at inference in the current mainline wording.
- physio and EEG are safest to describe as training-time privileged teacher signals or teacher-state supervision.
- the current active implementation is narrower than the biggest opening-report ambition:
  - it is best framed as pooled post-trigger steering-response prediction
  - not full scene planning
  - not cognition decoding
  - not scene-specific modeling
- the current result is a dual-keeper situation:
  - Run A for response-structure fidelity
  - `baseline_fixed_input` for fit and tail stability
- the paper should not claim a single universally best checkpoint unless new evidence closes that gap.

## What GPT Pro should focus on

- turn the current project into a strong small-paper story even though the model is not globally optimal on every axis
- propose a full section-by-section paper plan:
  - Introduction
  - Data collection and preprocessing
  - Model
  - Experiments and ablations
  - Results and discussion
  - Conclusion
- identify the minimum extra experiments needed to make the paper look professional at SCI level
- give wording that is ambitious but not unsafe
