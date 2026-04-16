---
description: Prepare a Codex-focused review brief for current changes, including repo-specific risks and expected evidence.
---
Create a review handoff for Codex.

Use the command arguments below plus the current repository state to define the review target:

ARGUMENTS: $ARGUMENTS

Before producing the handoff:

1. Read `CLAUDE.md`.
2. Inspect the current working tree if useful.
3. Identify the highest-risk files and behaviors.
4. Highlight project-specific failure modes such as split leakage, anchor drift, horizon mismatches, or editing generated outputs instead of source.
5. If the review target is part of thesis/model progress, include the requirement that meaningful review findings or validated safe conclusions should be written into `reports/project_progress_master.md` before the compressed review summary is returned.
6. Use the `codex-coordinator` subagent if that helps sharpen the review scope.

Output exactly these sections:

## Review Target
## Files To Inspect First
## Project-Specific Risks
## Questions Codex Should Answer
## Evidence Expected Back
## Suggested Prompt For Codex

The final prompt should ask Codex to review for bugs, regressions, data-integrity risks, and missing validation.
If the review materially advances thesis/model understanding, the prompt must also require a detailed progress-log update in `reports/project_progress_master.md`.
