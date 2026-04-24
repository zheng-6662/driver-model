---
description: Create a compact checkpoint of the current session for later resume in Claude or handoff to Codex.
---
Create a compact session checkpoint for this project.

Use the current conversation and the command arguments below to determine the focus:

ARGUMENTS: $ARGUMENTS

Before summarizing:

1. Read `CLAUDE.md`.
2. Preserve hard constraints and important file paths.
3. Separate confirmed facts from tentative assumptions.
4. If the session includes work that materially advanced thesis/model progress, make sure the detailed progress has already been appended to `04_project_logs/reports/project_progress_master.md` before producing the compact checkpoint.

Output exactly these sections:

## Current Goal
## What We Already Know
## Files And Folders In Focus
## Decisions Already Made
## Open Questions
## Next Three Actions
## Codex Handoff Notes

Keep it compact, concrete, and resume-friendly.
When relevant, reflect the latest thesis/model progress-log state without replacing the detailed log itself.

