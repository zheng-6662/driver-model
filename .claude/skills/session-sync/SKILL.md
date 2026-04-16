---
name: session-sync
description: Produce a compact checkpoint of the current project state for resuming later in Claude or handing off to Codex.
argument-hint: [topic-or-goal]
disable-model-invocation: true
allowed-tools: Read, Grep, Glob
---
Create a compact session checkpoint for this project.

Use the current conversation and `$ARGUMENTS` to determine the focus. Read `CLAUDE.md` before summarizing.
If the session materially advanced thesis/model work, ensure the detailed progress has already been written to `reports/project_progress_master.md` before producing the compact checkpoint.

Output exactly these sections:

## Current Goal
## What We Already Know
## Files And Folders In Focus
## Decisions Already Made
## Open Questions
## Next Three Actions
## Codex Handoff Notes

Rules:

- Keep it compact but precise.
- Preserve hard constraints and file paths.
- Separate confirmed facts from tentative assumptions.
- Do not add generic filler.
- Reflect the latest thesis/model progress-log state when relevant, but do not replace the detailed log with the checkpoint.
