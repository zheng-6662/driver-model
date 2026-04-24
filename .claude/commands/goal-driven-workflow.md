---
description: Run the repository's goal-driven autonomous workflow so the user can give a target, acceptance criteria, and red lines while Claude/Codex continue pushing until the target is met or an evidence-based blocker is reached.
---
Run the goal-driven autonomous workflow for this repository.

Use the command arguments below as the user task brief:

ARGUMENTS: $ARGUMENTS

Required reading before acting:

1. `CLAUDE.md`
2. `04_project_logs/reports/goal_driven_autonomous_workflow.md`
3. `04_project_logs/reports/goal_driven_target_template.md`

Workflow:

1. Interpret the task in goal-driven mode:
   - identify the `goal`
   - identify or infer `acceptance criteria`
   - identify or infer `red lines`
2. If one of those is missing, infer the narrowest safe default from repo context instead of stopping for routine clarification, unless the ambiguity is genuinely high-risk.
3. Preserve repository guardrails:
   - do not delete user files unless explicitly approved
   - preserve protocol safety unless the user explicitly requests a protocol change
   - log substantive progress to `04_project_logs/reports/project_progress_master.md` before compressed summaries
   - treat 鈥渉igh-risk branch鈥?primarily as a branch that could drift away from the core goal of predicting driver behavior and vehicle-state trends under extreme conditions
4. Use the standard collaboration pattern:
   - Claude scopes and risk-checks
   - Codex executes bounded concrete work
   - Claude reviews results and either continues the next bounded slice or closes out
5. Continue iterating through bounded slices until one of two end states is reached:
   - the acceptance criteria are met
   - an evidence-based blocker is reached and clearly documented
6. For any Python execution or code that may be run, default to the conda `predict2` environment unless the user explicitly specifies another environment.
7. For model training, evaluation, diagnostics, or other model programs, prefer GPU by default unless the user explicitly requests CPU or the target script/environment requires otherwise.
8. When delegating to Codex, the execution brief must include:
   - the concrete bounded task for this slice
   - the relevant files or directories
   - project constraints and red lines
   - acceptance criteria for the slice
   - validation expectations
   - the mandatory progress-log update requirement
   - whether the slice is still directly serving the main target rather than a local side issue
   - whether this issue has already consumed `3-4` attempts and should now be deprioritized
9. Use the Bash tool to run the Codex bridge command when a Codex handoff is appropriate:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/ClaudeCode/codex-bridge/claude-codex-entry.ps1" "<COMPACT_CODEX_BRIEF>"
```

Output:

## Current Goal
- State the current goal, acceptance criteria, and red lines being used.

## Current Slice
- State the bounded slice Claude/Codex are executing now.

## Progress
- Summarize what was completed in this cycle.
- Mention the log entry requirement and whether it was satisfied.

## Status
- Say whether the overall goal is complete, still in progress, or blocked by an evidence-based issue.

## Next Step
- If incomplete, state the next bounded slice.
- If blocked, state the blocker and the best next move.

Rules:

- Do not turn goal-driven mode into an unbounded vague delegation.
- Do not stop just because the user did not provide another small instruction.
- Do not invent Codex output; only report what actually happened.
- Only interrupt the user for decisions that are materially risky or ambiguous.
- Avoid tunnel vision: if the same narrow issue has already been tried `3-4` times without meaningful movement, switch direction unless new evidence strongly justifies one more pass.
- Do not mechanically force a tiny smoke run before every training task; if the path is already well-understood, a full run is allowed.

