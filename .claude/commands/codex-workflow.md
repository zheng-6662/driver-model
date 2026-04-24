---
description: Claude plans the task, delegates a bounded execution brief to local Codex, then summarizes the result and next steps.
---
Run the standard collaboration workflow for this repository:

Claude plans -> Codex executes -> Claude summarizes.

Use the command arguments below as the user task:

ARGUMENTS: $ARGUMENTS

Workflow:

1. Read `CLAUDE.md` first.
2. If the user is framing the task as 鈥済ive a goal and keep pushing until it is achieved,鈥?also read `04_project_logs/reports/goal_driven_autonomous_workflow.md` and treat this task in goal-driven autonomous mode instead of asking for routine micro-instructions.
3. If useful, use the `codex-coordinator` subagent briefly to sharpen scope and safety constraints.
4. Produce a short internal execution plan before delegating.
5. Convert the task into a bounded Codex execution brief that includes:
   - the concrete task
   - the most relevant files or directories
   - project constraints that Codex must respect
   - acceptance criteria
   - validation expectations
   - default environment expectations: use the conda `predict2` environment for Python execution unless the user explicitly specifies another environment
   - execution preference: use GPU by default for model training, evaluation, diagnostics, or other model programs unless the user explicitly requests CPU or the target script/environment requires otherwise
   - if the user gave explicit red lines, include them verbatim; otherwise default to the repository guardrail of not deleting user files
   - if relevant, remind Codex that high-risk branches are primarily branches that drift away from the core driver-behavior / vehicle-state prediction goal or keep overfitting attention to the same local issue
   - if the same narrow issue has already been attempted `3-4` times, require Codex to justify continuing on that issue instead of switching direction
6. If the task materially advances thesis/model work, the execution brief must explicitly require a detailed update to `04_project_logs/reports/project_progress_master.md` before any compressed summary is returned.
7. Use the Bash tool to run this command exactly:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/ClaudeCode/codex-bridge/claude-codex-entry.ps1" "<COMPACT_CODEX_BRIEF>"
```

8. After Codex returns, if the task materially advanced thesis/model work, verify that the detailed progress logging requirement was satisfied or call out the gap explicitly.
9. After Codex returns, respond in three sections:

## Claude Plan
- Briefly state how you scoped the task and what you asked Codex to do.

## Codex Result
- Summarize what Codex actually reported.
- Mention any files it changed or inspected, if any.

## Claude Summary
- Explain what the result means for the user.
- Call out risks, blockers, or the next best step.

Rules:

- Do not hand off an unbounded vague task to Codex. Ask one concise clarification question if needed.
- Prefer delegating a narrow executable unit, not the whole project.
- Preserve this project's constraints from `CLAUDE.md`, hooks, and protected paths.
- Do not invent Codex output. Only summarize what actually comes back.
- If the task is purely review-oriented, you may recommend `/codex-review` instead.
- For thesis/model work, treat meaningful analysis, inspection, implementation, diagnostics, and literature/plan work as progress that should be written into the project log.

