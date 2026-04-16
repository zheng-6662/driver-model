---
name: codex-handoff
description: Prepare a structured handoff prompt for Codex using the current task, CLAUDE.md constraints, relevant files, and acceptance criteria.
argument-hint: [task-or-scope]
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash
---
Prepare a high-quality Codex handoff for this project.

Use the current conversation plus `$ARGUMENTS` as the task source. Before writing the handoff:

1. Read the root `CLAUDE.md`.
2. Identify the active code path, likely files, and protocol family.
3. Capture the real constraints that Codex must respect.
4. For any Python execution or code that may be run, default commands and environment notes to the conda `predict2` environment unless the user explicitly specifies another environment.
5. For model training, evaluation, diagnostics, or other model programs, prefer GPU by default unless the user explicitly requests CPU or the target script/environment requires otherwise.
6. If the task materially advances thesis/model work, explicitly require that detailed progress be appended to `reports/project_progress_master.md` before any compressed summary is returned.
7. If the task is too vague to hand off safely, ask one concise clarification question instead of inventing details.

Output exactly these sections:

## Codex Task
- One paragraph that states the task clearly.

## Context Codex Must Know
- Project goal
- Active code path
- Relevant files
- Protocol or split assumptions
- Risks to avoid

## Scope
- In scope
- Out of scope

## Acceptance Criteria
- Concrete definition of done

## Validation
- Commands, files, or checks Codex should use to verify the result

## Suggested Prompt For Codex
Provide a single fenced Markdown block that I can paste into Codex directly.

Keep the handoff specific and implementation-ready. Do not start implementing the task yourself while this skill is active.
For thesis/model tasks, the suggested prompt must include the progress-log path and require recording what was done, why it was done, what was found, and the recommended next step.
