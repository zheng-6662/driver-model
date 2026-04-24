---
description: Prepare a structured handoff prompt for Codex from the current task, files, and project constraints.
---
Prepare a high-quality Codex handoff for this project.

Use the current conversation plus the command arguments below as the task source:

ARGUMENTS: $ARGUMENTS

Before writing the handoff:

1. Read the root `CLAUDE.md`.
2. Identify the active code path, likely files, and protocol family.
3. Capture the real constraints that Codex must respect.
4. Use the `codex-coordinator` subagent if that will improve the task split.
5. For any Python execution or code that may be run, default commands and environment notes to the conda `predict2` environment unless the user explicitly specifies another environment.
6. For model training, evaluation, diagnostics, or other model programs, prefer GPU by default unless the user explicitly requests CPU or the target script/environment requires otherwise.
7. If the task touches paper-writing, driver-response modeling, experiment analysis, literature organization, or any other work that materially advances the thesis/model progress, explicitly require that the latest detailed progress be appended to `04_project_logs/reports/project_progress_master.md` before any compressed summary is returned.
8. If the task is too vague to hand off safely, ask one concise clarification question instead of inventing details.

Output exactly these sections:

## Codex Task
## Context Codex Must Know
## Scope
## Acceptance Criteria
## Validation
## Suggested Prompt For Codex

The final prompt should be ready to paste into Codex directly.
If the task is related to thesis/model progress, the prompt must explicitly tell Codex to:

- write a detailed progress entry into `04_project_logs/reports/project_progress_master.md` before giving a compressed summary
- record what was done, why it was done, what was found, and the recommended next step

