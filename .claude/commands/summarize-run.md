---
description: Summarize one experiment run folder using run_summary.json, config.json, metrics JSON, and nearby protocol evidence.
---
Summarize a single experiment run from this repository.

Use the command arguments below as the target run folder or experiment scope:

ARGUMENTS: $ARGUMENTS

Before answering:

1. Read the root `CLAUDE.md`.
2. Use the `experiment-auditor` subagent proactively when helpful.
3. Prefer concrete evidence from:
   - `run_summary.json`
   - `config.json`
   - metrics JSON
   - protocol config files
   - copied run scripts only as supporting evidence, not as primary source
4. Because run summaries feed thesis/model progress, append a detailed progress entry to `reports/project_progress_master.md` before returning the compressed run summary.

Your goal is to produce a concise but decision-useful summary of what this run did and what happened.

Output exactly these sections:

## Run
## What This Run Is Testing
## Protocol And Data Assumptions
## Key Configuration
## Main Results
## Reliability Notes
## Best Next Action

Rules:

- If multiple candidate run folders match, ask one concise clarification question.
- Distinguish measured results from your interpretation.
- Mention fairness or comparability caveats if protocol or data assumptions are unclear.
- The progress entry should record what run was summarized, what it was testing, the main measured results, reliability notes, and the best next action.
