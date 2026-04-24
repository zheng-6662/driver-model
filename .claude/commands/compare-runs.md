---
description: Compare two experiment runs or protocol variants and explain differences in setup, fairness, and outcome.
---
Compare two experiment runs in this driver-model repository.

Use the command arguments below to identify the two runs or scopes to compare:

ARGUMENTS: $ARGUMENTS

Before answering:

1. Read the root `CLAUDE.md`.
2. Use the `experiment-auditor` subagent proactively when helpful.
3. Compare evidence from:
   - `run_summary.json`
   - `config.json`
   - metrics JSON
   - protocol config files
   - relevant training or diagnostics source files when needed
4. Because run comparison directly affects thesis/model conclusions, append a detailed progress entry to `04_project_logs/reports/project_progress_master.md` before returning the compressed comparison summary.

Pay special attention to:

- protocol family differences
- split differences
- label / anchor / target differences
- horizon differences
- optimization-only differences versus data-definition differences
- whether the comparison is fair

Output exactly these sections:

## Compared Runs
## What Stayed The Same
## What Changed
## Result Differences
## Fairness And Validity Notes
## Most Likely Explanation
## Recommended Next Experiment

Rules:

- Separate factual differences from causal hypotheses.
- If the command arguments do not clearly identify two targets, ask one concise clarification question.
- The progress entry should record what runs were compared, why this comparison mattered, what was found, and the best next experiment.

