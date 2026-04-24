---
description: Analyze failure cases, weak subjects, hard events, and likely causes using metrics, diagnostics, and experiment artifacts.
---
Run a failure analysis for this driver-model repository.

Use the command arguments below as the failure-analysis target:

ARGUMENTS: $ARGUMENTS

Before answering:

1. Read the root `CLAUDE.md`.
2. Use `experiment-auditor` and/or `split-safety-reviewer` proactively if they improve the analysis.
3. Prefer evidence from diagnostics outputs, metrics JSON, run summaries, protocol configs, and active diagnostics scripts.
4. Because this analysis feeds thesis/model understanding, append a detailed progress entry to `04_project_logs/reports/project_progress_master.md` before returning the compressed summary.

Look for failure modes such as:

- low-performing subjects
- hard event subtypes
- reversal or multi-peak cases
- timing lag or alignment issues
- horizon collapse near the tail
- protocol-specific weaknesses
- poor comparability due to data or label differences

Output exactly these sections:

## Failure Scope
## Evidence Reviewed
## Strongest Failure Patterns
## Likely Causes
## What Is Still Uncertain
## Highest-Value Follow-Up Checks

Rules:

- Be explicit about what is measured versus inferred.
- If the target is too vague, ask one concise clarification question.
- Prefer actionable hypotheses over generic advice.
- The progress entry should record the failure scope, evidence reviewed, strongest patterns, likely causes, and highest-value follow-up checks.

