---
description: Check dataset integrity, split safety, protocol consistency, path assumptions, and likely leakage risks for the current task or target scope.
---
Run a project-specific data and protocol safety check for this driver-model repository.

Use the command arguments below as the target scope:

ARGUMENTS: $ARGUMENTS

Before answering:

1. Read the root `CLAUDE.md`.
2. Inspect the most relevant active files under `02_code/final_code`.
3. Use the `split-safety-reviewer` subagent proactively if that improves confidence.
4. Prefer maintained pipeline files over historical archives unless the user explicitly asks for history.
5. Because data/protocol safety checks directly affect thesis/model validity, append a detailed progress entry to `04_project_logs/reports/project_progress_master.md` before returning the compressed summary.

Focus on these checks:

- train / val / test subject leakage
- online-only vs look-ahead input leakage
- protocol family mismatches
- future horizon mismatches
- event anchor definition drift
- split policy drift
- hard-coded path assumptions
- mixing generated outputs with maintained source
- time alignment risks across vehicle, physio, and EEG pipelines

Output exactly these sections:

## Scope Checked
## Files Inspected
## Confirmed Safe Findings
## Risks Or Inconsistencies Found
## Missing Evidence
## Recommended Next Checks

Rules:

- Separate confirmed problems from plausible risks.
- Be conservative and evidence-based.
- If the scope is too vague, ask one concise clarification question.
- The progress entry should capture what was checked, why it was checked, what was confirmed or flagged, and the recommended next checks.

