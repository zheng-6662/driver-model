---
name: codex-review
description: Prepare a Codex-oriented review brief for current changes, including review focus areas, high-risk files, and what evidence should come back.
argument-hint: [diff-or-scope]
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash(git status*), Bash(git diff*), Bash(git log*)
---
Create a review handoff for Codex.

Use `$ARGUMENTS` plus the current repository state to define the review target. Prefer the current working tree diff if no explicit scope is given.

Before producing the handoff:

1. Read `CLAUDE.md`.
2. Inspect `git status` and `git diff` when useful.
3. Identify the highest-risk files and behaviors.
4. Highlight project-specific failure modes such as split leakage, anchor drift, horizon mismatches, or editing generated outputs instead of source.
5. If the review may involve Python execution or runnable checks, default environment notes to the conda `predict2` environment unless the user explicitly specifies another environment.
6. If the review may involve model training, evaluation, diagnostics, or other model programs, prefer GPU by default unless the user explicitly requests CPU or the target script/environment requires otherwise.
7. If the review materially advances thesis/model understanding, require that the meaningful findings or validated safe conclusions be written into `reports/project_progress_master.md` before the compressed review summary is returned.

Output exactly these sections:

## Review Target
## Files To Inspect First
## Project-Specific Risks
## Questions Codex Should Answer
## Evidence Expected Back
## Suggested Prompt For Codex

The final prompt should ask Codex to review for bugs, regressions, data-integrity risks, and missing validation rather than rewrite the code.
For thesis/model work, also require a detailed progress-log update in `reports/project_progress_master.md`.
