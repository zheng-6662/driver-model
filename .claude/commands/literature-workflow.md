---
description: Use the restored ScholarAIO/Zotero literature workflow for search, screening, import, and Claude/Codex collaboration in this repository.
---
Run the restored literature workflow for this repository.

Use the command arguments below as the user's literature task:

ARGUMENTS: $ARGUMENTS

Before acting:

1. Read the root `CLAUDE.md`.
2. Read `reports/codex_academic_zotero_workflow.md`.
3. If the request does not clearly state both the topic and the desired action, ask one concise clarification question instead of guessing.
4. For any substantial literature search, import, organization, or evidence synthesis work, append a detailed progress entry to `reports/project_progress_master.md` before returning the compressed summary.

Default behavior for this command:

- Prefer the local ScholarAIO / Zotero workflow over generic web browsing.
- Prefer `tools/academic_search_to_zotero.py` for targeted search and import.
- Prefer ScholarAIO CLI for broader online search or JSON result inspection.
- Use the existing Codex bridge only when the task is multi-step, execution-heavy, or the user explicitly wants Claude/Codex collaboration.

Preferred execution paths:

### 1. Search only

Use one of these:

```powershell
py -3.11 .\tools\academic_search_to_zotero.py search "<QUERY>" --limit 8
```

or, when broader result inspection is useful:

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONPATH="D:/ClaudeCode/codex-home/scholaraio"
$env:SCHOLARAIO_CONFIG="D:/ClaudeCode/codex-home/scholaraio/config.yaml"
py -3.11 -m scholaraio.cli online-search "<QUERY>" --json
```

### 2. Import by keyword / DOI / URL

Use the local Zotero workflow:

```powershell
py -3.11 .\tools\academic_search_to_zotero.py import-query "<QUERY>" --limit 5 --pick 1 --collection "研究生论文/自动导入" --download-pdf
```

```powershell
py -3.11 .\tools\academic_search_to_zotero.py import-doi "<DOI>" --collection "研究生论文/自动导入" --download-pdf
```

```powershell
py -3.11 .\tools\academic_search_to_zotero.py import-url "<URL>" --collection "研究生论文/自动导入" --download-pdf
```

If the user gives a specific collection path, use that instead of the default.

### 3. Claude + Codex collaboration

If the task is not just a small search or single-paper import, use the Bash tool to run this command with a bounded literature brief:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:/ClaudeCode/codex-bridge/claude-codex-entry.ps1" "<LITERATURE_BRIEF>"
```

That brief should explicitly tell Codex to:

- use the local ScholarAIO / Zotero workflow instead of generic browsing when possible
- inspect `reports/codex_academic_zotero_workflow.md`
- record detailed progress in `reports/project_progress_master.md` before returning a compressed summary

Compliance boundaries:

- Do not bypass paywalls, login walls, CAPTCHAs, or site protections.
- Do not claim a paper was imported unless the command output confirms it.
- Distinguish search results from actually imported papers.
- Distinguish measured metadata from your own interpretation.

Output exactly these sections:

## Literature Goal
## Actions Run
## Evidence Or Import Results
## Best Next Step

Rules:

- Keep the final response concrete and execution-oriented.
- Mention the exact command path used: local search/import, ScholarAIO CLI, or Codex bridge.
- If something failed, say whether it failed at search, metadata retrieval, PDF retrieval, or Zotero import.
