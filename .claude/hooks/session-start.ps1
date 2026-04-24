$payload = @{
  hookSpecificOutput = @{
    hookEventName = "SessionStart"
    additionalContext = @"
Project hook reminder:
- Maintained source of truth is under 02_code/final_code.
- Avoid editing archives, generated run outputs, tmp outputs, artifacts, or backup folders unless the user explicitly asks.
- Use /data-check, /summarize-run, /compare-runs, and /failure-analysis for repo-specific workflows.
- Treat protocol_config.json and split-related files as high-risk because they affect fairness and comparability.
"@
  }
  suppressOutput = $true
}

$payload | ConvertTo-Json -Compress

