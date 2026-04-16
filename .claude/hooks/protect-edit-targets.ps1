$rawInput = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($rawInput)) {
  exit 0
}

try {
  $payload = $rawInput | ConvertFrom-Json
} catch {
  exit 0
}

function Get-ToolPath {
  param($toolInput)

  if ($null -eq $toolInput) { return $null }

  foreach ($name in @("file_path", "path", "notebook_path")) {
    $prop = $toolInput.PSObject.Properties[$name]
    if ($null -ne $prop -and -not [string]::IsNullOrWhiteSpace([string]$prop.Value)) {
      return [string]$prop.Value
    }
  }

  return $null
}

function Resolve-NormalizedPath {
  param(
    [string]$CandidatePath,
    [string]$BasePath
  )

  if ([string]::IsNullOrWhiteSpace($CandidatePath)) {
    return $null
  }

  if ([System.IO.Path]::IsPathRooted($CandidatePath)) {
    $resolved = [System.IO.Path]::GetFullPath($CandidatePath)
  } else {
    $resolved = [System.IO.Path]::GetFullPath((Join-Path $BasePath $CandidatePath))
  }

  return $resolved.Replace('/', '\').ToLowerInvariant()
}

function Emit-PreToolDecision {
  param(
    [string]$Decision,
    [string]$Reason,
    [string]$AdditionalContext = $null
  )

  $body = @{
    hookSpecificOutput = @{
      hookEventName = "PreToolUse"
      permissionDecision = $Decision
      permissionDecisionReason = $Reason
    }
    suppressOutput = $true
  }

  if (-not [string]::IsNullOrWhiteSpace($AdditionalContext)) {
    $body.hookSpecificOutput.additionalContext = $AdditionalContext
  }

  $body | ConvertTo-Json -Compress
}

$projectRootRaw = if ($env:CLAUDE_PROJECT_DIR) { $env:CLAUDE_PROJECT_DIR } else { [string]$payload.cwd }
$projectRoot = Resolve-NormalizedPath -CandidatePath $projectRootRaw -BasePath $projectRootRaw
$toolPathRaw = Get-ToolPath -toolInput $payload.tool_input
$targetPath = Resolve-NormalizedPath -CandidatePath $toolPathRaw -BasePath ([string]$payload.cwd)

if ([string]::IsNullOrWhiteSpace($targetPath) -or [string]::IsNullOrWhiteSpace($projectRoot)) {
  exit 0
}

if (-not $targetPath.StartsWith($projectRoot)) {
  exit 0
}

$blockedSegments = @(
  "\datasetprocess\多模态数据\01_历史入口归档\",
  "\datasetprocess\多模态数据\程序运行结果\",
  "\tmp\",
  "\artifacts\",
  "_backup_"
)

foreach ($segment in $blockedSegments) {
  if ($targetPath.Contains($segment)) {
    Emit-PreToolDecision `
      -Decision "deny" `
      -Reason "Editing archived, generated, tmp, artifacts, or backup paths is blocked by this project hook. Prefer maintained source under datasetprocess/final_code, tools, reports, or root docs instead." `
      -AdditionalContext "The attempted target was $targetPath. This repository mixes active source with archived/generated outputs, so do not edit these paths unless the user explicitly redirects the task."
    exit 0
  }
}

$isTrainingConfig = $targetPath -match "\\datasetprocess\\final_code\\model\\training\\.*(protocol_config\.json|frozen_subject_split.*\.json)$"
if ($isTrainingConfig) {
  Emit-PreToolDecision `
    -Decision "ask" `
    -Reason "This file controls protocol or split behavior. Confirm only if you intentionally want to change fairness-critical experiment definitions." `
    -AdditionalContext "Before editing this file, check CLAUDE.md and verify whether the change affects split policy, anchor definitions, labels, future horizon, or comparability with old runs."
  exit 0
}

exit 0
