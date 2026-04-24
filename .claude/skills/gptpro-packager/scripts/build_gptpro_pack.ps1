[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackName,

    [string]$ReportRoot = "04_project_logs/reports",
    [string]$DailyFile,
    [string[]]$TopLevelFiles = @(),
    [string[]]$EvidenceFiles = @(),
    [string[]]$ConfigFiles = @(),
    [string[]]$ProtocolFiles = @(),
    [string[]]$CodeFiles = @(),
    [string[]]$CodeDirs = @(),
    [switch]$NoZip
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..\\..")).Path
}

function Resolve-RepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }

    return (Resolve-Path -LiteralPath (Join-Path $script:RepoRoot $PathValue)).Path
}

function Assert-ChildPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePath,
        [Parameter(Mandatory = $true)]
        [string]$CandidatePath
    )

    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    $candidateFull = [System.IO.Path]::GetFullPath($CandidatePath)

    if (-not $candidateFull.StartsWith($baseFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside base path. base=$baseFull candidate=$candidateFull"
    }
}

function Parse-CopySpec {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Spec,
        [Parameter(Mandatory = $true)]
        [string]$DefaultTargetDir
    )

    $parts = $Spec -split "\|", 2
    $sourceSpec = $parts[0].Trim()
    $destSpec = if ($parts.Count -gt 1 -and $parts[1].Trim()) { $parts[1].Trim() } else { [System.IO.Path]::GetFileName($sourceSpec) }

    return [pscustomobject]@{
        Source = (Resolve-RepoPath -PathValue $sourceSpec)
        DestRelative = (Join-Path $DefaultTargetDir $destSpec)
    }
}

function Parse-DirSpec {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Spec,
        [Parameter(Mandatory = $true)]
        [string]$DefaultTargetDir
    )

    $parts = $Spec -split "\|", 2
    $sourceSpec = $parts[0].Trim()
    $sourcePath = Resolve-RepoPath -PathValue $sourceSpec
    $leafName = Split-Path -Leaf $sourcePath
    $destSpec = if ($parts.Count -gt 1 -and $parts[1].Trim()) { $parts[1].Trim() } else { $leafName }

    return [pscustomobject]@{
        Source = $sourcePath
        DestRelative = (Join-Path $DefaultTargetDir $destSpec)
    }
}

function Ensure-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    if (-not (Test-Path -LiteralPath $PathValue)) {
        New-Item -ItemType Directory -Path $PathValue -Force | Out-Null
    }
}

function Copy-FileItem {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$DestPath
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
        throw "File not found: $SourcePath"
    }

    $parentDir = Split-Path -Parent $DestPath
    Ensure-Directory -PathValue $parentDir
    Copy-Item -LiteralPath $SourcePath -Destination $DestPath -Force
}

function Copy-DirItem {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$DestPath
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Container)) {
        throw "Directory not found: $SourcePath"
    }

    if (Test-Path -LiteralPath $DestPath) {
        Remove-Item -LiteralPath $DestPath -Recurse -Force
    }

    $parentDir = Split-Path -Parent $DestPath
    Ensure-Directory -PathValue $parentDir
    Copy-Item -LiteralPath $SourcePath -Destination $DestPath -Recurse -Force

    Get-ChildItem -LiteralPath $DestPath -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.PSIsContainer -and $_.Name -eq "__pycache__" } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
}

function Get-LatestDailyFile {
    $dailyDir = Join-Path $script:RepoRoot "04_project_logs/reports/progress/daily"
    $latest = Get-ChildItem -LiteralPath $dailyDir -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        throw "No daily log found under $dailyDir"
    }
    return $latest.FullName
}

$RepoRoot = Get-RepoRoot
$script:RepoRoot = $RepoRoot

$reportRootPath = if ([System.IO.Path]::IsPathRooted($ReportRoot)) { $ReportRoot } else { Join-Path $RepoRoot $ReportRoot }
$reportRootPath = [System.IO.Path]::GetFullPath($reportRootPath)
Ensure-Directory -PathValue $reportRootPath

$packDir = Join-Path $reportRootPath $PackName
$packZip = "$packDir.zip"
Assert-ChildPath -BasePath $reportRootPath -CandidatePath $packDir

Ensure-Directory -PathValue $packDir

$managedSubdirs = @("context", "evidence", "configs", "protocol", "code")
foreach ($subdirName in $managedSubdirs) {
    $subdirPath = Join-Path $packDir $subdirName
    Ensure-Directory -PathValue $subdirPath
    Assert-ChildPath -BasePath $packDir -CandidatePath $subdirPath
    Get-ChildItem -LiteralPath $subdirPath -Force -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force
    }
}

if (Test-Path -LiteralPath $packZip) {
    Remove-Item -LiteralPath $packZip -Force
}

$resolvedDailyFile = if ($DailyFile) { Resolve-RepoPath -PathValue $DailyFile } else { Get-LatestDailyFile }

$defaultContextSpecs = @(
    "04_project_logs/references/current-state.md|current-state.md",
    ("{0}|{1}" -f $resolvedDailyFile, (Split-Path -Leaf $resolvedDailyFile)),
    "04_project_logs/reports/progress/decision_log.md|decision_log.md",
    "04_project_logs/reports/progress/experiment_registry.md|experiment_registry.md"
)

$defaultProtocolSpecs = @(
    "02_code/final_code/model/training/protocol_primary_control_v2_context_full2s/protocol_config.json|protocol_config.json",
    "02_code/final_code/model/training/protocol_primary_control_v2_context_full2s/frozen_subject_split.json|frozen_subject_split.json"
)

$defaultCodeFileSpecs = @(
    "02_code/final_code/model/training/future_steer_event_rollpeak_transformer_v5_8_amp_tuned_fixed.py",
    "02_code/tools/recalc_v58_checkpoint_with_current_metrics.py",
    "02_code/tools/run_effectiveness_followup.py",
    "02_code/tools/summarize_effectiveness_followup.py"
)

$defaultCodeDirSpecs = @(
    "02_code/final_code/model/training/v58_modular|v58_modular"
)

$optionalDefaultCodeFiles = @(
    "tmp/recalc_v58_metrics_shim_20260423.py"
)

foreach ($spec in $TopLevelFiles) {
    $item = Parse-CopySpec -Spec $spec -DefaultTargetDir "."
    $destPath = Join-Path $packDir $item.DestRelative
    Copy-FileItem -SourcePath $item.Source -DestPath $destPath
}

foreach ($spec in $defaultContextSpecs) {
    $item = Parse-CopySpec -Spec $spec -DefaultTargetDir "context"
    $destPath = Join-Path $packDir $item.DestRelative
    Copy-FileItem -SourcePath $item.Source -DestPath $destPath
}

foreach ($spec in $EvidenceFiles) {
    $item = Parse-CopySpec -Spec $spec -DefaultTargetDir "evidence"
    $destPath = Join-Path $packDir $item.DestRelative
    Copy-FileItem -SourcePath $item.Source -DestPath $destPath
}

foreach ($spec in $ConfigFiles) {
    $item = Parse-CopySpec -Spec $spec -DefaultTargetDir "configs"
    $destPath = Join-Path $packDir $item.DestRelative
    Copy-FileItem -SourcePath $item.Source -DestPath $destPath
}

foreach ($spec in ($defaultProtocolSpecs + $ProtocolFiles)) {
    $item = Parse-CopySpec -Spec $spec -DefaultTargetDir "protocol"
    $destPath = Join-Path $packDir $item.DestRelative
    Copy-FileItem -SourcePath $item.Source -DestPath $destPath
}

foreach ($spec in ($defaultCodeFileSpecs + $CodeFiles)) {
    $item = Parse-CopySpec -Spec $spec -DefaultTargetDir "code"
    $destPath = Join-Path $packDir $item.DestRelative
    Copy-FileItem -SourcePath $item.Source -DestPath $destPath
}

foreach ($optionalSpec in $optionalDefaultCodeFiles) {
    $optionalPath = Join-Path $RepoRoot $optionalSpec
    if (Test-Path -LiteralPath $optionalPath -PathType Leaf) {
        $item = Parse-CopySpec -Spec $optionalSpec -DefaultTargetDir "code"
        $destPath = Join-Path $packDir $item.DestRelative
        Copy-FileItem -SourcePath $item.Source -DestPath $destPath
    }
}

foreach ($spec in ($defaultCodeDirSpecs + $CodeDirs)) {
    $item = Parse-DirSpec -Spec $spec -DefaultTargetDir "code"
    $destPath = Join-Path $packDir $item.DestRelative
    Copy-DirItem -SourcePath $item.Source -DestPath $destPath
}

$requiredTopLevelDocs = @(
    "README.md",
    "PROJECT_STATUS_AND_CODEGEN_BRIEF_CN.md",
    "GPTPRO_PROMPT_CN.md"
)

foreach ($docName in $requiredTopLevelDocs) {
    $docPath = Join-Path $packDir $docName
    if (-not (Test-Path -LiteralPath $docPath -PathType Leaf)) {
        Write-Warning "Top-level doc missing: $docPath"
    }
}

if (-not $NoZip) {
    Compress-Archive -Path $packDir -DestinationPath $packZip -Force
}

Write-Host "Pack directory: $packDir"
if (-not $NoZip) {
    Write-Host "Pack zip: $packZip"
}
Write-Host "Files in pack:" (Get-ChildItem -LiteralPath $packDir -Recurse -File | Measure-Object).Count
