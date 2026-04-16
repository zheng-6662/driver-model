$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$builder = Join-Path $projectRoot "tools\build_progress_dashboard.py"
$dashboard = Join-Path $projectRoot "reports\project_progress_dashboard.html"

Write-Host "[progress-dashboard] rebuilding dashboard..."
py -3 $builder

if ($LASTEXITCODE -ne 0) {
    throw "Dashboard build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $dashboard)) {
    throw "Dashboard file not found: $dashboard"
}

Write-Host "[progress-dashboard] opening $dashboard"
Start-Process $dashboard
