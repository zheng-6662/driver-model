param()

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$runtimeDir = Join-Path $projectRoot "tmp\claude_codex_adapter"
$stateFile = Join-Path $runtimeDir "service_state.json"
$stopFlag = Join-Path $runtimeDir "stop.flag"

if (-not (Test-Path $runtimeDir)) {
    Write-Host "Adapter runtime directory not found." -ForegroundColor Yellow
    exit 0
}

if (-not (Test-Path $stateFile)) {
    Write-Host "Adapter state file not found." -ForegroundColor Yellow
    exit 0
}

$state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
New-Item -ItemType File -Path $stopFlag -Force | Out-Null

foreach ($processId in @($state.adapterPid, $state.servicePid)) {
    if ($processId) {
        try {
            Stop-Process -Id ([int]$processId) -Force -ErrorAction SilentlyContinue
        } catch {
        }
    }
}

Write-Host "Adapter service stop signal sent." -ForegroundColor Green
