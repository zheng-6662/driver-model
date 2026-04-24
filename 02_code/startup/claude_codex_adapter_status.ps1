param()

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$runtimeDir = Join-Path $projectRoot "tmp\claude_codex_adapter"
$stateFile = Join-Path $runtimeDir "service_state.json"

if (-not (Test-Path $stateFile)) {
    Write-Host "Adapter state file not found." -ForegroundColor Yellow
    exit 0
}

$state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
$healthy = $false

if ($state.baseUrl) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$($state.baseUrl)/health" -TimeoutSec 3
        $healthy = $response.StatusCode -eq 200
    } catch {
        $healthy = $false
    }
}

[pscustomobject]@{
    Status     = $state.status
    Healthy    = $healthy
    BaseUrl    = $state.baseUrl
    Port       = $state.port
    ServicePid = $state.servicePid
    AdapterPid = $state.adapterPid
    UpdatedAt  = $state.updatedAt
} | Format-List
