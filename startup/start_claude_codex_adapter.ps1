param(
    [string]$ListenHost = "127.0.0.1",
    [int]$PreferredPort = 8417,
    [int]$MaxPort = 8427,
    [string]$BackendBaseUrl = "http://localhost:8317/v1",
    [string]$BackendApiKey = "sk-dummy",
    [string]$BackendModel = "gpt-5.4",
    [string]$ProxyApiKey = "sk-claude-codex-proxy"
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$runtimeDir = Join-Path $projectRoot "tmp\claude_codex_adapter"
$stateFile = Join-Path $runtimeDir "service_state.json"
$stopFlag = Join-Path $runtimeDir "stop.flag"
$serviceLog = Join-Path $runtimeDir "supervisor.log"
$supervisorScript = Join-Path $PSScriptRoot "claude_codex_adapter_supervisor.ps1"
$cpaBootstrapScript = Join-Path $PSScriptRoot "openclaw_cpa_oneclick.ps1"

if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
}

function Read-State {
    if (-not (Test-Path $stateFile)) {
        return $null
    }

    try {
        return Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Test-ProcessAlive {
    param([int]$ProcessId)

    if (-not $ProcessId) {
        return $false
    }

    return $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Test-ServiceHealth {
    param([string]$BaseUrl)

    if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
        return $false
    }

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/health" -TimeoutSec 3
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-BackendHealth {
    param(
        [string]$BaseUrl,
        [string]$ApiKey
    )

    if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
        return $false
    }

    $headers = @{}
    if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
        $headers.Authorization = "Bearer $ApiKey"
    }

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/models" -Headers $headers -TimeoutSec 5
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Ensure-BackendReady {
    if (Test-BackendHealth -BaseUrl $BackendBaseUrl -ApiKey $BackendApiKey) {
        return
    }

    if (-not (Test-Path $cpaBootstrapScript)) {
        throw "CPA bootstrap script not found: $cpaBootstrapScript"
    }

    Write-Host "CPA backend is not ready. Starting OpenClaw + CPA..." -ForegroundColor Yellow
    & $cpaBootstrapScript -SkipPrompt -SkipBrowser -NoPause
    if ($LASTEXITCODE -ne 0) {
        throw "OpenClaw + CPA bootstrap failed with exit code $LASTEXITCODE."
    }

    if (-not (Test-BackendHealth -BaseUrl $BackendBaseUrl -ApiKey $BackendApiKey)) {
        throw "CPA backend is still unavailable after bootstrap: $BackendBaseUrl"
    }
}

Ensure-BackendReady

$state = Read-State
if ($state -and (Test-ProcessAlive -ProcessId ([int]$state.servicePid)) -and (Test-ServiceHealth -BaseUrl $state.baseUrl)) {
    Write-Host "Adapter service already healthy at $($state.baseUrl)" -ForegroundColor Yellow
    return $state
}

if (Test-Path $stopFlag) {
    Remove-Item -LiteralPath $stopFlag -Force
}

if ($state -and (Test-ProcessAlive -ProcessId ([int]$state.servicePid))) {
    try {
        Stop-Process -Id ([int]$state.servicePid) -Force -ErrorAction SilentlyContinue
    } catch {
    }
}

$supervisorArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $supervisorScript,
    "-ListenHost", $ListenHost,
    "-PreferredPort", $PreferredPort,
    "-MaxPort", $MaxPort,
    "-BackendBaseUrl", $BackendBaseUrl,
    "-BackendApiKey", $BackendApiKey,
    "-BackendModel", $BackendModel,
    "-ProxyApiKey", $ProxyApiKey
)

$serviceProcess = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $supervisorArgs `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru

$healthyState = $null
for ($attempt = 1; $attempt -le 20; $attempt++) {
    Start-Sleep -Seconds 1
    $healthyState = Read-State
    if (
        $healthyState -and
        $healthyState.status -eq "healthy" -and
        (Test-ServiceHealth -BaseUrl $healthyState.baseUrl)
    ) {
        break
    }
    $healthyState = $null
}

if (-not $healthyState) {
    Write-Host "Adapter service failed to become healthy. Service PID=$($serviceProcess.Id)" -ForegroundColor Red
    Write-Host "State file: $stateFile" -ForegroundColor DarkGray
    Write-Host "Supervisor log: $serviceLog" -ForegroundColor DarkGray
    exit 1
}

Write-Host "Adapter service ready at $($healthyState.baseUrl)" -ForegroundColor Green
Write-Host "Service PID: $($healthyState.servicePid)" -ForegroundColor DarkGray
Write-Host "Adapter PID: $($healthyState.adapterPid)" -ForegroundColor DarkGray
Write-Host "Supervisor log: $serviceLog" -ForegroundColor DarkGray

return $healthyState
