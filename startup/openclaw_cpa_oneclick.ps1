param(
    [switch]$SkipPrompt
)

$ErrorActionPreference = 'Stop'

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "[OpenClaw-CPA] $Message" -ForegroundColor Cyan
}

function Invoke-WslBash {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    & wsl -d Ubuntu-24.04 -- bash -lc $Command
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed: $Command"
    }
}

function Wait-ForUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [hashtable]$Headers = @{},
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Url -Headers $Headers -UseBasicParsing -TimeoutSec 5 | Out-Null
            return $true
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    return $false
}

Write-Host "OpenClaw + CPA one-click startup" -ForegroundColor Green
Write-Host "Please make sure Docker Desktop and your VPN / proxy are already connected." -ForegroundColor Yellow
if (-not $SkipPrompt) {
    $ready = Read-Host "Type Y to continue"
    if ($ready -notin @('Y', 'y')) {
        Write-Host "Cancelled. Nothing was changed." -ForegroundColor Yellow
        exit 0
    }
}

$distro = 'Ubuntu-24.04'
$cpaDir = '/home/administrator/apps/CLIProxyAPI'
$managementUrl = 'http://localhost:8317/management.html'
$modelsUrl = 'http://127.0.0.1:8317/v1/models'

Write-Step "Starting or refreshing the CPA container"
Invoke-WslBash "cd $cpaDir; docker compose -f docker-compose.local.yml up -d --force-recreate"

Write-Step "Restarting the OpenClaw gateway"
Invoke-WslBash 'XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user restart openclaw-gateway.service'

Write-Step "Reading the OpenClaw token"
$openClawConfigPath = "\\wsl$\$distro\home\administrator\.openclaw\openclaw.json"
$openClawConfig = Get-Content -Raw -LiteralPath $openClawConfigPath | ConvertFrom-Json
$gatewayPort = [int]$openClawConfig.gateway.port
$gatewayToken = [string]$openClawConfig.gateway.auth.token
$dashboardUrl = "http://127.0.0.1:$gatewayPort/#token=$gatewayToken"

Write-Step "Waiting for the CPA API"
$cpaReady = Wait-ForUrl -Url $modelsUrl -Headers @{ Authorization = 'Bearer sk-dummy' } -TimeoutSeconds 60
if (-not $cpaReady) {
    Write-Host "CPA API did not become ready within 60 seconds." -ForegroundColor Red
    exit 1
}

Write-Step "Waiting for the OpenClaw dashboard"
$dashboardReady = Wait-ForUrl -Url $dashboardUrl -TimeoutSeconds 30
if (-not $dashboardReady) {
    Write-Host "OpenClaw dashboard did not become ready within 30 seconds." -ForegroundColor Red
    exit 1
}

Write-Step "Opening browser pages"
Start-Process $managementUrl
Start-Process $dashboardUrl

Write-Step "Startup complete"
Write-Host "CPA management page: $managementUrl" -ForegroundColor Green
Write-Host "OpenClaw dashboard:  $dashboardUrl" -ForegroundColor Green
Write-Host ""
Write-Host "If something looks wrong, you can inspect logs with:" -ForegroundColor Yellow
$cpaLogCmd = 'wsl -d ' + $distro + ' -- bash -lc "cd ' + $cpaDir + '; docker logs --tail 100 cli-proxy-api"'
$gatewayLogCmd = 'wsl -d ' + $distro + ' -- bash -lc ''XDG_RUNTIME_DIR=/run/user/$(id -u) journalctl --user -u openclaw-gateway.service -n 100 --no-pager'''
Write-Host $cpaLogCmd -ForegroundColor DarkGray
Write-Host $gatewayLogCmd -ForegroundColor DarkGray
Write-Host ""
Read-Host "Press Enter to exit"
