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

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$runtimeDir = Join-Path $projectRoot "03_results\tmp\claude_codex_adapter"
$stateFile = Join-Path $runtimeDir "service_state.json"
$stopFlag = Join-Path $runtimeDir "stop.flag"
$serviceLog = Join-Path $runtimeDir "supervisor.log"
$pythonExeCandidates = @(
    "F:\python3.11\pythonw.exe",
    "F:\python3.11\python.exe"
)
$pythonExe = $pythonExeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$adapterScript = Join-Path $projectRoot "02_code\tools\anthropic_codex_adapter.py"
$script:RestartCount = 0
$script:AdapterPid = $null

if (-not $pythonExe) {
    throw "Python runtime not found. Checked: $($pythonExeCandidates -join ', ')"
}

if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
}

function Write-ServiceLog {
    param([string]$Message)

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $serviceLog -Value "[$ts] $Message"
}

function Get-ListenConnection {
    param([int]$Port)

    try {
        return Get-NetTCPConnection -LocalAddress $ListenHost -LocalPort $Port -State Listen -ErrorAction Stop |
            Select-Object -First 1
    } catch {
        return $null
    }
}

function Test-AdapterHealth {
    param([int]$Port)

    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://${ListenHost}:$Port/health" -TimeoutSec 3
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Get-AvailablePort {
    for ($candidate = $PreferredPort; $candidate -le $MaxPort; $candidate++) {
        $conn = Get-ListenConnection -Port $candidate
        if (-not $conn) {
            return $candidate
        }

        if ($script:AdapterPid -and $conn.OwningProcess -eq $script:AdapterPid) {
            return $candidate
        }
    }

    throw "No free adapter port found in range ${PreferredPort}-${MaxPort}."
}

function Write-State {
    param(
        [string]$Status,
        [int]$Port = 0,
        [int]$AdapterPid = 0
    )

    $state = [pscustomobject]@{
        servicePid     = $PID
        adapterPid     = $AdapterPid
        listenHost     = $ListenHost
        port           = $Port
        baseUrl        = if ($Port -gt 0) { "http://${ListenHost}:$Port" } else { $null }
        backendBaseUrl = $BackendBaseUrl
        backendModel   = $BackendModel
        status         = $Status
        restartCount   = $script:RestartCount
        updatedAt      = (Get-Date).ToString("o")
    }

    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $stateFile -Encoding UTF8
}

function Start-AdapterProcess {
    param([int]$Port)

    $stdoutLog = Join-Path $runtimeDir ("adapter_stdout_port{0}.log" -f $Port)
    $stderrLog = Join-Path $runtimeDir ("adapter_stderr_port{0}.log" -f $Port)

    $arguments = @(
        $adapterScript,
        "--host", $ListenHost,
        "--port", $Port,
        "--backend-base-url", $BackendBaseUrl,
        "--backend-api-key", $BackendApiKey,
        "--backend-model", $BackendModel,
        "--proxy-api-key", $ProxyApiKey
    )

    return Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $arguments `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru
}

if (Test-Path $stopFlag) {
    Remove-Item -LiteralPath $stopFlag -Force
}

Write-ServiceLog "Supervisor started. preferredPort=$PreferredPort maxPort=$MaxPort pythonExe=$pythonExe"
Write-State -Status "booting"

while ($true) {
    if (Test-Path $stopFlag) {
        Write-ServiceLog "Stop flag detected before (re)start. Exiting supervisor."
        break
    }

    try {
        $port = Get-AvailablePort
    } catch {
        Write-ServiceLog "Port allocation failed: $($_.Exception.Message)"
        Write-State -Status "port_unavailable"
        Start-Sleep -Seconds 3
        continue
    }

    Write-ServiceLog "Starting adapter on port $port"
    $adapter = Start-AdapterProcess -Port $port
    $script:AdapterPid = $adapter.Id
    Write-State -Status "starting" -Port $port -AdapterPid $adapter.Id

    $healthy = $false
    for ($attempt = 1; $attempt -le 20; $attempt++) {
        if (Test-Path $stopFlag) {
            break
        }

        Start-Sleep -Seconds 1
        if ($adapter.HasExited) {
            break
        }

        if (Test-AdapterHealth -Port $port) {
            $healthy = $true
            break
        }
    }

    if (-not $healthy) {
        Write-ServiceLog "Adapter failed health check on port $port. Restarting."
        Write-State -Status "unhealthy" -Port $port -AdapterPid $adapter.Id
        if (-not $adapter.HasExited) {
            Stop-Process -Id $adapter.Id -Force -ErrorAction SilentlyContinue
        }
        $script:RestartCount++
        Start-Sleep -Seconds 2
        continue
    }

    Write-ServiceLog "Adapter healthy on port $port (adapterPid=$($adapter.Id))"
    Write-State -Status "healthy" -Port $port -AdapterPid $adapter.Id

    try {
        Wait-Process -Id $adapter.Id
    } catch {
    }

    if (Test-Path $stopFlag) {
        Write-ServiceLog "Stop flag detected after adapter exit. Exiting supervisor."
        break
    }

    Write-ServiceLog "Adapter exited unexpectedly. Scheduling restart."
    $script:RestartCount++
    Start-Sleep -Seconds 2
}

Write-State -Status "stopped"
Write-ServiceLog "Supervisor stopped."
