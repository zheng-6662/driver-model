param(
    [switch]$SkipPrompt,
    [switch]$SkipCPA,
    [switch]$NoPause,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"

function Show-LaunchError {
    param([string]$Message)

    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            $Message,
            "Claude Driver Model Project startup failed",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    } catch {
        Write-Host $Message -ForegroundColor Red
    }
}

function Wait-ForUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [hashtable]$Headers = @{},
        [int]$TimeoutSeconds = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Url -Headers $Headers -UseBasicParsing -TimeoutSec 5 | Out-Null
            return $true
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    return $false
}

try {
    $projectRoot = Split-Path -Parent $PSScriptRoot
    $cpaScript = Join-Path $PSScriptRoot "openclaw_cpa_oneclick.ps1"
    $projectClaudeScript = Join-Path $PSScriptRoot "claude_via_codex_api.ps1"

    if (-not $SkipCPA) {
        if (-not (Test-Path $cpaScript)) {
            throw "CPA bootstrap script not found: $cpaScript"
        }

        & $cpaScript -SkipPrompt:$true -SkipBrowser:$true -NoPause:$true
        if ($LASTEXITCODE -ne 0) {
            throw "CPA bootstrap exited with code $LASTEXITCODE"
        }
    }

    $cpaReady = Wait-ForUrl -Url "http://127.0.0.1:8317/v1/models" -Headers @{ Authorization = "Bearer sk-dummy" } -TimeoutSeconds 20
    if (-not $cpaReady) {
        throw "CPA API is not reachable at http://127.0.0.1:8317/v1/models"
    }

    if (-not (Test-Path $projectClaudeScript)) {
        throw "Project Claude launcher not found: $projectClaudeScript"
    }

    if ($SmokeTest) {
        Write-Host "Project root: $projectRoot" -ForegroundColor DarkGray
        Write-Host "CPA API: ready" -ForegroundColor Green
        Write-Host "Project Claude launcher: $projectClaudeScript" -ForegroundColor Green
        return
    }

    Set-Location $projectRoot
    & $projectClaudeScript
    if ($LASTEXITCODE -ne 0) {
        throw "claude_via_codex_api.ps1 exited with code $LASTEXITCODE"
    }
} catch {
    $message = if ($_.Exception) { $_.Exception.Message } else { $_.ToString() }
    Show-LaunchError $message
    if (-not $NoPause) {
        Write-Host $message -ForegroundColor Red
    }
    exit 1
}

if (-not $NoPause) {
    Read-Host "Press Enter to exit"
}
