$ErrorActionPreference = "Stop"

function Show-LaunchError {
    param(
        [string]$Message
    )

    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            $Message,
            "项目进度看板启动失败",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    } catch {
        Write-Error $Message
    }
}

try {
    $projectRoot = Split-Path -Parent $PSScriptRoot

    $builderCandidates = @(
        (Join-Path $projectRoot "02_code\tools\build_progress_dashboard.py"),
        (Join-Path $projectRoot "tools\build_progress_dashboard.py")
    )
    $dashboardCandidates = @(
        (Join-Path $projectRoot "04_project_logs\reports\project_progress_dashboard.html"),
        (Join-Path $projectRoot "reports\project_progress_dashboard.html")
    )

    $builder = $builderCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $builder) {
        throw "Dashboard builder not found. Checked:`n$($builderCandidates -join "`n")"
    }

    Write-Host "[progress-dashboard] rebuilding dashboard..."
    py -3 $builder

    if ($LASTEXITCODE -ne 0) {
        throw "Dashboard build failed with exit code $LASTEXITCODE"
    }

    $dashboard = $dashboardCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $dashboard) {
        throw "Dashboard file not found after rebuild. Checked:`n$($dashboardCandidates -join "`n")"
    }

    Write-Host "[progress-dashboard] opening $dashboard"
    Start-Process $dashboard
} catch {
    $message = if ($_.Exception) { $_.Exception.Message } else { $_.ToString() }
    Show-LaunchError $message
    exit 1
}
