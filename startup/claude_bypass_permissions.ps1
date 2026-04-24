param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ClaudeArgs
)

$ErrorActionPreference = "Stop"

function Set-Utf8Console {
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = $utf8NoBom
    [Console]::OutputEncoding = $utf8NoBom
    $global:OutputEncoding = $utf8NoBom

    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:LANG = "C.UTF-8"
    $env:LC_ALL = "C.UTF-8"

    & chcp 65001 | Out-Null
}

Set-Utf8Console

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$claudeTempRoot = Join-Path $env:LOCALAPPDATA "Temp\claude"
$launcherConfigRoot = "D:\ClaudeCode\profiles\driver-model-bypass-official"
$launcherSettingsPath = Join-Path $launcherConfigRoot "settings.json"

function Resolve-ClaudeLauncher {
    $preferredLaunchers = @(
        "D:\ClaudeCode\global\claude.cmd",
        "D:\ClaudeCode\global\claude.ps1",
        "D:\Apps\nodejs\claude.cmd",
        "D:\Apps\nodejs\claude.ps1"
    )

    foreach ($candidate in $preferredLaunchers) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $resolved = Get-Command claude -ErrorAction Stop
    return $resolved.Source
}

Set-Location $projectRoot

if (-not (Test-Path $claudeTempRoot)) {
    New-Item -ItemType Directory -Path $claudeTempRoot -Force | Out-Null
}

if (-not (Test-Path $launcherConfigRoot)) {
    New-Item -ItemType Directory -Path $launcherConfigRoot -Force | Out-Null
}

if (-not (Test-Path $launcherSettingsPath)) {
    throw "Claude launcher profile settings missing: $launcherSettingsPath"
}

$env:CLAUDE_CONFIG_DIR = $launcherConfigRoot
Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:ANTHROPIC_BASE_URL -ErrorAction SilentlyContinue

$claudeLauncher = Resolve-ClaudeLauncher
$claudeVersion = & $claudeLauncher --version
$baseArgs = @(
    "--permission-mode", "bypassPermissions",
    "--add-dir", $claudeTempRoot
)

$authStatusRaw = ""
$authStatus = $null

try {
    $authStatusRaw = (& $claudeLauncher auth status 2>$null | Out-String).Trim()
    if ($authStatusRaw) {
        $authStatus = $authStatusRaw | ConvertFrom-Json
    }
} catch {
    $authStatus = $null
}

Write-Host "UTF-8 console ready." -ForegroundColor Green
Write-Host "Project root: $projectRoot" -ForegroundColor DarkGray
Write-Host "Claude temp allow dir: $claudeTempRoot" -ForegroundColor DarkGray
Write-Host "Claude config dir: $launcherConfigRoot" -ForegroundColor DarkGray
Write-Host "Launcher: $claudeLauncher" -ForegroundColor DarkGray
Write-Host "Version: $claudeVersion" -ForegroundColor DarkGray
Write-Host "Mode: bypassPermissions" -ForegroundColor Yellow
Write-Host "Profile: isolated official profile (does not inherit D:\ClaudeCode\home proxy settings)" -ForegroundColor Yellow

if ($authStatus -and -not $authStatus.loggedIn) {
    Write-Host "This isolated profile is not logged in yet. After Claude opens, run /login once." -ForegroundColor Cyan
} elseif (-not $authStatus) {
    Write-Host "Auth status could not be parsed. If Claude says not logged in, run /login once in this window." -ForegroundColor Cyan
}

& $claudeLauncher @baseArgs @ClaudeArgs
