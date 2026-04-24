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

Set-Utf8Console

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$claudeTempRoot = Join-Path $env:LOCALAPPDATA "Temp\claude"
$profileRoot = "D:\ClaudeCode\profiles\driver-model-bypass-official"
$proxyApiKey = "sk-claude-codex-proxy"

if (-not (Test-Path $claudeTempRoot)) {
    New-Item -ItemType Directory -Path $claudeTempRoot -Force | Out-Null
}

if (-not (Test-Path $profileRoot)) {
    throw "Claude profile directory missing: $profileRoot"
}

$serviceInfo = & (Join-Path $PSScriptRoot "start_claude_codex_adapter.ps1") `
    -ListenHost "127.0.0.1" `
    -PreferredPort 8417 `
    -MaxPort 8427 `
    -BackendBaseUrl "http://localhost:8317/v1" `
    -BackendApiKey "sk-dummy" `
    -BackendModel "gpt-5.4" `
    -ProxyApiKey $proxyApiKey

$adapterBaseUrl = [string]$serviceInfo.baseUrl

$env:CLAUDE_CONFIG_DIR = $profileRoot
$env:ANTHROPIC_BASE_URL = $adapterBaseUrl
Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
$env:ANTHROPIC_API_KEY = $proxyApiKey

$claudeLauncher = Resolve-ClaudeLauncher
$claudeVersion = & $claudeLauncher --version
$baseArgs = @(
    "--permission-mode", "bypassPermissions",
    "--add-dir", $claudeTempRoot
)

Set-Location $projectRoot

Write-Host "UTF-8 console ready." -ForegroundColor Green
Write-Host "Project root: $projectRoot" -ForegroundColor DarkGray
Write-Host "Claude temp allow dir: $claudeTempRoot" -ForegroundColor DarkGray
Write-Host "Claude config dir: $profileRoot" -ForegroundColor DarkGray
Write-Host "Anthropic base URL: $($env:ANTHROPIC_BASE_URL)" -ForegroundColor DarkGray
Write-Host "Adapter service port: $($serviceInfo.port)" -ForegroundColor DarkGray
Write-Host "Launcher: $claudeLauncher" -ForegroundColor DarkGray
Write-Host "Version: $claudeVersion" -ForegroundColor DarkGray
Write-Host "Mode: bypassPermissions + local CPA adapter" -ForegroundColor Yellow

& $claudeLauncher @baseArgs @ClaudeArgs
