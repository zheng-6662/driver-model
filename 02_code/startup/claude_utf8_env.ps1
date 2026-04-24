param(
    [string]$Command
)

$ErrorActionPreference = 'Stop'

function Set-Utf8Console {
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = $utf8NoBom
    [Console]::OutputEncoding = $utf8NoBom
    $global:OutputEncoding = $utf8NoBom

    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:LANG = 'C.UTF-8'
    $env:LC_ALL = 'C.UTF-8'

    & chcp 65001 | Out-Null
}

Set-Utf8Console

Write-Host "UTF-8 console ready." -ForegroundColor Green
Write-Host "Code page: $(chcp | ForEach-Object { ($_ -split ':')[-1].Trim() })" -ForegroundColor DarkGray
Write-Host "PYTHONIOENCODING=$env:PYTHONIOENCODING" -ForegroundColor DarkGray

if ($Command) {
    Write-Host "Running: $Command" -ForegroundColor Cyan
    Invoke-Expression $Command
}
