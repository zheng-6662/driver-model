@echo off
setlocal
powershell.exe -NoLogo -NoProfile -NoExit -ExecutionPolicy Bypass -File "%~dp0claude_via_codex_api.ps1" %*
