@echo off
setlocal
powershell.exe -NoLogo -NoProfile -NoExit -ExecutionPolicy Bypass -File "%~dp0claude_bypass_permissions.ps1" %*
