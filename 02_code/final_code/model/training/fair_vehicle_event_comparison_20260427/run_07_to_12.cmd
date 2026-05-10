@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_07_to_12.ps1"
echo.
echo ??????????????
pause >nul
