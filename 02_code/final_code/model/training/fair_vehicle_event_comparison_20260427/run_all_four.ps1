$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

py -3.11 (Get-ChildItem -LiteralPath $here -Filter "01_*.py").FullName
py -3.11 (Get-ChildItem -LiteralPath $here -Filter "02_*.py").FullName
py -3.11 (Get-ChildItem -LiteralPath $here -Filter "03_*.py").FullName
py -3.11 (Get-ChildItem -LiteralPath $here -Filter "04_*.py").FullName
