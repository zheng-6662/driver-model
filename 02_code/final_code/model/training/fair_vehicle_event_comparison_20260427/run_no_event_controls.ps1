$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

py -3.11 (Get-ChildItem -LiteralPath $here -Filter "01_*.py").FullName
py -3.11 (Get-ChildItem -LiteralPath $here -Filter "07_*.py").FullName
py -3.11 (Get-ChildItem -LiteralPath $here -Filter "08_*.py").FullName
py -3.11 (Get-ChildItem -LiteralPath $here -Filter "06_*.py").FullName
py -3.11 (Get-ChildItem -LiteralPath $here -Filter "09_*.py").FullName
py -3.11 (Get-ChildItem -LiteralPath $here -Filter "10_*.py").FullName
py -3.11 (Get-ChildItem -LiteralPath $here -Filter "11_*.py").FullName
py -3.11 (Get-ChildItem -LiteralPath $here -Filter "12_*.py").FullName
