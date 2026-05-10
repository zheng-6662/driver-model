param(
    [switch]$ContinueOnError,
    [switch]$DryRun,
    [switch]$AllowCpu,
    [string]$CondaEnv = "predict_2"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$condaExe = "D:\ProgramData\anaconda3\Scripts\conda.exe"
if (-not (Test-Path -LiteralPath $condaExe)) {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -eq $condaCmd) {
        throw "conda was not found. Expected: $condaExe"
    }
    $condaExe = $condaCmd.Source
}
$runnerArgs = @("run", "-n", $CondaEnv, "--no-capture-output", "python", "-u")
$logDir = Join-Path $here "run_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$prefixes = @("07", "08", "09", "10", "11", "12")
$scripts = @()
foreach ($prefix in $prefixes) {
    $matches = @(Get-ChildItem -LiteralPath $here -Filter ("{0}_*.py" -f $prefix) | Sort-Object Name)
    if ($matches.Count -ne 1) {
        throw ("Expected exactly one script for prefix {0}, found {1}" -f $prefix, $matches.Count)
    }
    $scripts += $matches[0].Name
}

Write-Host "Work dir: $here"
Write-Host "Run range: 07-12"
Write-Host "Log dir: $logDir"
Write-Host "Conda env: $CondaEnv"
Write-Host "Runner: $condaExe run -n $CondaEnv --no-capture-output python -u"

$cudaCheck = & $condaExe @runnerArgs -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
$pythonExe = $cudaCheck[0]
$torchVersion = $cudaCheck[1]
$torchCudaBuild = $cudaCheck[2]
$cudaAvailable = $cudaCheck[3]
$cudaDeviceName = $cudaCheck[4]
Write-Host "Python exe: $pythonExe"
Write-Host "PyTorch: $torchVersion, cuda_build=$torchCudaBuild, cuda_available=$cudaAvailable, device=$cudaDeviceName"
if ($cudaAvailable -ne "True" -and -not $AllowCpu) {
    Write-Host "Current conda env is CPU-only. Stop to avoid a very slow CPU run." -ForegroundColor Red
    Write-Host "If you really want CPU, run: .\run_07_to_12.ps1 -AllowCpu" -ForegroundColor Yellow
    exit 2
}
Write-Host ""

if ($DryRun) {
    Write-Host "DryRun: files checked. Execution order:"
    foreach ($scriptName in $scripts) {
        Write-Host " - $scriptName"
    }
    exit 0
}

$total = $scripts.Count
$failed = @()

for ($i = 0; $i -lt $total; $i++) {
    $scriptName = $scripts[$i]
    $scriptPath = Join-Path $here $scriptName
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $baseName = [IO.Path]::GetFileNameWithoutExtension($scriptName)
    $logPath = Join-Path $logDir ("{0}_{1}.log" -f $baseName, $stamp)

    Write-Host "============================================================"
    Write-Host ("[{0}/{1}] START: {2}" -f ($i + 1), $total, $scriptName)
    Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Log file: $logPath"
    Write-Host "============================================================"

    & $condaExe @runnerArgs $scriptPath 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        $failed += [PSCustomObject]@{
            Script = $scriptName
            ExitCode = $exitCode
            Log = $logPath
        }
        Write-Host ""
        Write-Host ("FAILED: {0}, exit code={1}" -f $scriptName, $exitCode) -ForegroundColor Red
        Write-Host "Failure log: $logPath" -ForegroundColor Red
        if (-not $ContinueOnError) {
            Write-Host "Stopped. To continue after failures, run: .\run_07_to_12.ps1 -ContinueOnError" -ForegroundColor Yellow
            exit $exitCode
        }
    } else {
        Write-Host ""
        Write-Host ("[{0}/{1}] DONE: {2}" -f ($i + 1), $total, $scriptName) -ForegroundColor Green
        Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    }

    Write-Host ""
}

Write-Host "============================================================"
if ($failed.Count -eq 0) {
    Write-Host "07-12 finished successfully." -ForegroundColor Green
    exit 0
}

Write-Host "Some scripts failed:" -ForegroundColor Red
$failed | Format-Table -AutoSize
exit 1
