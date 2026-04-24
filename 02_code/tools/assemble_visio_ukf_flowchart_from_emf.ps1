# Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$AssetDir = "",
    [string]$OutputDir = "",
    [string]$BaseName = "ukf_flowchart_high_fidelity",
    [switch]$ShowVisio
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Set-CellFormula {
    param($Shape, [string]$CellName, [string]$Formula)
    $Shape.CellsU($CellName).FormulaU = $Formula
}

function Add-RoundedBlock {
    param($Page, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2)
    $shape = $Page.DrawRectangle($X1, $Y1, $X2, $Y2)
    Set-CellFormula -Shape $shape -CellName "Rounding" -Formula "0.18 in"
    Set-CellFormula -Shape $shape -CellName "LineColor" -Formula "RGB(0,0,0)"
    Set-CellFormula -Shape $shape -CellName "LineWeight" -Formula "1.4 pt"
    Set-CellFormula -Shape $shape -CellName "FillForegnd" -Formula "RGB(255,255,255)"
    Set-CellFormula -Shape $shape -CellName "FillPattern" -Formula "1"
    return $shape
}

function Add-LineSegment {
    param($Page, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2, [switch]$ArrowAtEnd)
    $line = $Page.DrawLine($X1, $Y1, $X2, $Y2)
    Set-CellFormula -Shape $line -CellName "LineColor" -Formula "RGB(0,0,0)"
    Set-CellFormula -Shape $line -CellName "LineWeight" -Formula "1.8 pt"
    Set-CellFormula -Shape $line -CellName "BeginArrow" -Formula "0"
    if ($ArrowAtEnd) {
        Set-CellFormula -Shape $line -CellName "EndArrow" -Formula "13"
        Set-CellFormula -Shape $line -CellName "EndArrowSize" -Formula "2"
    } else {
        Set-CellFormula -Shape $line -CellName "EndArrow" -Formula "0"
    }
    return $line
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $repoRoot = Split-Path -Parent $scriptDir
    $OutputDir = Join-Path $repoRoot "output\visio"
}
if ([string]::IsNullOrWhiteSpace($AssetDir)) {
    $AssetDir = Join-Path $OutputDir ($BaseName + "_emf")
}

$vsdxPath = Join-Path $OutputDir ($BaseName + ".vsdx")
$pngPath = Join-Path $OutputDir ($BaseName + ".png")

$blocks = @(
    @{ Key = "init"; X1 = 0.45; Y1 = 7.55; X2 = 5.45; Y2 = 9.65; InnerW = 4.25; InnerH = 1.50 },
    @{ Key = "sigma"; X1 = 6.80; Y1 = 7.60; X2 = 13.60; Y2 = 9.65; InnerW = 5.90; InnerH = 1.55 },
    @{ Key = "measurement"; X1 = 1.55; Y1 = 5.15; X2 = 4.85; Y2 = 7.45; InnerW = 2.45; InnerH = 1.75 },
    @{ Key = "estimate"; X1 = 1.55; Y1 = 3.65; X2 = 4.85; Y2 = 4.95; InnerW = 2.45; InnerH = 0.78 },
    @{ Key = "prior"; X1 = 6.80; Y1 = 4.35; X2 = 13.60; Y2 = 6.95; InnerW = 5.80; InnerH = 2.00 },
    @{ Key = "predict"; X1 = 6.85; Y1 = 0.45; X2 = 13.60; Y2 = 3.20; InnerW = 5.85; InnerH = 2.05 },
    @{ Key = "update"; X1 = 0.20; Y1 = 0.45; X2 = 5.80; Y2 = 3.15; InnerW = 4.75; InnerH = 2.05 }
)

$visio = $null
$doc = $null

try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $ShowVisio.IsPresent
    $visio.AlertResponse = 7
    $doc = $visio.Documents.Add("")
    $page = $visio.ActivePage
    $page.Name = "UKF High Fidelity"
    $page.PageSheet.CellsU("PageWidth").FormulaU = "14 in"
    $page.PageSheet.CellsU("PageHeight").FormulaU = "10.2 in"

    foreach ($block in $blocks) {
        $null = Add-RoundedBlock -Page $page -X1 $block.X1 -Y1 $block.Y1 -X2 $block.X2 -Y2 $block.Y2
        $emfPath = Join-Path $AssetDir ($block.Key + ".emf")
        $shape = $page.Import($emfPath)
        $shape.CellsU("PinX").FormulaU = ("{0} in" -f (($block.X1 + $block.X2) / 2.0))
        $shape.CellsU("PinY").FormulaU = ("{0} in" -f (($block.Y1 + $block.Y2) / 2.0))
        $shape.CellsU("Width").FormulaU = ("{0} in" -f $block.InnerW)
        $shape.CellsU("Height").FormulaU = ("{0} in" -f $block.InnerH)
        Set-CellFormula -Shape $shape -CellName "LinePattern" -Formula "0"
    }

    Add-LineSegment -Page $page -X1 5.45 -Y1 8.60 -X2 6.80 -Y2 8.60 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -X1 10.20 -Y1 7.60 -X2 10.20 -Y2 6.95 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -X1 4.85 -Y1 6.30 -X2 5.90 -Y2 6.30 | Out-Null
    Add-LineSegment -Page $page -X1 5.90 -Y1 6.30 -X2 5.90 -Y2 5.65 | Out-Null
    Add-LineSegment -Page $page -X1 5.90 -Y1 5.65 -X2 6.80 -Y2 5.65 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -X1 10.20 -Y1 4.35 -X2 10.20 -Y2 3.20 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -X1 6.85 -Y1 1.83 -X2 5.80 -Y2 1.83 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -X1 3.20 -Y1 3.15 -X2 3.20 -Y2 3.65 -ArrowAtEnd | Out-Null

    $doc.SaveAs($vsdxPath)
    $page.Export($pngPath)
    Write-Output ("OUTPUT_VSDX=" + $vsdxPath)
    Write-Output ("OUTPUT_PNG=" + $pngPath)
}
finally {
    if ($doc) {
        try { $doc.Close() } catch {}
    }
    if ($visio) {
        try { $visio.Quit() } catch {}
    }
}
