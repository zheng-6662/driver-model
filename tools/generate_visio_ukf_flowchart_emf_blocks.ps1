# Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$BaseName = "ukf_flowchart_high_fidelity",
    [switch]$ShowVisio
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function From-CodePoints {
    param([int[]]$Codes)
    return (-join ($Codes | ForEach-Object { [char]$_ }))
}

function Set-CellFormula {
    param($Shape, [string]$CellName, [string]$Formula)
    $Shape.CellsU($CellName).FormulaU = $Formula
}

function Save-Bytes {
    param([string]$Path, $Bytes)
    [System.IO.File]::WriteAllBytes($Path, [byte[]]$Bytes)
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

function Add-TextParagraph {
    param($WordDocument, [string]$Text, [double]$FontSize = 11, [int]$Alignment = 1, [double]$SpaceAfter = 2, [bool]$Bold = $false)

    $range = $WordDocument.Range($WordDocument.Content.End - 1, $WordDocument.Content.End - 1)
    $range.InsertAfter($Text) | Out-Null
    $range = $WordDocument.Range($range.Start, $WordDocument.Content.End - 1)
    $range.ParagraphFormat.Alignment = $Alignment
    $range.ParagraphFormat.SpaceAfter = $SpaceAfter
    $range.ParagraphFormat.SpaceBefore = 0
    $range.Font.NameFarEast = "SimSun"
    $range.Font.Name = "Times New Roman"
    $range.Font.Size = $FontSize
    $range.Font.Bold = [int]$Bold
    $range.InsertParagraphAfter() | Out-Null
}

function Add-EquationParagraph {
    param($WordDocument, [string]$LinearText, [double]$FontSize = 11, [double]$SpaceAfter = 2)

    $range = $WordDocument.Range($WordDocument.Content.End - 1, $WordDocument.Content.End - 1)
    $range.InsertAfter($LinearText) | Out-Null
    $range = $WordDocument.Range($range.Start, $WordDocument.Content.End - 1)
    $range.ParagraphFormat.Alignment = 1
    $range.ParagraphFormat.SpaceAfter = $SpaceAfter
    $range.ParagraphFormat.SpaceBefore = 0
    $range.Font.Name = "Cambria Math"
    $range.Font.Size = $FontSize
    $null = $WordDocument.OMaths.Add($range)
    $WordDocument.OMaths.Item($WordDocument.OMaths.Count).BuildUp() | Out-Null
    $range.InsertParagraphAfter() | Out-Null
}

function Trim-WordEndParagraph {
    param($WordDocument)
    if ($WordDocument.Content.End -gt 1) {
        $endRange = $WordDocument.Range($WordDocument.Content.End - 1, $WordDocument.Content.End)
        if ($endRange.Text -eq "`r") {
            $endRange.Delete() | Out-Null
        }
    }
}

function New-WordBlockAssets {
    param(
        $WordApp,
        [string]$DocxPath,
        [string]$EmfPath,
        [double]$CanvasWidthInches,
        [double]$CanvasHeightInches,
        [scriptblock]$PopulateScript
    )

    $doc = $WordApp.Documents.Add()
    try {
        $doc.PageSetup.TopMargin = 6
        $doc.PageSetup.BottomMargin = 6
        $doc.PageSetup.LeftMargin = 8
        $doc.PageSetup.RightMargin = 8
        $doc.PageSetup.HeaderDistance = 0
        $doc.PageSetup.FooterDistance = 0
        $doc.PageSetup.PageWidth = [int][Math]::Round($CanvasWidthInches * 72)
        $doc.PageSetup.PageHeight = [int][Math]::Round($CanvasHeightInches * 72)
        & $PopulateScript $doc
        Trim-WordEndParagraph -WordDocument $doc
        $doc.SaveAs2($DocxPath)
        $doc.Range(0, $doc.Content.End - 1).Select()
        $WordApp.Selection.CopyAsPicture()
        Start-Sleep -Milliseconds 150
        $bytes = $doc.Range(0, $doc.Content.End - 1).EnhMetaFileBits
        Save-Bytes -Path $EmfPath -Bytes $bytes
    }
    finally {
        $doc.Close(0)
    }
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $repoRoot = Split-Path -Parent $scriptDir
    $OutputDir = Join-Path $repoRoot "output\visio"
}

$sourceDir = Join-Path $OutputDir ($BaseName + "_word_sources")
$emfDir = Join-Path $OutputDir ($BaseName + "_emf")
$vsdxPath = Join-Path $OutputDir ($BaseName + ".vsdx")
$pngPath = Join-Path $OutputDir ($BaseName + ".png")

$null = New-Item -ItemType Directory -Force -Path $OutputDir
$null = New-Item -ItemType Directory -Force -Path $sourceDir
$null = New-Item -ItemType Directory -Force -Path $emfDir

$hat = [string][char]0x0302
$tilde = [string][char]0x0303
$bar = [string][char]0x0304
$chi = [string][char]0x03C7
$delta = [string][char]0x03B4
$sum = [string][char]0x2211
$sqrt = [string][char]0x221A
$ellipsis = [string][char]0x2026
$yhat = "y$hat"
$xhat = "x$hat"
$xtilde = "x$tilde"
$xbar = "x$bar"

$titleInit = From-CodePoints @(0x521D,0x59CB,0x5316)
$titleSigma = (From-CodePoints @(0x751F,0x6210)) + "sigma" + (From-CodePoints @(0x70B9))
$titleMeasurement = From-CodePoints @(0x91CF,0x6D4B,0x91CF)
$titleInput = From-CodePoints @(0x8F93,0x5165,0x91CF)
$titleEstimate = From-CodePoints @(0x4F30,0x8BA1,0x91CF)
$titlePrior = From-CodePoints @(0x5148,0x9A8C,0x72B6,0x6001,0x4F30,0x8BA1)
$titleWhere = From-CodePoints @(0x5176,0x4E2D)
$titlePriorCov = From-CodePoints @(0x5148,0x9A8C,0x4F30,0x8BA1,0x8BEF,0x5DEE,0x7684,0x534F,0x65B9,0x5DEE)
$titlePredict = From-CodePoints @(0x91CF,0x6D4B,0x9884,0x6D4B)
$titlePredictCov = From-CodePoints @(0x91CF,0x6D4B,0x9884,0x6D4B,0x7684,0x534F,0x65B9,0x5DEE)
$titleUpdate = From-CodePoints @(0x72B6,0x6001,0x66F4,0x65B0)

$blocks = @(
    @{
        Key = "init"; X1 = 0.45; Y1 = 7.55; X2 = 5.45; Y2 = 9.65; InnerW = 4.25; InnerH = 1.50
        Build = {
            param($doc)
            Add-TextParagraph -WordDocument $doc -Text $titleInit -FontSize 12 -SpaceAfter 5
            Add-EquationParagraph -WordDocument $doc -LinearText ("x_0 = [0, " + $ellipsis + ", 0]^T") -FontSize 11
            Add-EquationParagraph -WordDocument $doc -LinearText ($xhat + "_0^+ = E(" + $xtilde + "_0)") -FontSize 11
            Add-EquationParagraph -WordDocument $doc -LinearText ("P_0^+ = E[(" + $xtilde + "_0 - " + $xbar + "_0)(" + $xtilde + "_0 - " + $xbar + "_0)^T]") -FontSize 11 -SpaceAfter 0
        }
    },
    @{
        Key = "sigma"; X1 = 6.80; Y1 = 7.60; X2 = 13.60; Y2 = 9.65; InnerW = 5.90; InnerH = 1.55
        Build = {
            param($doc)
            Add-TextParagraph -WordDocument $doc -Text $titleSigma -FontSize 12 -SpaceAfter 4
            Add-EquationParagraph -WordDocument $doc -LinearText ($chi + $tilde + "_(k-1)^(i) = " + $xhat + "_(k-1)^+ + " + $chi + $tilde + "^(i)    i = 1, " + $ellipsis + ", 2n") -FontSize 10.5
            Add-TextParagraph -WordDocument $doc -Text $titleWhere -FontSize 10.5 -SpaceAfter 1
            Add-EquationParagraph -WordDocument $doc -LinearText ($chi + $tilde + "^(i) = (" + $sqrt + "(nP_(k-1)^+))_i^T,    " + $chi + $tilde + "^(n+i) = -(" + $sqrt + "(nP_(k-1)^+))_i^T") -FontSize 10.2 -SpaceAfter 0
        }
    },
    @{
        Key = "measurement"; X1 = 1.55; Y1 = 5.15; X2 = 4.85; Y2 = 7.45; InnerW = 2.45; InnerH = 1.75
        Build = {
            param($doc)
            Add-TextParagraph -WordDocument $doc -Text $titleMeasurement -FontSize 12 -SpaceAfter 5
            Add-EquationParagraph -WordDocument $doc -LinearText "y(t) = [a_y, r]^T" -FontSize 11 -SpaceAfter 5
            Add-TextParagraph -WordDocument $doc -Text $titleInput -FontSize 11 -SpaceAfter 1
            Add-EquationParagraph -WordDocument $doc -LinearText ("u_k = " + $delta + "_f(k)") -FontSize 11 -SpaceAfter 0
        }
    },
    @{
        Key = "estimate"; X1 = 1.55; Y1 = 3.65; X2 = 4.85; Y2 = 4.95; InnerW = 2.45; InnerH = 0.78
        Build = {
            param($doc)
            Add-TextParagraph -WordDocument $doc -Text $titleEstimate -FontSize 12 -SpaceAfter 3
            Add-EquationParagraph -WordDocument $doc -LinearText ("x(t) = [a_1, a_2, " + $ellipsis + ", a_8]^T") -FontSize 10.7 -SpaceAfter 0
        }
    },
    @{
        Key = "prior"; X1 = 6.80; Y1 = 4.35; X2 = 13.60; Y2 = 6.95; InnerW = 5.80; InnerH = 2.00
        Build = {
            param($doc)
            Add-TextParagraph -WordDocument $doc -Text $titlePrior -FontSize 12 -SpaceAfter 4
            Add-EquationParagraph -WordDocument $doc -LinearText ($xhat + "_k^- = 1/(2n) " + $sum + "_(i=1)^(2n) " + $chi + "_k^(i)") -FontSize 10.6
            Add-TextParagraph -WordDocument $doc -Text ($titleWhere + "  " + $chi + "_k^(i) = f(" + $chi + "_(k-1)^(i), u_k, t_k)") -FontSize 10.3 -SpaceAfter 4
            Add-TextParagraph -WordDocument $doc -Text $titlePriorCov -FontSize 10.6 -SpaceAfter 2
            Add-EquationParagraph -WordDocument $doc -LinearText ("P_k^- = 1/(2n) " + $sum + "_(i=1)^(2n) (" + $chi + "_k^(i) - " + $xhat + "_k^-)(" + $chi + "_k^(i) - " + $xhat + "_k^-)^T + Q_(k-1)") -FontSize 9.8 -SpaceAfter 0
        }
    },
    @{
        Key = "predict"; X1 = 6.85; Y1 = 0.45; X2 = 13.60; Y2 = 3.20; InnerW = 5.85; InnerH = 2.05
        Build = {
            param($doc)
            Add-TextParagraph -WordDocument $doc -Text $titlePredict -FontSize 12 -SpaceAfter 4
            Add-EquationParagraph -WordDocument $doc -LinearText ($yhat + "_k^(i) = h(" + $chi + "_k^(i), t_k)    " + $yhat + "_k = 1/(2n) " + $sum + "_(i=1)^(2n) " + $yhat + "_k^(i)") -FontSize 10.0 -SpaceAfter 4
            Add-TextParagraph -WordDocument $doc -Text $titlePredictCov -FontSize 10.6 -SpaceAfter 2
            Add-EquationParagraph -WordDocument $doc -LinearText ("P_y = 1/(2n) " + $sum + "_(i=1)^(2n) (" + $yhat + "_k^(i) - " + $yhat + "_k)(" + $yhat + "_k^(i) - " + $yhat + "_k)^T + R_k") -FontSize 9.8 -SpaceAfter 0
        }
    },
    @{
        Key = "update"; X1 = 0.20; Y1 = 0.45; X2 = 5.80; Y2 = 3.15; InnerW = 4.75; InnerH = 2.05
        Build = {
            param($doc)
            Add-TextParagraph -WordDocument $doc -Text $titleUpdate -FontSize 12 -SpaceAfter 4
            Add-EquationParagraph -WordDocument $doc -LinearText ("P_xy = 1/(2n) " + $sum + "_(i=1)^(2n) (" + $chi + "_k^(i) - " + $xhat + "_k^-)(" + $yhat + "_k^(i) - " + $yhat + "_k)^T") -FontSize 9.7
            Add-EquationParagraph -WordDocument $doc -LinearText "K_k = P_xy P_y^(-1)" -FontSize 10.3
            Add-EquationParagraph -WordDocument $doc -LinearText ($xhat + "_k^+ = " + $xhat + "_k^- + K_k(y_k - " + $yhat + "_k)") -FontSize 10.3
            Add-EquationParagraph -WordDocument $doc -LinearText "P_k^+ = P_k^- - K_k P_y K_k^T" -FontSize 10.3 -SpaceAfter 0
        }
    }
)

$word = $null
$visio = $null
$vdoc = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    foreach ($block in $blocks) {
        $docxPath = Join-Path $sourceDir ($block.Key + ".docx")
        $emfPath = Join-Path $emfDir ($block.Key + ".emf")
        New-WordBlockAssets -WordApp $word -DocxPath $docxPath -EmfPath $emfPath -CanvasWidthInches $block.InnerW -CanvasHeightInches $block.InnerH -PopulateScript $block.Build
    }

    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $ShowVisio.IsPresent
    $visio.AlertResponse = 7

    $vdoc = $visio.Documents.Add("")
    $page = $visio.ActivePage
    $page.Name = "UKF High Fidelity"
    $page.PageSheet.CellsU("PageWidth").FormulaU = "14 in"
    $page.PageSheet.CellsU("PageHeight").FormulaU = "10.2 in"

    foreach ($block in $blocks) {
        $null = Add-RoundedBlock -Page $page -X1 $block.X1 -Y1 $block.Y1 -X2 $block.X2 -Y2 $block.Y2
        $emfPath = Join-Path $emfDir ($block.Key + ".emf")
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

    $vdoc.SaveAs($vsdxPath)
    $page.Export($pngPath)

    Write-Output ("OUTPUT_VSDX=" + $vsdxPath)
    Write-Output ("OUTPUT_PNG=" + $pngPath)
    Write-Output ("WORD_SOURCES=" + $sourceDir)
}
finally {
    if ($vdoc) {
        try { $vdoc.Close() } catch {}
    }
    if ($visio) {
        try { $visio.Quit() } catch {}
    }
    if ($word) {
        try { $word.Quit() } catch {}
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
