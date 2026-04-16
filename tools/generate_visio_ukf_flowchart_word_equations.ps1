# Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$BaseName = "ukf_flowchart_word_equations"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function From-CodePoints {
    param(
        [Parameter(Mandatory = $true)]
        [int[]]$Codes
    )

    return (-join ($Codes | ForEach-Object { [char]$_ }))
}

function Set-CellFormula {
    param(
        [Parameter(Mandatory = $true)] $Shape,
        [Parameter(Mandatory = $true)] [string]$CellName,
        [Parameter(Mandatory = $true)] [string]$Formula
    )

    $Shape.CellsU($CellName).FormulaU = $Formula
}

function Add-RoundedBlock {
    param(
        [Parameter(Mandatory = $true)] $Page,
        [Parameter(Mandatory = $true)] [double]$X1,
        [Parameter(Mandatory = $true)] [double]$Y1,
        [Parameter(Mandatory = $true)] [double]$X2,
        [Parameter(Mandatory = $true)] [double]$Y2
    )

    $shape = $Page.DrawRectangle($X1, $Y1, $X2, $Y2)
    Set-CellFormula -Shape $shape -CellName "Rounding" -Formula "0.18 in"
    Set-CellFormula -Shape $shape -CellName "LineColor" -Formula "RGB(0,0,0)"
    Set-CellFormula -Shape $shape -CellName "LineWeight" -Formula "1.4 pt"
    Set-CellFormula -Shape $shape -CellName "FillForegnd" -Formula "RGB(255,255,255)"
    Set-CellFormula -Shape $shape -CellName "FillPattern" -Formula "1"
    return $shape
}

function Add-LineSegment {
    param(
        [Parameter(Mandatory = $true)] $Page,
        [Parameter(Mandatory = $true)] [double]$X1,
        [Parameter(Mandatory = $true)] [double]$Y1,
        [Parameter(Mandatory = $true)] [double]$X2,
        [Parameter(Mandatory = $true)] [double]$Y2,
        [switch]$ArrowAtEnd
    )

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

function New-EmptyWordDocumentShape {
    param(
        [Parameter(Mandatory = $true)] $Page,
        [Parameter(Mandatory = $true)] [double]$CenterX,
        [Parameter(Mandatory = $true)] [double]$CenterY,
        [Parameter(Mandatory = $true)] [double]$WidthInches,
        [Parameter(Mandatory = $true)] [double]$HeightInches
    )

    $shape = $Page.InsertObject("Word.Document.12", 0)
    Set-CellFormula -Shape $shape -CellName "PinX" -Formula ("{0} in" -f $CenterX)
    Set-CellFormula -Shape $shape -CellName "PinY" -Formula ("{0} in" -f $CenterY)
    Set-CellFormula -Shape $shape -CellName "Width" -Formula ("{0} in" -f $WidthInches)
    Set-CellFormula -Shape $shape -CellName "Height" -Formula ("{0} in" -f $HeightInches)
    return $shape
}

function Get-InsertRange {
    param(
        [Parameter(Mandatory = $true)] $WordDocument
    )

    if ($WordDocument.Content.Text.Trim().Length -eq 0) {
        return $WordDocument.Range(0, 0)
    }

    $range = $WordDocument.Range($WordDocument.Content.End - 1, $WordDocument.Content.End - 1)
    $null = $range.InsertParagraphAfter()
    return $WordDocument.Range($WordDocument.Content.End - 1, $WordDocument.Content.End - 1)
}

function Add-WordParagraph {
    param(
        [Parameter(Mandatory = $true)] $WordDocument,
        [Parameter(Mandatory = $true)] [string]$Text,
        [double]$FontSize = 11,
        [int]$Alignment = 1,
        [double]$SpaceAfter = 2,
        [switch]$Bold
    )

    $range = Get-InsertRange -WordDocument $WordDocument
    $range.Text = $Text
    $range.ParagraphFormat.Alignment = $Alignment
    $range.ParagraphFormat.SpaceAfter = $SpaceAfter
    $range.ParagraphFormat.SpaceBefore = 0
    $range.Font.NameFarEast = "SimSun"
    $range.Font.Name = "Times New Roman"
    $range.Font.Size = $FontSize
    $range.Font.Bold = [int]$Bold.IsPresent
    return $range
}

function Add-WordEquation {
    param(
        [Parameter(Mandatory = $true)] $WordDocument,
        [Parameter(Mandatory = $true)] [string]$LinearText,
        [double]$FontSize = 11,
        [double]$SpaceAfter = 2
    )

    $range = Get-InsertRange -WordDocument $WordDocument
    $range.Text = $LinearText
    $range.ParagraphFormat.Alignment = 1
    $range.ParagraphFormat.SpaceAfter = $SpaceAfter
    $range.ParagraphFormat.SpaceBefore = 0
    $range.Font.Size = $FontSize
    $range.Font.Name = "Cambria Math"
    $null = $WordDocument.OMaths.Add($range)
    $WordDocument.OMaths.Item($WordDocument.OMaths.Count).BuildUp() | Out-Null
    return $range
}

function Set-WordDocCanvas {
    param(
        [Parameter(Mandatory = $true)] $WordDocument,
        [Parameter(Mandatory = $true)] [double]$WidthInches,
        [Parameter(Mandatory = $true)] [double]$HeightInches
    )

    $WordDocument.PageSetup.TopMargin = 0
    $WordDocument.PageSetup.BottomMargin = 0
    $WordDocument.PageSetup.LeftMargin = 0
    $WordDocument.PageSetup.RightMargin = 0
    $WordDocument.PageSetup.HeaderDistance = 0
    $WordDocument.PageSetup.FooterDistance = 0
    $WordDocument.PageSetup.PageWidth = [int][Math]::Round($WidthInches * 72)
    $WordDocument.PageSetup.PageHeight = [int][Math]::Round($HeightInches * 72)
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $repoRoot = Split-Path -Parent $scriptDir
    $OutputDir = Join-Path $repoRoot "output\visio"
}

$null = New-Item -ItemType Directory -Force -Path $OutputDir
$vsdxPath = Join-Path $OutputDir ($BaseName + ".vsdx")
$pngPath = Join-Path $OutputDir ($BaseName + ".png")

$hat = [string][char]0x0302
$tilde = [string][char]0x0303
$bar = [string][char]0x0304
$supPlus = [string][char]0x207A
$supMinus = [string][char]0x207B
$chi = [string][char]0x03C7
$delta = [string][char]0x03B4
$sum = [string][char]0x2211
$sqrt = [string][char]0x221A
$ellipsis = [string][char]0x2026
$yhat = "y$hat"
$xhat = "x$hat"
$xtilde = "x$tilde"
$xbar = "x$bar"

$titleInit = From-CodePoints -Codes @(0x521D, 0x59CB, 0x5316)
$titleSigma = (From-CodePoints -Codes @(0x751F, 0x6210)) + "sigma" + (From-CodePoints -Codes @(0x70B9))
$titleMeasurement = From-CodePoints -Codes @(0x91CF, 0x6D4B, 0x91CF)
$titleInput = From-CodePoints -Codes @(0x8F93, 0x5165, 0x91CF)
$titleEstimate = From-CodePoints -Codes @(0x4F30, 0x8BA1, 0x91CF)
$titlePrior = From-CodePoints -Codes @(0x5148, 0x9A8C, 0x72B6, 0x6001, 0x4F30, 0x8BA1)
$titleWhere = From-CodePoints -Codes @(0x5176, 0x4E2D)
$titlePriorCov = From-CodePoints -Codes @(0x5148, 0x9A8C, 0x4F30, 0x8BA1, 0x8BEF, 0x5DEE, 0x7684, 0x534F, 0x65B9, 0x5DEE)
$titlePredict = From-CodePoints -Codes @(0x91CF, 0x6D4B, 0x9884, 0x6D4B)
$titlePredictCov = From-CodePoints -Codes @(0x91CF, 0x6D4B, 0x9884, 0x6D4B, 0x7684, 0x534F, 0x65B9, 0x5DEE)
$titleUpdate = From-CodePoints -Codes @(0x72B6, 0x6001, 0x66F4, 0x65B0)

$visio = $null
$document = $null
$page = $null

try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    $visio.AlertResponse = 7

    $document = $visio.Documents.Add("")
    $page = $visio.ActivePage
    $page.Name = "UKF Word Eq Flowchart"
    $page.PageSheet.CellsU("PageWidth").FormulaU = "14 in"
    $page.PageSheet.CellsU("PageHeight").FormulaU = "10.2 in"

    $null = Add-RoundedBlock -Page $page -X1 0.45 -Y1 7.55 -X2 5.45 -Y2 9.65
    $objInit = New-EmptyWordDocumentShape -Page $page -CenterX 2.95 -CenterY 8.60 -WidthInches 4.4 -HeightInches 1.70
    $wInit = $objInit.Object
    Set-WordDocCanvas -WordDocument $wInit -WidthInches 4.4 -HeightInches 1.70
    Add-WordParagraph -WordDocument $wInit -Text $titleInit -FontSize 12 -SpaceAfter 4 | Out-Null
    Add-WordEquation -WordDocument $wInit -LinearText ("x_0 = [0, " + $ellipsis + ", 0]^T") -FontSize 11 -SpaceAfter 2 | Out-Null
    Add-WordEquation -WordDocument $wInit -LinearText ($xhat + "_0^+ = E(" + $xtilde + "_0)") -FontSize 11 -SpaceAfter 2 | Out-Null
    Add-WordEquation -WordDocument $wInit -LinearText ("P_0^+ = E[(" + $xtilde + "_0 - " + $xbar + "_0)(" + $xtilde + "_0 - " + $xbar + "_0)^T]") -FontSize 11 -SpaceAfter 0 | Out-Null

    $null = Add-RoundedBlock -Page $page -X1 6.80 -Y1 7.60 -X2 13.60 -Y2 9.65
    $objSigma = New-EmptyWordDocumentShape -Page $page -CenterX 10.20 -CenterY 8.62 -WidthInches 6.2 -HeightInches 1.72
    $wSigma = $objSigma.Object
    Set-WordDocCanvas -WordDocument $wSigma -WidthInches 6.2 -HeightInches 1.72
    Add-WordParagraph -WordDocument $wSigma -Text $titleSigma -FontSize 12 -SpaceAfter 4 | Out-Null
    Add-WordEquation -WordDocument $wSigma -LinearText ($chi + $tilde + "_(k-1)^(i) = " + $xhat + "_(k-1)^+ + " + $chi + $tilde + "^(i)    i = 1, " + $ellipsis + ", 2n") -FontSize 10.5 -SpaceAfter 2 | Out-Null
    Add-WordParagraph -WordDocument $wSigma -Text $titleWhere -FontSize 10.5 -SpaceAfter 1 | Out-Null
    Add-WordEquation -WordDocument $wSigma -LinearText ($chi + $tilde + "^(i) = (" + $sqrt + "(nP_(k-1)^+))_i^T,    " + $chi + $tilde + "^(n+i) = -(" + $sqrt + "(nP_(k-1)^+))_i^T") -FontSize 10.2 -SpaceAfter 0 | Out-Null

    $null = Add-RoundedBlock -Page $page -X1 1.55 -Y1 5.15 -X2 4.85 -Y2 7.45
    $objMeasurement = New-EmptyWordDocumentShape -Page $page -CenterX 3.20 -CenterY 6.30 -WidthInches 2.7 -HeightInches 1.95
    $wMeasurement = $objMeasurement.Object
    Set-WordDocCanvas -WordDocument $wMeasurement -WidthInches 2.7 -HeightInches 1.95
    Add-WordParagraph -WordDocument $wMeasurement -Text $titleMeasurement -FontSize 12 -SpaceAfter 5 | Out-Null
    Add-WordEquation -WordDocument $wMeasurement -LinearText ("y(t) = [a_y, r]^T") -FontSize 11 -SpaceAfter 5 | Out-Null
    Add-WordParagraph -WordDocument $wMeasurement -Text $titleInput -FontSize 11 -SpaceAfter 1 | Out-Null
    Add-WordEquation -WordDocument $wMeasurement -LinearText ("u_k = " + $delta + "_f(k)") -FontSize 11 -SpaceAfter 0 | Out-Null

    $null = Add-RoundedBlock -Page $page -X1 1.55 -Y1 3.65 -X2 4.85 -Y2 4.95
    $objEstimate = New-EmptyWordDocumentShape -Page $page -CenterX 3.20 -CenterY 4.30 -WidthInches 2.7 -HeightInches 0.95
    $wEstimate = $objEstimate.Object
    Set-WordDocCanvas -WordDocument $wEstimate -WidthInches 2.7 -HeightInches 0.95
    Add-WordParagraph -WordDocument $wEstimate -Text $titleEstimate -FontSize 12 -SpaceAfter 3 | Out-Null
    Add-WordEquation -WordDocument $wEstimate -LinearText ("x(t) = [a_1, a_2, " + $ellipsis + ", a_8]^T") -FontSize 10.5 -SpaceAfter 0 | Out-Null

    $null = Add-RoundedBlock -Page $page -X1 6.80 -Y1 4.35 -X2 13.60 -Y2 6.95
    $objPrior = New-EmptyWordDocumentShape -Page $page -CenterX 10.20 -CenterY 5.65 -WidthInches 6.15 -HeightInches 2.18
    $wPrior = $objPrior.Object
    Set-WordDocCanvas -WordDocument $wPrior -WidthInches 6.15 -HeightInches 2.18
    Add-WordParagraph -WordDocument $wPrior -Text $titlePrior -FontSize 12 -SpaceAfter 4 | Out-Null
    Add-WordEquation -WordDocument $wPrior -LinearText ($xhat + "_k^- = 1/(2n) " + $sum + "_(i=1)^(2n) " + $chi + "_k^(i)") -FontSize 10.7 -SpaceAfter 2 | Out-Null
    Add-WordParagraph -WordDocument $wPrior -Text ($titleWhere + "  " + $chi + "_k^(i) = f(" + $chi + "_(k-1)^(i), u_k, t_k)") -FontSize 10.5 -SpaceAfter 4 | Out-Null
    Add-WordParagraph -WordDocument $wPrior -Text $titlePriorCov -FontSize 10.8 -SpaceAfter 2 | Out-Null
    Add-WordEquation -WordDocument $wPrior -LinearText ("P_k^- = 1/(2n) " + $sum + "_(i=1)^(2n) (" + $chi + "_k^(i) - " + $xhat + "_k^-)(" + $chi + "_k^(i) - " + $xhat + "_k^-)^T + Q_(k-1)") -FontSize 9.9 -SpaceAfter 0 | Out-Null

    $null = Add-RoundedBlock -Page $page -X1 6.85 -Y1 0.45 -X2 13.60 -Y2 3.20
    $objPredict = New-EmptyWordDocumentShape -Page $page -CenterX 10.225 -CenterY 1.825 -WidthInches 6.1 -HeightInches 2.30
    $wPredict = $objPredict.Object
    Set-WordDocCanvas -WordDocument $wPredict -WidthInches 6.1 -HeightInches 2.30
    Add-WordParagraph -WordDocument $wPredict -Text $titlePredict -FontSize 12 -SpaceAfter 4 | Out-Null
    Add-WordEquation -WordDocument $wPredict -LinearText ($yhat + "_k^(i) = h(" + $chi + "_k^(i), t_k)    " + $yhat + "_k = 1/(2n) " + $sum + "_(i=1)^(2n) " + $yhat + "_k^(i)") -FontSize 10.0 -SpaceAfter 4 | Out-Null
    Add-WordParagraph -WordDocument $wPredict -Text $titlePredictCov -FontSize 10.8 -SpaceAfter 2 | Out-Null
    Add-WordEquation -WordDocument $wPredict -LinearText ("P_y = 1/(2n) " + $sum + "_(i=1)^(2n) (" + $yhat + "_k^(i) - " + $yhat + "_k)(" + $yhat + "_k^(i) - " + $yhat + "_k)^T + R_k") -FontSize 9.9 -SpaceAfter 0 | Out-Null

    $null = Add-RoundedBlock -Page $page -X1 0.20 -Y1 0.45 -X2 5.80 -Y2 3.15
    $objUpdate = New-EmptyWordDocumentShape -Page $page -CenterX 3.00 -CenterY 1.80 -WidthInches 5.05 -HeightInches 2.25
    $wUpdate = $objUpdate.Object
    Set-WordDocCanvas -WordDocument $wUpdate -WidthInches 5.05 -HeightInches 2.25
    Add-WordParagraph -WordDocument $wUpdate -Text $titleUpdate -FontSize 12 -SpaceAfter 4 | Out-Null
    Add-WordEquation -WordDocument $wUpdate -LinearText ("P_xy = 1/(2n) " + $sum + "_(i=1)^(2n) (" + $chi + "_k^(i) - " + $xhat + "_k^-)(" + $yhat + "_k^(i) - " + $yhat + "_k)^T") -FontSize 9.8 -SpaceAfter 2 | Out-Null
    Add-WordEquation -WordDocument $wUpdate -LinearText ("K_k = P_xy P_y^(-1)") -FontSize 10.5 -SpaceAfter 2 | Out-Null
    Add-WordEquation -WordDocument $wUpdate -LinearText ($xhat + "_k^+ = " + $xhat + "_k^- + K_k(y_k - " + $yhat + "_k)") -FontSize 10.5 -SpaceAfter 2 | Out-Null
    Add-WordEquation -WordDocument $wUpdate -LinearText ("P_k^+ = P_k^- - K_k P_y K_k^T") -FontSize 10.5 -SpaceAfter 0 | Out-Null

    Add-LineSegment -Page $page -X1 5.45 -Y1 8.60 -X2 6.80 -Y2 8.60 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -X1 10.20 -Y1 7.60 -X2 10.20 -Y2 6.95 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -X1 4.85 -Y1 6.30 -X2 5.90 -Y2 6.30 | Out-Null
    Add-LineSegment -Page $page -X1 5.90 -Y1 6.30 -X2 5.90 -Y2 5.65 | Out-Null
    Add-LineSegment -Page $page -X1 5.90 -Y1 5.65 -X2 6.80 -Y2 5.65 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -X1 10.20 -Y1 4.35 -X2 10.20 -Y2 3.20 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -X1 6.85 -Y1 1.83 -X2 5.80 -Y2 1.83 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -X1 3.20 -Y1 3.15 -X2 3.20 -Y2 3.65 -ArrowAtEnd | Out-Null

    $document.SaveAs($vsdxPath)
    $page.Export($pngPath)

    Write-Output ("OUTPUT_VSDX=" + $vsdxPath)
    Write-Output ("OUTPUT_PNG=" + $pngPath)
}
finally {
    if ($document) {
        try {
            $document.Close()
        } catch {
        }
    }

    if ($visio) {
        try {
            $visio.Quit()
        } catch {
        }
    }

    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
