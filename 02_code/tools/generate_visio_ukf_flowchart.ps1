# Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$BaseName = "ukf_flowchart_editable"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Set-CellFormula {
    param(
        [Parameter(Mandatory = $true)] $Shape,
        [Parameter(Mandatory = $true)] [string]$CellName,
        [Parameter(Mandatory = $true)] [string]$Formula
    )

    $Shape.CellsU($CellName).FormulaU = $Formula
}

function Set-BlockStyle {
    param(
        [Parameter(Mandatory = $true)] $Shape,
        [Parameter(Mandatory = $true)] [string]$FontName,
        [Parameter(Mandatory = $true)] [string]$FontSize
    )

    Set-CellFormula -Shape $Shape -CellName "Rounding" -Formula "0.18 in"
    Set-CellFormula -Shape $Shape -CellName "LineColor" -Formula "RGB(0,0,0)"
    Set-CellFormula -Shape $Shape -CellName "LineWeight" -Formula "1.4 pt"
    Set-CellFormula -Shape $Shape -CellName "FillForegnd" -Formula "RGB(255,255,255)"
    Set-CellFormula -Shape $Shape -CellName "FillPattern" -Formula "1"
    Set-CellFormula -Shape $Shape -CellName "Char.Font" -Formula ('FONT("' + $FontName.Replace('"', '""') + '")')
    Set-CellFormula -Shape $Shape -CellName "Char.Size" -Formula $FontSize
    Set-CellFormula -Shape $Shape -CellName "Char.Color" -Formula "RGB(0,0,0)"
    Set-CellFormula -Shape $Shape -CellName "Para.HorzAlign" -Formula "1"
    Set-CellFormula -Shape $Shape -CellName "VerticalAlign" -Formula "1"
    Set-CellFormula -Shape $Shape -CellName "LeftMargin" -Formula "0.10 in"
    Set-CellFormula -Shape $Shape -CellName "RightMargin" -Formula "0.10 in"
    Set-CellFormula -Shape $Shape -CellName "TopMargin" -Formula "0.08 in"
    Set-CellFormula -Shape $Shape -CellName "BottomMargin" -Formula "0.08 in"
}

function Add-Block {
    param(
        [Parameter(Mandatory = $true)] $Page,
        [Parameter(Mandatory = $true)] [AllowEmptyCollection()] [System.Collections.Generic.List[object]]$ComObjects,
        [Parameter(Mandatory = $true)] [hashtable]$Spec
    )

    $shape = $Page.DrawRectangle($Spec.X1, $Spec.Y1, $Spec.X2, $Spec.Y2)
    $shape.Text = $Spec.Text
    Set-BlockStyle -Shape $shape -FontName $Spec.FontName -FontSize $Spec.FontSize
    $ComObjects.Add($shape)
    return $shape
}

function Add-LineSegment {
    param(
        [Parameter(Mandatory = $true)] $Page,
        [Parameter(Mandatory = $true)] [AllowEmptyCollection()] [System.Collections.Generic.List[object]]$ComObjects,
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

    $ComObjects.Add($line)
    return $line
}

function From-CodePoints {
    param(
        [Parameter(Mandatory = $true)]
        [int[]]$Codes
    )

    return (-join ($Codes | ForEach-Object { [char]$_ }))
}

function Join-Lines {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string[]]$Lines
    )

    return [string]::Join("`r`n", $Lines)
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $repoRoot = Split-Path -Parent $scriptDir
    $OutputDir = Join-Path $repoRoot "output\visio"
}

$null = New-Item -ItemType Directory -Force -Path $OutputDir
$vsdxPath = Join-Path $OutputDir ($BaseName + ".vsdx")

$visio = $null
$doc = $null
$page = $null
$comObjects = [System.Collections.Generic.List[object]]::new()

try {
    $titleInit = From-CodePoints -Codes @(0x521D, 0x59CB, 0x5316)
    $titleSigma = (Join-Lines -Lines @(
        ((From-CodePoints -Codes @(0x751F, 0x6210)) + " sigma " + (From-CodePoints -Codes @(0x70B9)))
    ))
    $titleMeasurement = From-CodePoints -Codes @(0x91CF, 0x6D4B, 0x91CF)
    $titleInput = From-CodePoints -Codes @(0x8F93, 0x5165, 0x91CF)
    $titleEstimate = From-CodePoints -Codes @(0x4F30, 0x8BA1, 0x91CF)
    $titlePrior = From-CodePoints -Codes @(0x5148, 0x9A8C, 0x72B6, 0x6001, 0x4F30, 0x8BA1)
    $titlePriorCov = From-CodePoints -Codes @(0x5148, 0x9A8C, 0x4F30, 0x8BA1, 0x8BEF, 0x5DEE, 0x7684, 0x534F, 0x65B9, 0x5DEE)
    $titlePredict = From-CodePoints -Codes @(0x91CF, 0x6D4B, 0x9884, 0x6D4B)
    $titlePredictCov = From-CodePoints -Codes @(0x91CF, 0x6D4B, 0x9884, 0x6D4B, 0x7684, 0x534F, 0x65B9, 0x5DEE)
    $titleUpdate = From-CodePoints -Codes @(0x72B6, 0x6001, 0x66F4, 0x65B0)
    $titleWhere = From-CodePoints -Codes @(0x5176, 0x4E2D)

    $blocks = @(
        @{
            Key = "init"
            X1 = 0.45; Y1 = 7.55; X2 = 5.45; Y2 = 9.65
            FontName = "Cambria Math"; FontSize = "10 pt"
            Text = Join-Lines -Lines @(
                $titleInit,
                "",
                "x_0 = [0, ..., 0]^T",
                "xhat_0+ = E(xtilde_0)",
                "P_0+ = E[(xtilde_0 - xbar_0)(xtilde_0 - xbar_0)^T]"
            )
        },
        @{
            Key = "sigma"
            X1 = 6.80; Y1 = 7.60; X2 = 13.60; Y2 = 9.65
            FontName = "Cambria Math"; FontSize = "9 pt"
            Text = Join-Lines -Lines @(
                $titleSigma,
                "",
                "chi_tilde_(k-1)^(i) = xhat_(k-1)+ + chi_tilde^(i)    i = 1, ..., 2n",
                "",
                $titleWhere,
                "chi_tilde^(i) = (sqrt(nP_(k-1)+))_i^T,",
                "chi_tilde^(n+i) = -(sqrt(nP_(k-1)+))_i^T"
            )
        },
        @{
            Key = "measurement"
            X1 = 1.55; Y1 = 5.15; X2 = 4.85; Y2 = 7.45
            FontName = "Cambria Math"; FontSize = "10 pt"
            Text = Join-Lines -Lines @(
                $titleMeasurement,
                "",
                "y(t) = [a_y, r]^T",
                "",
                $titleInput,
                "u_k = delta_f(k)"
            )
        },
        @{
            Key = "estimate"
            X1 = 1.55; Y1 = 3.65; X2 = 4.85; Y2 = 4.95
            FontName = "Cambria Math"; FontSize = "10 pt"
            Text = Join-Lines -Lines @(
                $titleEstimate,
                "",
                "x(t) = [a_1, a_2, ..., a_8]^T"
            )
        },
        @{
            Key = "prior"
            X1 = 6.80; Y1 = 4.35; X2 = 13.60; Y2 = 6.95
            FontName = "Cambria Math"; FontSize = "8.5 pt"
            Text = Join-Lines -Lines @(
                $titlePrior,
                "",
                "xhat_k- = (1 / 2n) Sum(i=1 to 2n) chi_k^(i)",
                ($titleWhere + " chi_k^(i) = f(chi_(k-1)^(i), u_k, t_k)"),
                "",
                $titlePriorCov,
                "",
                "P_k- = (1 / 2n) Sum(i=1 to 2n)",
                "       (chi_k^(i) - xhat_k-)(chi_k^(i) - xhat_k-)^T + Q_(k-1)"
            )
        },
        @{
            Key = "predict"
            X1 = 6.85; Y1 = 0.45; X2 = 13.60; Y2 = 3.20
            FontName = "Cambria Math"; FontSize = "8.5 pt"
            Text = Join-Lines -Lines @(
                $titlePredict,
                "",
                "yhat_k^(i) = h(chi_k^(i), t_k)",
                "yhat_k = (1 / 2n) Sum(i=1 to 2n) yhat_k^(i)",
                "",
                $titlePredictCov,
                "",
                "P_y = (1 / 2n) Sum(i=1 to 2n)",
                "      (yhat_k^(i) - yhat_k)(yhat_k^(i) - yhat_k)^T + R_k"
            )
        },
        @{
            Key = "update"
            X1 = 0.20; Y1 = 0.45; X2 = 5.80; Y2 = 3.15
            FontName = "Cambria Math"; FontSize = "8.5 pt"
            Text = Join-Lines -Lines @(
                $titleUpdate,
                "",
                "P_xy = (1 / 2n) Sum(i=1 to 2n)",
                "       (chi_k^(i) - xhat_k-)(yhat_k^(i) - yhat_k)^T",
                "K_k = P_xy P_y^-1",
                "xhat_k+ = xhat_k- + K_k (y_k - yhat_k)",
                "P_k+ = P_k- - K_k P_y K_k^T"
            )
        }
    )

    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false
    $visio.AlertResponse = 7

    $doc = $visio.Documents.Add("")
    $page = $visio.ActivePage
    $page.Name = "UKF Flowchart"
    Set-CellFormula -Shape $page.PageSheet -CellName "PageWidth" -Formula "14 in"
    Set-CellFormula -Shape $page.PageSheet -CellName "PageHeight" -Formula "10.2 in"

    $shapeMap = @{}
    foreach ($block in $blocks) {
        $shapeMap[$block.Key] = Add-Block -Page $page -ComObjects $comObjects -Spec $block
    }

    Add-LineSegment -Page $page -ComObjects $comObjects -X1 5.45 -Y1 8.60 -X2 6.80 -Y2 8.60 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -ComObjects $comObjects -X1 10.20 -Y1 7.60 -X2 10.20 -Y2 6.95 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -ComObjects $comObjects -X1 4.85 -Y1 6.30 -X2 5.90 -Y2 6.30 | Out-Null
    Add-LineSegment -Page $page -ComObjects $comObjects -X1 5.90 -Y1 6.30 -X2 5.90 -Y2 5.65 | Out-Null
    Add-LineSegment -Page $page -ComObjects $comObjects -X1 5.90 -Y1 5.65 -X2 6.80 -Y2 5.65 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -ComObjects $comObjects -X1 10.20 -Y1 4.35 -X2 10.20 -Y2 3.20 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -ComObjects $comObjects -X1 6.85 -Y1 1.83 -X2 5.80 -Y2 1.83 -ArrowAtEnd | Out-Null
    Add-LineSegment -Page $page -ComObjects $comObjects -X1 3.20 -Y1 3.15 -X2 3.20 -Y2 3.65 -ArrowAtEnd | Out-Null

    $doc.SaveAs($vsdxPath)
    $doc.Saved = $true

    Write-Output ("OUTPUT_VSDX=" + $vsdxPath)
    Write-Output ("BLOCK_COUNT=" + $blocks.Count)
    Write-Output ("LINE_COUNT=8")
}
finally {
    for ($i = $comObjects.Count - 1; $i -ge 0; $i--) {
        if ($null -ne $comObjects[$i]) {
            [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($comObjects[$i])
        }
    }

    if ($null -ne $page) {
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($page)
    }

    if ($null -ne $doc) {
        try {
            $doc.Close()
        } catch {
        }
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc)
    }

    if ($null -ne $visio) {
        try {
            $visio.Quit()
        } catch {
        }
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($visio)
    }

    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
