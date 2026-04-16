# Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$BaseName = "ukf_flowchart_final",
    [string]$WordSourceDir = "",
    [switch]$ShowPowerPoint,
    [switch]$ShowVisio
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Inches-ToPoints([double]$Value) {
    return $Value * 72.0
}

function Set-VisioCell($Shape, [string]$Name, [string]$Formula) {
    $Shape.CellsU($Name).FormulaU = $Formula
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $repoRoot = Split-Path -Parent $scriptDir
    $OutputDir = Join-Path $repoRoot "output\visio"
}
if ([string]::IsNullOrWhiteSpace($WordSourceDir)) {
    $WordSourceDir = Join-Path $OutputDir "ukf_flowchart_high_fidelity_word_sources"
}

$pptxPath = Join-Path $OutputDir ($BaseName + ".pptx")
$pngPath = Join-Path $OutputDir ($BaseName + ".png")
$vsdxPath = Join-Path $OutputDir ($BaseName + ".vsdx")

$null = New-Item -ItemType Directory -Force -Path $OutputDir

$pageW = 14.0
$pageH = 10.2

$blocks = @(
    @{ Key = "init"; X1 = 0.45; Y1 = 7.55; X2 = 5.45; Y2 = 9.65; InnerW = 4.25; InnerH = 1.50 },
    @{ Key = "sigma"; X1 = 6.80; Y1 = 7.60; X2 = 13.60; Y2 = 9.65; InnerW = 5.90; InnerH = 1.55 },
    @{ Key = "measurement"; X1 = 1.55; Y1 = 5.15; X2 = 4.85; Y2 = 7.45; InnerW = 2.45; InnerH = 1.75 },
    @{ Key = "estimate"; X1 = 1.55; Y1 = 3.65; X2 = 4.85; Y2 = 4.95; InnerW = 2.45; InnerH = 0.78 },
    @{ Key = "prior"; X1 = 6.80; Y1 = 4.35; X2 = 13.60; Y2 = 6.95; InnerW = 5.80; InnerH = 2.00 },
    @{ Key = "predict"; X1 = 6.85; Y1 = 0.45; X2 = 13.60; Y2 = 3.20; InnerW = 5.85; InnerH = 2.05 },
    @{ Key = "update"; X1 = 0.20; Y1 = 0.45; X2 = 5.80; Y2 = 3.15; InnerW = 4.75; InnerH = 2.05 }
)

$word = $null
$ppt = $null
$pres = $null
$visio = $null
$vdoc = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Visible = -1
    $pres = $ppt.Presentations.Add()
    $pres.PageSetup.SlideWidth = Inches-ToPoints $pageW
    $pres.PageSetup.SlideHeight = Inches-ToPoints $pageH
    $slide = $pres.Slides.Add(1, 12)
    $slide.FollowMasterBackground = 0
    $slide.Background.Fill.ForeColor.RGB = 16777215

    foreach ($block in $blocks) {
        $left = Inches-ToPoints $block.X1
        $top = Inches-ToPoints ($pageH - $block.Y2)
        $width = Inches-ToPoints ($block.X2 - $block.X1)
        $height = Inches-ToPoints ($block.Y2 - $block.Y1)

        $outer = $slide.Shapes.AddShape(5, $left, $top, $width, $height)
        $outer.Fill.ForeColor.RGB = 16777215
        $outer.Line.ForeColor.RGB = 0
        $outer.Line.Weight = 1.4
        $outer.Adjustments.Item(1) = 0.12

        $docxPath = Join-Path $WordSourceDir ($block.Key + ".docx")
        $doc = $word.Documents.Open($docxPath, $false, $true)
        try {
            $doc.Range(0, $doc.Content.End - 1).Select()
            $word.Selection.CopyAsPicture()
        }
        finally {
            $doc.Close(0)
        }

        $shapeRange = $slide.Shapes.Paste()
        $pic = $shapeRange.Item(1)
        $pic.LockAspectRatio = -1
        $targetW = Inches-ToPoints $block.InnerW
        $targetH = Inches-ToPoints $block.InnerH
        $sourceRatio = $pic.Width / $pic.Height
        $targetRatio = $targetW / $targetH
        if ($sourceRatio -ge $targetRatio) {
            $pic.Width = $targetW
        } else {
            $pic.Height = $targetH
        }
        $pic.Left = $left + (($width - $pic.Width) / 2.0)
        $pic.Top = $top + (($height - $pic.Height) / 2.0)
    }

    $line1 = $slide.Shapes.AddLine((Inches-ToPoints 5.45), (Inches-ToPoints ($pageH - 8.60)), (Inches-ToPoints 6.80), (Inches-ToPoints ($pageH - 8.60)))
    $line1.Line.EndArrowheadStyle = 3
    $line1.Line.Weight = 1.8
    $line1.Line.ForeColor.RGB = 0

    $line2 = $slide.Shapes.AddLine((Inches-ToPoints 10.20), (Inches-ToPoints ($pageH - 7.60)), (Inches-ToPoints 10.20), (Inches-ToPoints ($pageH - 6.95)))
    $line2.Line.EndArrowheadStyle = 3
    $line2.Line.Weight = 1.8
    $line2.Line.ForeColor.RGB = 0

    $line3 = $slide.Shapes.AddLine((Inches-ToPoints 4.85), (Inches-ToPoints ($pageH - 6.30)), (Inches-ToPoints 5.90), (Inches-ToPoints ($pageH - 6.30)))
    $line3.Line.Weight = 1.8
    $line3.Line.ForeColor.RGB = 0

    $line4 = $slide.Shapes.AddLine((Inches-ToPoints 5.90), (Inches-ToPoints ($pageH - 6.30)), (Inches-ToPoints 5.90), (Inches-ToPoints ($pageH - 5.65)))
    $line4.Line.Weight = 1.8
    $line4.Line.ForeColor.RGB = 0

    $line5 = $slide.Shapes.AddLine((Inches-ToPoints 5.90), (Inches-ToPoints ($pageH - 5.65)), (Inches-ToPoints 6.80), (Inches-ToPoints ($pageH - 5.65)))
    $line5.Line.EndArrowheadStyle = 3
    $line5.Line.Weight = 1.8
    $line5.Line.ForeColor.RGB = 0

    $line6 = $slide.Shapes.AddLine((Inches-ToPoints 10.20), (Inches-ToPoints ($pageH - 4.35)), (Inches-ToPoints 10.20), (Inches-ToPoints ($pageH - 3.20)))
    $line6.Line.EndArrowheadStyle = 3
    $line6.Line.Weight = 1.8
    $line6.Line.ForeColor.RGB = 0

    $line7 = $slide.Shapes.AddLine((Inches-ToPoints 6.85), (Inches-ToPoints ($pageH - 1.83)), (Inches-ToPoints 5.80), (Inches-ToPoints ($pageH - 1.83)))
    $line7.Line.EndArrowheadStyle = 3
    $line7.Line.Weight = 1.8
    $line7.Line.ForeColor.RGB = 0

    $line8 = $slide.Shapes.AddLine((Inches-ToPoints 3.20), (Inches-ToPoints ($pageH - 3.15)), (Inches-ToPoints 3.20), (Inches-ToPoints ($pageH - 3.65)))
    $line8.Line.EndArrowheadStyle = 3
    $line8.Line.Weight = 1.8
    $line8.Line.ForeColor.RGB = 0

    $pres.SaveAs($pptxPath)
    $slide.Export($pngPath, "PNG", [int](2800), [int](2040))

    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $ShowVisio.IsPresent
    $visio.AlertResponse = 7
    $vdoc = $visio.Documents.Add("")
    $page = $visio.ActivePage
    $page.Name = "UKF Final"
    $page.PageSheet.CellsU("PageWidth").FormulaU = ("{0} in" -f $pageW)
    $page.PageSheet.CellsU("PageHeight").FormulaU = ("{0} in" -f $pageH)
    $img = $page.Import($pngPath)
    $img.CellsU("PinX").FormulaU = ("{0} in" -f ($pageW / 2.0))
    $img.CellsU("PinY").FormulaU = ("{0} in" -f ($pageH / 2.0))
    $img.CellsU("Width").FormulaU = ("{0} in" -f $pageW)
    $img.CellsU("Height").FormulaU = ("{0} in" -f $pageH)
    Set-VisioCell -Shape $img -Name "LinePattern" -Formula "0"
    $vdoc.SaveAs($vsdxPath)

    Write-Output ("OUTPUT_PPTX=" + $pptxPath)
    Write-Output ("OUTPUT_PNG=" + $pngPath)
    Write-Output ("OUTPUT_VSDX=" + $vsdxPath)
}
finally {
    if ($vdoc) {
        try { $vdoc.Close() } catch {}
    }
    if ($visio) {
        try { $visio.Quit() } catch {}
    }
    if ($pres) {
        try { $pres.Close() } catch {}
    }
    if ($ppt) {
        try { $ppt.Quit() } catch {}
    }
    if ($word) {
        try { $word.Quit() } catch {}
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
