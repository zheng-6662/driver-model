param(
  [string]$BasePath
)

$ProgressPreference = 'SilentlyContinue'
$base = $BasePath

function Get-Field($content, $prefix) {
  $line = $content | Where-Object { $_ -like "$prefix*" } | Select-Object -First 1
  if ($line) {
    return ($line -replace [regex]::Escape($prefix), '').Trim()
  }
  return $null
}

function Sanitize-FileName([string]$name) {
  foreach ($c in [IO.Path]::GetInvalidFileNameChars()) {
    $name = $name.Replace($c, '_')
  }
  return ($name -replace '\s+', ' ').Trim()
}

function Try-DownloadPdf($url, $outFile) {
  try {
    Invoke-WebRequest -Uri $url -OutFile $outFile -MaximumRedirection 10 -ErrorAction Stop
    $bytes = Get-Content -LiteralPath $outFile -Encoding Byte -TotalCount 5
    if ($bytes.Length -ge 4 -and $bytes[0] -eq 37 -and $bytes[1] -eq 80 -and $bytes[2] -eq 68 -and $bytes[3] -eq 70) {
      return $true
    }
    Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue
    return $false
  } catch {
    if (Test-Path -LiteralPath $outFile) {
      Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue
    }
    return $false
  }
}

function Find-PdfLink($url) {
  try {
    $resp = Invoke-WebRequest -Uri $url -MaximumRedirection 10 -ErrorAction Stop
    $links = @()
    foreach ($l in $resp.Links) {
      if ($l.href -and ($l.href -match '\.pdf($|\?)' -or $l.outerHTML -match 'pdf')) {
        $links += $l.href
      }
    }
    if ($resp.Content -match 'citation_pdf_url"\s+content="([^"]+)"') {
      $links += $Matches[1]
    }
    if ($resp.Content -match 'href="([^"]+\.pdf[^"]*)"') {
      $links += $Matches[1]
    }
    foreach ($href in $links) {
      if (-not $href) {
        continue
      }
      try {
        return ([uri]::new([uri]$resp.BaseResponse.ResponseUri, $href)).AbsoluteUri
      } catch {
      }
    }
  } catch {
  }
  return $null
}

function Get-OpenAlexUrls($doi) {
  $urls = @()
  try {
    $api = 'https://api.openalex.org/works?filter=doi:' + [uri]::EscapeDataString($doi)
    $json = Invoke-RestMethod -Uri $api -ErrorAction Stop
    if ($json.results.Count -gt 0) {
      $w = $json.results[0]
      if ($w.open_access.oa_url) { $urls += $w.open_access.oa_url }
      if ($w.best_oa_location.pdf_url) { $urls += $w.best_oa_location.pdf_url }
      if ($w.best_oa_location.landing_page_url) { $urls += $w.best_oa_location.landing_page_url }
      foreach ($loc in $w.locations) {
        if ($loc.pdf_url) { $urls += $loc.pdf_url }
        if ($loc.is_oa -and $loc.landing_page_url) { $urls += $loc.landing_page_url }
      }
    }
  } catch {
  }
  return $urls | Where-Object { $_ } | Select-Object -Unique
}

$results = @()
Get-ChildItem -LiteralPath $base -Recurse -Filter *.txt |
  Where-Object { $_.Name -notlike '_*' } |
  ForEach-Object {
    $txt = $_
    $content = Get-Content -LiteralPath $txt.FullName
    $title = Get-Field $content 'Title:'
    if (-not $title) {
      $title = $txt.BaseName -replace '^[0-9]{4}_', ''
    }
    $doi = Get-Field $content 'DOI:'
    $year = Get-Field $content 'Year:'
    if (-not $year) {
      if ($txt.BaseName -match '^([0-9]{4})_') {
        $year = $Matches[1]
      } else {
        $year = '0000'
      }
    }

    $existing = Get-ChildItem -LiteralPath $txt.DirectoryName -Filter ($txt.BaseName + '*.pdf') -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existing) {
      $results += [pscustomobject]@{
        Title = $title
        Status = 'already_exists'
        Pdf = $existing.FullName
        Source = ''
      }
      return
    }

    $outName = Sanitize-FileName("${year}_${title}.pdf")
    $outFile = Join-Path $txt.DirectoryName $outName
    $candidateUrls = @()
    $oa = Get-Field $content 'Open access PDF URL:'
    if ($oa) {
      $candidateUrls += $oa
    }
    if ($doi) {
      $candidateUrls += Get-OpenAlexUrls $doi
      $candidateUrls += ('https://doi.org/' + $doi)
    }
    $candidateUrls = $candidateUrls | Where-Object { $_ } | Select-Object -Unique

    $ok = $false
    $src = $null
    foreach ($url in $candidateUrls) {
      if (Try-DownloadPdf $url $outFile) {
        $ok = $true
        $src = $url
        break
      }
      $pdfLink = Find-PdfLink $url
      if ($pdfLink -and (Try-DownloadPdf $pdfLink $outFile)) {
        $ok = $true
        $src = $pdfLink
        break
      }
      Start-Sleep -Milliseconds 300
    }

    $results += [pscustomobject]@{
      Title = $title
      Status = $(if ($ok) { 'downloaded' } else { 'not_found' })
      Pdf = $(if ($ok) { $outFile } else { '' })
      Source = $(if ($ok) { $src } else { ($candidateUrls -join ' | ') })
    }
  }

$results | ConvertTo-Json -Depth 4
