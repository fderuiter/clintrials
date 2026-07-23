$ErrorActionPreference = "Stop"

Write-Host "Fetching external vendor dependencies..."

# Hub dependencies
$hubDir = "hub/vendor"
if (-not (Test-Path $hubDir)) {
    New-Item -ItemType Directory -Force -Path $hubDir | Out-Null
}
Invoke-WebRequest -Uri "https://cdn.jsdelivr.net/npm/@stlite/mountable@0.55.0/build/stlite.css" -OutFile "$hubDir/stlite.css"
Invoke-WebRequest -Uri "https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js" -OutFile "$hubDir/iframeResizer.contentWindow.min.js"
Invoke-WebRequest -Uri "https://cdn.jsdelivr.net/npm/@stlite/mountable@0.55.0/build/stlite.js" -OutFile "$hubDir/stlite.js"

# Docs dependencies
$docsDir = "docs/_static/vendor"
if (-not (Test-Path $docsDir)) {
    New-Item -ItemType Directory -Force -Path $docsDir | Out-Null
}
Invoke-WebRequest -Uri "https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.min.js" -OutFile "$docsDir/iframeResizer.min.js"

Write-Host "Vendor dependencies fetched successfully."
