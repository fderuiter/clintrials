$ErrorActionPreference = "Stop"

Write-Host "Fetching external vendor dependencies..."

function Download-FileWithRetry {
    param (
        [string]$Uri,
        [string]$OutFile,
        [string]$FeatureDesc
    )
    
    $maxAttempts = 4  # 1 initial + 3 retries
    $timeoutSec = 10
    $attempt = 1
    $success = $false
    
    # Ensure parent directory exists
    $parentDir = Split-Path -Path $OutFile -Parent
    if (-not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Force -Path $parentDir | Out-Null
    }
    
    while ($attempt -le $maxAttempts) {
        Write-Host "Downloading $Uri (attempt $attempt/$maxAttempts)..."
        try {
            # -TimeoutSec specifies the timeout for the request in seconds.
            # -ErrorAction Stop ensures any HTTP or connection failure triggers the catch block.
            # We use -UseBasicParsing because on system without IE configured, Invoke-WebRequest might fail without it.
            Invoke-WebRequest -Uri $Uri -OutFile $OutFile -TimeoutSec $timeoutSec -ErrorAction Stop -UseBasicParsing
            $success = $true
            break
        }
        catch {
            Write-Host "Attempt $attempt failed: $_"
            $attempt++
            if ($attempt -le $maxAttempts) {
                Start-Sleep -Seconds 1
            }
        }
    }
    
    if (-not $success) {
        Write-Warning "========================================================================"
        Write-Warning "WARNING: Failed to download unreachable dependency:"
        Write-Warning "  URL: $Uri"
        Write-Warning "  Destination: $OutFile"
        Write-Warning "Impacted frontend feature:"
        Write-Warning "  $FeatureDesc"
        Write-Warning "========================================================================"
    }
}

Download-FileWithRetry `
    -Uri "https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js" `
    -OutFile "hub/vendor/iframeResizer.contentWindow.min.js" `
    -FeatureDesc "nested client-side iframe communication and automatic height resizing of the embedded Simulation Hub dashboard inside parent layouts"

Download-FileWithRetry `
    -Uri "https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.min.js" `
    -OutFile "docs/_static/vendor/iframeResizer.min.js" `
    -FeatureDesc "interactive embedded frame resizing and layout responsiveness within clinical trials documentation pages, such as the Simulation Hub drawer"

Write-Host "Vendor dependencies fetched successfully."
