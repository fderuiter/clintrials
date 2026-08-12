#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Fetching external vendor dependencies..."

# Helper function to download file with timeout and retry
download_with_retry() {
    local url="$1"
    local output_path="$2"
    local feature_desc="$3"
    
    local max_attempts=4  # 1 initial attempt + up to 3 retries = 4 attempts total
    local attempt=1
    local success=0
    
    # Ensure parent directory exists
    mkdir -p "$(dirname "$output_path")"
    
    while [ $attempt -le $max_attempts ]; do
        echo "Downloading $url (attempt $attempt/$max_attempts)..."
        # We use -f to fail on HTTP errors (e.g. 404, 500)
        # We use --connect-timeout 10 and --max-time 10 to limit connection and total attempt time to 10s
        if curl -sLf --connect-timeout 10 --max-time 10 "$url" -o "$output_path"; then
            success=1
            break
        else
            echo "Attempt $attempt failed." >&2
            attempt=$((attempt + 1))
            if [ $attempt -le $max_attempts ]; then
                sleep 1
            fi
        fi
    done
    
    if [ $success -eq 0 ]; then
        echo "========================================================================" >&2
        echo "WARNING: Failed to download unreachable dependency:" >&2
        echo "  URL: $url" >&2
        echo "  Destination: $output_path" >&2
        echo "Impacted frontend feature:" >&2
        echo "  $feature_desc" >&2
        echo "========================================================================" >&2
    fi
    return 0
}

# Hub dependencies
download_with_retry \
    "https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js" \
    "$SCRIPT_DIR/hub/vendor/iframeResizer.contentWindow.min.js" \
    "nested client-side iframe communication and automatic height resizing of the embedded Simulation Hub dashboard inside parent layouts"

# Plotly visualization dependency
download_with_retry \
    "https://cdn.plot.ly/plotly-2.24.1.min.js" \
    "$SCRIPT_DIR/hub/vendor/plotly-2.24.1.min.js" \
    "interactive charting and scientific visualization of clinical trial simulations within the Simulation Hub dashboard"

# Docs dependencies
download_with_retry \
    "https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.min.js" \
    "$SCRIPT_DIR/docs/_static/vendor/iframeResizer.min.js" \
    "interactive embedded frame resizing and layout responsiveness within clinical trials documentation pages, such as the Simulation Hub drawer"

echo "Vendor dependencies fetched successfully."
