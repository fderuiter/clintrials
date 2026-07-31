#!/usr/bin/env bash
set -euo pipefail

echo "Fetching external vendor dependencies..."

# Hub dependencies
mkdir -p hub/vendor
curl -sL "https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js" -o hub/vendor/iframeResizer.contentWindow.min.js

# Docs dependencies
mkdir -p docs/_static/vendor
curl -sL "https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.min.js" -o docs/_static/vendor/iframeResizer.min.js

echo "Vendor dependencies fetched successfully."
