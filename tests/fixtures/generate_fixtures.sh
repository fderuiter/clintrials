#!/bin/bash
set -e

# Navigate to the project root
cd "$(dirname "$0")/../../"

echo "Building Docker image for R environment..."
docker build -t clintrials-r-env tests/fixtures/

echo "Generating fixtures inside Docker container..."
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$(pwd)/tests/fixtures:/fixtures" -w /fixtures clintrials-r-env Rscript gen_fixtures.R

echo "Done! Fixtures have been updated."
