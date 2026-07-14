#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

mkdir -p tests/fixtures

echo "Building Docker image for R environment..."
docker build -t clintrials-r-env tests/fixtures/

echo "Generating fixtures inside Docker container..."
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$(pwd):/app" -w /app clintrials-r-env Rscript tests/fixtures/gen_fixtures.R

echo "Done! Fixtures have been updated."
