#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

echo "Generating fixtures using Python environment..."
poetry run python tests/fixtures/gen_fixtures.py

echo "Done! Fixtures have been updated."
