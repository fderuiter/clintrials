#!/usr/bin/env bash
set -euo pipefail

# Setup clintrials development environment using Poetry.
# Ensures Poetry is installed, project dependencies are installed,
# pre-commit hooks are configured, and the test suite runs.

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry not found. Installing..."
  pip install poetry
fi

echo "Installing project dependencies with Poetry..."
poetry install

echo "Setup complete."

