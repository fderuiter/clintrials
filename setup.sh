#!/usr/bin/env bash
set -euo pipefail

# Setup clintrials development environment using Poetry.
# Ensures Poetry is installed, project dependencies are installed,
# pre-commit hooks are configured, and the test suite runs.

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry not found. Installing..."
  pipx install poetry
fi

echo "Installing project dependencies with Poetry..."
poetry install --all-extras --no-interaction

echo "Installing pre-commit hooks..."
poetry run pre-commit install

echo "Running verification tests..."
poetry run pytest -m "not slow"

echo "Fetching vendor dependencies..."
./fetch_vendor.sh

echo "Setup complete."

