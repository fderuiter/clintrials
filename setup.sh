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

echo "Fetching vendor dependencies..."
./fetch_vendor.sh

if [ "${SETUP_NO_TESTS:-0}" = "1" ] || [ "${SETUP_NO_TESTS:-}" = "true" ]; then
  echo "Skipping git hooks configuration and verification checks in non-interactive/CI mode."
else
  echo "Installing pre-commit hooks..."
  poetry run pre-commit install

  echo "Running verification tests..."
  poetry run pytest -m "not slow"

  echo "Running documentation doctests..."
  poetry run make -C docs doctest
fi

echo "Setup complete."

