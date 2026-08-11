#!/usr/bin/env bash
set -euo pipefail

# Restrict package manager connection timeouts to 30 seconds
export PIP_TIMEOUT=30
export PIP_DEFAULT_TIMEOUT=30
export POETRY_HTTP_TIMEOUT=30

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

echo "Running documentation doctests..."
if command -v pandoc >/dev/null 2>&1; then
  poetry run make -C docs doctest
else
  echo "========================================================================"
  echo "WARNING: pandoc is not installed. Documentation testing is skipped."
  echo "To install pandoc, please refer to: https://pandoc.org/installing.html"
  echo "========================================================================"
fi

echo "Fetching vendor dependencies..."
./fetch_vendor.sh

echo "Setup complete."

