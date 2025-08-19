#!/usr/bin/env bash
# Setup script for the clintrials project.
# Installs Poetry if necessary, installs dependencies,
# and configures pre-commit hooks.

set -euo pipefail

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry not found. Installing..."
  pip install --user poetry
fi

poetry install
poetry run pre-commit install

echo "Setup complete. Activate the environment with 'poetry shell'."
