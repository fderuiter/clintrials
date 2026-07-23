$ErrorActionPreference = "Stop"

# Setup clintrials development environment using Poetry.
# Ensures Poetry is installed, project dependencies are installed,
# pre-commit hooks are configured, and the test suite runs.

if (-not (Get-Command "poetry" -ErrorAction SilentlyContinue)) {
    Write-Host "Poetry not found. Installing..."
    pipx install poetry
}

Write-Host "Installing project dependencies with Poetry..."
poetry install --all-extras --no-interaction

Write-Host "Installing pre-commit hooks..."
poetry run pre-commit install

Write-Host "Running verification tests..."
poetry run pytest -m "not slow"

Write-Host "Running documentation doctests..."
poetry run sphinx-build -b doctest -d docs/_build/doctrees docs docs/_build/doctest

Write-Host "Fetching vendor dependencies..."
.\fetch_vendor.ps1

Write-Host "Setup complete."
