$ErrorActionPreference = "Stop"

# Restrict package manager connection timeouts to 30 seconds
$env:PIP_TIMEOUT = "30"
$env:PIP_DEFAULT_TIMEOUT = "30"
$env:POETRY_HTTP_TIMEOUT = "30"

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
if (Get-Command "pandoc" -ErrorAction SilentlyContinue) {
    poetry run sphinx-build -b doctest -d docs/_build/doctrees docs docs/_build/doctest
} else {
    Write-Host "========================================================================"
    Write-Host "WARNING: 'pandoc' is not installed. Documentation testing is skipped."
    Write-Host "To install pandoc, please refer to: https://pandoc.org/installing.html"
    Write-Host "========================================================================"
}

Write-Host "Fetching vendor dependencies..."
.\fetch_vendor.ps1

Write-Host "Setup complete."
