#!/usr/bin/env bash
# Synchronize the static analysis whitelist with active line references.

echo "Generating whitelist..."
poetry run vulture clintrials --make-whitelist > .vulture_whitelist.py || true
echo "Whitelist updated successfully."
