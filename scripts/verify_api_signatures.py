"""API Signature Verification Script.

This script parses core package modules and extracts their public API signatures
(classes, methods, and functions). It compares them to a baseline JSON manifest
to prevent accidental breaking changes to the public interface.
"""

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path


def get_signature_info(obj):
    """Extract parameter names from a callable object."""
    try:
        sig = inspect.signature(obj)
        params = []
        for name, param in sig.parameters.items():
            if name.startswith('_'):
                continue
            params.append(name)
        return params
    except (ValueError, TypeError):
        return []

def get_public_methods(cls):
    """Get a dictionary of public methods and their signatures for a class."""
    methods = {}
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith('_') or name == '__init__':
            methods[name] = get_signature_info(method)
    return methods

def scan_module(module_name):
    """Scan a module and return a manifest of its public exports."""
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        sys.stdout.write(f"Error: Missing optional dependency to scan {module_name} ({e}).\n")
        sys.stdout.write("Please install with extras, e.g.: poetry install -E viz\n")
        sys.exit(1)

    exports = getattr(mod, '__all__', [])
    if not exports:
        exports = [n for n in dir(mod) if not n.startswith('_')]

    manifest = {}
    for name in exports:
        if name.startswith('_'):
            continue
        obj = getattr(mod, name, None)
        if obj is None:
            continue

        if inspect.isclass(obj):
            manifest[name] = {
                "type": "class",
                "methods": get_public_methods(obj)
            }
        elif inspect.isfunction(obj):
            manifest[name] = {
                "type": "function",
                "parameters": get_signature_info(obj)
            }
    return manifest

def generate_manifest():
    """Generate a manifest for the predefined list of core modules."""
    modules = [
        'clintrials.core',
        'clintrials.dosefinding',
        'clintrials.winratio',
        'clintrials.visualization'
    ]
    manifest = {}
    for m in modules:
        manifest[m] = scan_module(m)
    return manifest

def main():
    """Run the API signature verification process."""
    parser = argparse.ArgumentParser(description="Automated Package-Wide JSON Manifest Hook")
    parser.add_argument('--generate', action='store_true', help='Recreate or update the baseline JSON manifest file')
    parser.add_argument('--manifest', default='api_manifest.json', help='Path to the manifest file')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    current_manifest = generate_manifest()

    if args.generate:
        with open(manifest_path, 'w') as f:
            json.dump(current_manifest, f, indent=2)
        sys.stdout.write(f"Manifest successfully generated and saved to {manifest_path}\n")
        sys.exit(0)

    if not manifest_path.exists():
        sys.stdout.write(f"Error: Manifest file {manifest_path} does not exist. Run with --generate to create it.\n")
        sys.exit(1)

    with open(manifest_path, 'r') as f:
        baseline_manifest = json.load(f)

    diffs = []

    # Compare baseline and current
    for module_name, baseline_exports in baseline_manifest.items():
        if module_name not in current_manifest:
            diffs.append(f"Module '{module_name}' is missing.")
            continue

        current_exports = current_manifest[module_name]

        for name, baseline_obj in baseline_exports.items():
            if name not in current_exports:
                diffs.append(f"Export '{name}' missing in module '{module_name}'.")
                continue

            current_obj = current_exports[name]

            if baseline_obj['type'] != current_obj['type']:
                diffs.append(f"Export '{name}' type changed from {baseline_obj['type']} to {current_obj['type']}.")
                continue

            if baseline_obj['type'] == 'class':
                for method, params in baseline_obj['methods'].items():
                    if method not in current_obj['methods']:
                        diffs.append(f"Method '{method}' missing in class '{name}'.")
                        continue
                    if params != current_obj['methods'][method]:
                        diffs.append(f"Parameters for method '{name}.{method}' changed from {params} to {current_obj['methods'][method]}.")

            elif baseline_obj['type'] == 'function':
                if baseline_obj['parameters'] != current_obj['parameters']:
                    diffs.append(f"Parameters for function '{name}' changed from {baseline_obj['parameters']} to {current_obj['parameters']}.")

    # Also check for newly added things not in baseline
    for module_name, current_exports in current_manifest.items():
        baseline_exports = baseline_manifest.get(module_name, {})
        for name, current_obj in current_exports.items():
            if name not in baseline_exports:
                diffs.append(f"Export '{name}' is newly added to module '{module_name}'.")
                continue

            baseline_obj = baseline_exports[name]
            if baseline_obj['type'] == 'class':
                for method, params in current_obj['methods'].items():
                    if method not in baseline_obj['methods']:
                        diffs.append(f"Method '{method}' is newly added to class '{name}'.")

    if diffs:
        sys.stdout.write("API Signature Mismatch Detected!\n")
        sys.stdout.write("The following differences were found compared to the baseline:\n")
        for diff in diffs:
            sys.stdout.write(f" - {diff}\n")
        sys.stdout.write("\nIf these changes are intentional, run 'poetry run python scripts/verify_api_signatures.py --generate' to update the baseline.\n")
        sys.exit(1)

    sys.stdout.write("API Signatures match the baseline.\n")
    sys.exit(0)

if __name__ == '__main__':
    main()
