"""API Signature Verification Script.

This script parses core package modules and extracts their public API signatures
(classes, methods, and functions). It compares them to a baseline JSON manifest
to prevent accidental breaking changes to the public interface.
"""

import argparse
import importlib
import inspect
import json
import re
import sys
import typing
from pathlib import Path


def clean_dynamic_info(val: typing.Optional[str]) -> typing.Optional[str]:
    """Normalize hex memory addresses to make representations deterministic across runs."""
    if val is None:
        return None
    return re.sub(r"at 0x[0-9a-fA-F]+", "at 0x...", val)


def get_signature_info(obj: typing.Any) -> typing.List[typing.Dict[str, typing.Any]]:
    """Extract parameters, annotations, kinds, and default values from a callable object."""
    try:
        sig = inspect.signature(obj)
        params = []
        for name, param in sig.parameters.items():
            if name.startswith("_"):
                continue

            annotation = None
            if param.annotation is not inspect.Parameter.empty:
                if hasattr(param.annotation, "__name__"):
                    annotation = param.annotation.__name__
                else:
                    annotation = str(param.annotation)
            annotation = clean_dynamic_info(annotation)

            default = None
            if param.default is not inspect.Parameter.empty:
                default = repr(param.default)
            default = clean_dynamic_info(default)

            params.append({
                "name": name,
                "kind": param.kind.name,
                "annotation": annotation,
                "default": default,
            })
        return params
    except (ValueError, TypeError):
        return []


def get_public_methods(cls: typing.Any) -> typing.Dict[str, typing.Any]:
    """Get a dictionary of public methods and properties and their signatures for a class."""
    methods: typing.Dict[str, typing.Any] = {}
    for name, value in inspect.getmembers(cls):
        if name.startswith("_") and name != "__init__":
            continue

        static_val = inspect.getattr_static(cls, name)

        if isinstance(static_val, property):
            methods[name] = {"type": "property"}
        elif isinstance(static_val, classmethod):
            methods[name] = {
                "type": "classmethod",
                "parameters": get_signature_info(static_val.__func__),
            }
        elif isinstance(static_val, staticmethod):
            methods[name] = {
                "type": "staticmethod",
                "parameters": get_signature_info(static_val.__func__),
            }
        elif inspect.isroutine(value):
            methods[name] = {
                "type": "method",
                "parameters": get_signature_info(value),
            }
    return methods


def scan_module(module_name: str) -> typing.Dict[str, typing.Any]:
    """Scan a module and return a manifest of its public exports."""
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        sys.stdout.write(f"Error: Missing optional dependency to scan {module_name} ({e}).\n")
        sys.stdout.write("Please install with extras, e.g.: poetry install -E viz\n")
        sys.exit(1)

    exports = getattr(mod, "__all__", [])
    if not exports:
        exports = [n for n in dir(mod) if not n.startswith("_")]

    manifest: typing.Dict[str, typing.Any] = {}
    for name in exports:
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        if obj is None:
            continue

        if inspect.isclass(obj):
            manifest[name] = {
                "type": "class",
                "methods": get_public_methods(obj),
            }
        elif inspect.isfunction(obj):
            manifest[name] = {
                "type": "function",
                "parameters": get_signature_info(obj),
            }
    return manifest


def generate_manifest() -> typing.Dict[str, typing.Any]:
    """Generate a manifest for all dynamically discovered submodules of clintrials."""
    import pkgutil

    import clintrials

    manifest = {}
    # Dynamically find all submodules under clintrials package
    for module_info in pkgutil.walk_packages(clintrials.__path__, clintrials.__name__ + "."):
        module_name = module_info.name

        # Skip private submodules
        parts = module_name.split(".")
        if any(p.startswith("_") for p in parts):
            continue

        # Scan the module
        module_manifest = scan_module(module_name)
        if module_manifest:
            manifest[module_name] = module_manifest

    return manifest


def compare_parameters(baseline_params: typing.Any, current_params: typing.Any) -> typing.List[str]:
    """Compare two parameter lists and return a list of differences."""
    if baseline_params == current_params:
        return []

    # If they are different types or structure (e.g. baseline is old simple list of strings)
    # We should handle backward compatibility/migration elegantly or report mismatch
    if not isinstance(baseline_params, list):
        return ["invalid parameter list structure in baseline"]

    # Check if baseline is in the old format (list of strings)
    if baseline_params and isinstance(baseline_params[0], str):
        # Converting old format parameters to names list for comparison
        current_names = (
            [p["name"] for p in current_params]
            if isinstance(current_params, list)
            and current_params
            and isinstance(current_params[0], dict)
            else current_params
        )
        if baseline_params != current_names:
            return [f"parameter list changed from names {baseline_params} to {current_names}"]
        return []

    diffs = []
    baseline_by_name = {p["name"]: p for p in baseline_params if isinstance(p, dict) and "name" in p}
    current_by_name = {p["name"]: p for p in current_params if isinstance(p, dict) and "name" in p}

    # Check for missing parameters
    for name, b_param in baseline_by_name.items():
        if name not in current_by_name:
            diffs.append(f"parameter '{name}' was removed")
            continue

        c_param = current_by_name[name]
        param_changes = []
        if b_param.get("kind") != c_param.get("kind"):
            param_changes.append(f"kind changed from {b_param.get('kind')} to {c_param.get('kind')}")
        if b_param.get("annotation") != c_param.get("annotation"):
            param_changes.append(
                f"annotation changed from {b_param.get('annotation')} to {c_param.get('annotation')}"
            )
        if b_param.get("default") != c_param.get("default"):
            param_changes.append(
                f"default value changed from {b_param.get('default')} to {c_param.get('default')}"
            )

        if param_changes:
            diffs.append(f"parameter '{name}' modified ({', '.join(param_changes)})")

    # Check for newly added parameters
    for name in current_by_name:
        if name not in baseline_by_name:
            diffs.append(f"parameter '{name}' was added")

    # Check if order of parameters changed
    b_names = [p["name"] for p in baseline_params if isinstance(p, dict) and "name" in p]
    c_names = [p["name"] for p in current_params if isinstance(p, dict) and "name" in p]
    if b_names != c_names and set(b_names) == set(c_names):
        diffs.append("parameter order changed")

    return diffs


def compare_manifests(
    baseline_manifest: typing.Dict[str, typing.Any], current_manifest: typing.Dict[str, typing.Any]
) -> typing.List[str]:
    """Compare baseline and current manifests and return a list of differences."""
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

            if baseline_obj.get("type") != current_obj.get("type"):
                diffs.append(
                    f"Export '{name}' type changed from {baseline_obj.get('type')} to {current_obj.get('type')}."
                )
                continue

            if baseline_obj.get("type") == "class":
                baseline_methods = baseline_obj.get("methods", {})
                current_methods = current_obj.get("methods", {})

                for method, b_method_info in baseline_methods.items():
                    if method not in current_methods:
                        diffs.append(f"Method/Property '{method}' missing in class '{name}'.")
                        continue

                    c_method_info = current_methods[method]

                    # Support old manifest format compatibility where baseline method was just a list of parameters
                    if isinstance(b_method_info, list):
                        # Convert old format to check parameters
                        c_params = (
                            c_method_info.get("parameters", [])
                            if isinstance(c_method_info, dict)
                            else c_method_info
                        )
                        param_diffs = compare_parameters(b_method_info, c_params)
                        if param_diffs:
                            diffs.append(
                                f"Parameters for method '{name}.{method}' changed: {'; '.join(param_diffs)}."
                            )
                        continue

                    if b_method_info.get("type") != c_method_info.get("type"):
                        diffs.append(
                            f"Member '{method}' in class '{name}' type changed from {b_method_info.get('type')} to {c_method_info.get('type')}."
                        )
                        continue

                    if b_method_info.get("type") in ("method", "classmethod", "staticmethod"):
                        b_params = b_method_info.get("parameters", [])
                        c_params = c_method_info.get("parameters", [])
                        param_diffs = compare_parameters(b_params, c_params)
                        if param_diffs:
                            diffs.append(
                                f"Parameters for {b_method_info.get('type')} '{name}.{method}' changed: {'; '.join(param_diffs)}."
                            )

            elif baseline_obj.get("type") == "function":
                b_params = baseline_obj.get("parameters", [])
                c_params = current_obj.get("parameters", [])
                param_diffs = compare_parameters(b_params, c_params)
                if param_diffs:
                    diffs.append(f"Parameters for function '{name}' changed: {'; '.join(param_diffs)}.")

    # Also check for newly added things not in baseline
    for module_name, current_exports in current_manifest.items():
        baseline_exports = baseline_manifest.get(module_name, {})
        for name, current_obj in current_exports.items():
            if name not in baseline_exports:
                diffs.append(f"Export '{name}' is newly added to module '{module_name}'.")
                continue

            baseline_obj = baseline_exports[name]
            if baseline_obj.get("type") == "class" and current_obj.get("type") == "class":
                current_methods = current_obj.get("methods", {})
                baseline_methods = baseline_obj.get("methods", {})
                for method in current_methods:
                    if method not in baseline_methods:
                        diffs.append(f"Method/Property '{method}' is newly added to class '{name}'.")

    return diffs


def main() -> None:
    """Run the API signature verification process."""
    parser = argparse.ArgumentParser(description="Automated Package-Wide JSON Manifest Hook")
    parser.add_argument("--generate", action="store_true", help="Recreate or update the baseline JSON manifest file")
    parser.add_argument("--manifest", default="api_manifest.json", help="Path to the manifest file")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    current_manifest = generate_manifest()

    if args.generate:
        with open(manifest_path, "w") as f:
            json.dump(current_manifest, f, indent=2)
        sys.stdout.write(f"Manifest successfully generated and saved to {manifest_path}\n")
        sys.exit(0)

    if not manifest_path.exists():
        sys.stdout.write(f"Error: Manifest file {manifest_path} does not exist. Run with --generate to create it.\n")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        baseline_manifest = json.load(f)

    diffs = compare_manifests(baseline_manifest, current_manifest)

    if diffs:
        sys.stdout.write("API Signature Mismatch Detected!\n")
        sys.stdout.write("The following differences were found compared to the baseline:\n")
        for diff in diffs:
            sys.stdout.write(f" - {diff}\n")
        sys.stdout.write(
            "\nIf these changes are intentional, run 'poetry run python scripts/verify_api_signatures.py --generate' to update the baseline.\n"
        )
        sys.exit(1)

    sys.stdout.write("API Signatures match the baseline.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
