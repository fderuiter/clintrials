"""PEP 440 version verification tool.

This tool checks that the versions declared in pyproject.toml and the
stlite build-manifest.json (and its wheel filename) are valid PEP 440
version strings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Union

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore

from clintrials.validation import validate_version


def extract_wheel_version(wheel_filename: str) -> str:
    """Extracts the version segment from a wheel filename.

    Args:
        wheel_filename: The filename or path of the wheel.

    Returns:
        The extracted version segment.

    Raises:
        ValueError: If the wheel filename format is invalid.
    """
    name = Path(wheel_filename).name
    if name.endswith(".whl"):
        name = name[:-4]
    parts = name.split("-")
    if len(parts) < 2:
        raise ValueError(f"Invalid wheel filename format: '{wheel_filename}'")
    return parts[1]


def validate_pyproject_toml(path: Union[str, Path]) -> str:
    """Validates the [project] version in pyproject.toml.

    Args:
        path: The path to pyproject.toml.

    Returns:
        The validated version string.

    Raises:
        FileNotFoundError: If pyproject.toml does not exist.
        ValueError: If the project version is missing or invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {path}")

    with p.open("rb") as f:
        data = tomllib.load(f)

    try:
        project_section = data["project"]
        version = project_section["version"]
    except KeyError as e:
        raise ValueError(
            f"pyproject.toml is missing expected section or key: {e}"
        ) from e

    if not isinstance(version, str):
        raise ValueError(
            f"[project] version must be a string, got {type(version).__name__}"
        )

    validate_version(version, "[project] version")
    return version


def validate_build_manifest(path: Union[str, Path]) -> tuple[str, str]:
    """Validates the version and wheel filename in hub/build-manifest.json.

    Args:
        path: The path to build-manifest.json.

    Returns:
        A tuple of (version, wheel_version_segment).

    Raises:
        FileNotFoundError: If build-manifest.json does not exist.
        ValueError: If the manifest format is invalid or versions are invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"build-manifest.json not found at {path}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("build-manifest.json must contain a JSON object.")

    version = data.get("version")
    wheel = data.get("wheel")

    if version is None:
        raise ValueError("build-manifest.json is missing 'version' key.")
    if wheel is None:
        raise ValueError("build-manifest.json is missing 'wheel' key.")

    validate_version(version, "build-manifest.json version")

    wheel_version = extract_wheel_version(wheel)
    validate_version(wheel_version, "build-manifest.json wheel version segment")

    return version, wheel_version


def main(argv: list[str] | None = None) -> None:
    """Runs the version verification process for pyproject.toml and build-manifest.json."""
    if argv is None:
        if any("pytest" in arg for arg in sys.argv):
            argv = []
        else:
            argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="PEP 440 version verification tool.")
    parser.add_argument("manifest_path", nargs="?", default=None, help="Optional alternative path to build-manifest.json")
    args = parser.parse_args(argv)

    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent

    pyproject_path = repo_root / "pyproject.toml"

    is_custom_path = args.manifest_path is not None
    if is_custom_path:
        manifest_path = Path(args.manifest_path)
    else:
        manifest_path = repo_root / "public" / "hub" / "build-manifest.json"

    display_path = manifest_path
    if manifest_path.is_absolute():
        try:
            display_path = manifest_path.relative_to(repo_root)
        except ValueError:
            pass

    errors = []

    try:
        pyproject_version = validate_pyproject_toml(pyproject_path)
        sys.stdout.write(f"Validated pyproject.toml version: {pyproject_version}\n")
    except Exception as e:
        errors.append(f"pyproject.toml validation failed: {e}")

    if manifest_path.exists():
        try:
            manifest_version, wheel_version = validate_build_manifest(manifest_path)
            sys.stdout.write(
                f"Validated {display_path} version: {manifest_version}\n"
            )
            sys.stdout.write(f"Validated wheel version segment: {wheel_version}\n")
        except Exception as e:
            errors.append(f"{display_path} validation failed: {e}")
    else:
        if is_custom_path:
            errors.append(f"Custom build manifest file not found at {display_path}")
        else:
            sys.stdout.write(
                f"WARNING: {display_path} not found, skipping manifest and wheel validation.\n"
            )

    if errors:
        sys.stderr.write("PEP 440 Version Verification Failed!\n")
        for error in errors:
            sys.stderr.write(f" - {error}\n")
        sys.exit(1)

    sys.stdout.write("All version validations passed successfully.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
