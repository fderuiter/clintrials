import json
from pathlib import Path

import pytest

from scripts.verify_pep440_version import (
    extract_wheel_version,
    validate_build_manifest,
    validate_pyproject_toml,
)


def test_extract_wheel_version() -> None:
    # Valid wheel filenames
    assert extract_wheel_version("clintrials-0.1.4-py3-none-any.whl") == "0.1.4"
    assert extract_wheel_version("clintrials-1.0.0b1-py3-none-any.whl") == "1.0.0b1"
    assert extract_wheel_version("sub_dir/clintrials-0.1.4-py3-none-any.whl") == "0.1.4"
    assert extract_wheel_version("clintrials-latest-py3-none-any.whl") == "latest"

    # Invalid wheel filename formats
    with pytest.raises(ValueError, match="Invalid wheel filename format"):
        extract_wheel_version("no_hyphens.whl")
    with pytest.raises(ValueError, match="Invalid wheel filename format"):
        extract_wheel_version("clintrials")


def test_validate_pyproject_toml_valid(tmp_path: Path) -> None:
    toml_content = """
[tool.poetry]
name = "clintrials"
version = "0.1.4"
description = "clintrials"
"""
    p = tmp_path / "pyproject.toml"
    p.write_text(toml_content, encoding="utf-8")

    assert validate_pyproject_toml(p) == "0.1.4"


def test_validate_pyproject_toml_invalid(tmp_path: Path) -> None:
    # "latest" version should fail
    toml_content = """
[tool.poetry]
version = "latest"
"""
    p = tmp_path / "pyproject.toml"
    p.write_text(toml_content, encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[tool\.poetry\] version must be a valid PEP 440 version string"):
        validate_pyproject_toml(p)


def test_validate_pyproject_toml_missing_key(tmp_path: Path) -> None:
    toml_content = """
[tool.not_poetry]
version = "0.1.4"
"""
    p = tmp_path / "pyproject.toml"
    p.write_text(toml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="pyproject.toml is missing expected section or key"):
        validate_pyproject_toml(p)


def test_validate_pyproject_toml_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="pyproject.toml not found"):
        validate_pyproject_toml("nonexistent_file.toml")


def test_validate_build_manifest_valid(tmp_path: Path) -> None:
    manifest_data = {
        "version": "0.1.4",
        "wheel": "clintrials-0.1.4-py3-none-any.whl"
    }
    p = tmp_path / "build-manifest.json"
    p.write_text(json.dumps(manifest_data), encoding="utf-8")

    v, wv = validate_build_manifest(p)
    assert v == "0.1.4"
    assert wv == "0.1.4"


def test_validate_build_manifest_invalid_version(tmp_path: Path) -> None:
    manifest_data = {
        "version": "latest",
        "wheel": "clintrials-0.1.4-py3-none-any.whl"
    }
    p = tmp_path / "build-manifest.json"
    p.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="build-manifest.json version must be a valid PEP 440 version string"):
        validate_build_manifest(p)


def test_validate_build_manifest_invalid_wheel_version(tmp_path: Path) -> None:
    manifest_data = {
        "version": "0.1.4",
        "wheel": "clintrials-latest-py3-none-any.whl"
    }
    p = tmp_path / "build-manifest.json"
    p.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="build-manifest.json wheel version segment must be a valid PEP 440 version string"):
        validate_build_manifest(p)


def test_validate_build_manifest_missing_keys(tmp_path: Path) -> None:
    manifest_data = {
        "version": "0.1.4"
    }
    p = tmp_path / "build-manifest.json"
    p.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="build-manifest.json is missing"):
        validate_build_manifest(p)


def test_validate_build_manifest_invalid_json_type(tmp_path: Path) -> None:
    p = tmp_path / "build-manifest.json"
    p.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    with pytest.raises(ValueError, match="build-manifest.json must contain a JSON object"):
        validate_build_manifest(p)


def test_validate_build_manifest_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="build-manifest.json not found"):
        validate_build_manifest("nonexistent_file.json")
