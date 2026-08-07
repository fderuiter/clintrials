# SPDX-License-Identifier: MIT

import os
import subprocess
import sys

# Determine the project root directory dynamically
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))


def test_internal_deprecation_warning_fails_pipeline() -> None:
    """Verify that an uncaught internal deprecation warning fails the test suite."""
    test_file_path = os.path.join(TESTS_DIR, "test_tmp_fail_deprecation.py")
    pyproject_path = os.path.join(PROJECT_ROOT, "pyproject.toml")
    try:
        with open(test_file_path, "w") as f:
            f.write("""
import pytest
from clintrials.phase3.gsd import GroupSequentialDesign

def test_internal_deprecation():
    design = GroupSequentialDesign(k=2)
    # This should trigger DeprecationWarning from clintrials
    design.simulate(n_sims=1)
""")
        env = dict(os.environ, PYTHONPATH=PROJECT_ROOT)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file_path, "-c", pyproject_path],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode != 0, (
            f"Expected pytest to fail on internal deprecation, but it passed.\nOutput:\n{result.stdout}\n{result.stderr}"
        )
        assert (
            "DeprecationWarning" in result.stdout
            or "DeprecationWarning" in result.stderr
        )
    finally:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


def test_external_deprecation_warning_passes_pipeline() -> None:
    """Verify that an uncaught external deprecation warning does not fail the test suite."""
    test_file_path = os.path.join(TESTS_DIR, "test_tmp_pass_deprecation.py")
    pyproject_path = os.path.join(PROJECT_ROOT, "pyproject.toml")
    try:
        with open(test_file_path, "w") as f:
            f.write("""
import warnings

def test_external_deprecation():
    # Trigger a DeprecationWarning from an external/non-clintrials module context
    warnings.warn_explicit(
        "external deprecation",
        DeprecationWarning,
        filename="external_module.py",
        lineno=1,
        module="external_module"
    )
""")
        env = dict(os.environ, PYTHONPATH=PROJECT_ROOT)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file_path, "-c", pyproject_path],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, (
            f"Expected pytest to pass on external deprecation, but it failed.\nOutput:\n{result.stdout}\n{result.stderr}"
        )
    finally:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
