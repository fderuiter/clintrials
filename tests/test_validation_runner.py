# SPDX-License-Identifier: MIT

"""Tests for clintrials.cli.validation_runner.

This test module verifies the correctness and output of the GxP qualification validation CLI runner.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from clintrials.cli.validation_runner import (
    calculate_codebase_hash,
    get_environment_info,
    main,
    run_validations,
)


def test_calculate_codebase_hash() -> None:
    """Test that calculating codebase hash works and produces a valid SHA-256 signature."""
    root_dir = Path(__file__).resolve().parent.parent / "clintrials"
    h = calculate_codebase_hash(root_dir)
    assert isinstance(h, str)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_get_environment_info() -> None:
    """Test that retrieving environmental information returns correct keys."""
    info = get_environment_info()
    assert "OS" in info
    assert "Python Version" in info
    assert "NumPy Version" in info
    assert "Pandas Version" in info
    assert "SciPy Version" in info
    assert "Statsmodels Version" in info
    assert "fpdf2 Version" in info


def test_run_validations() -> None:
    """Test that the validation suite executes correctly and all tests PASS."""
    results = run_validations()
    assert len(results) > 0
    # Every individual validation scenario should pass
    for r in results:
        assert r["status"] == "PASS"


def test_cli_execution(tmp_path: Path) -> None:
    """Test full CLI main entrypoint to ensure it generates PDF successfully."""
    pdf_out = tmp_path / "gxp_report.pdf"
    assert not pdf_out.exists()

    # Call main with mocked sys.argv to specify the output path
    test_args = ["clintrials-validate", "--output", str(pdf_out)]
    with patch("sys.argv", test_args):
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 0

    assert pdf_out.exists()
    assert pdf_out.stat().st_size > 0
