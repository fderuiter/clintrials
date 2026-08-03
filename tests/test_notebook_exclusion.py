import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

has_docs_dependencies = importlib.util.find_spec("sphinx") is not None and shutil.which("pandoc") is not None

if not has_docs_dependencies:
    pytest.skip(
        "Optional docs dependencies or pandoc are not installed, skipping integration tests.",
        allow_module_level=True,
    )


def test_notebook_exclusion_during_compilation() -> None:
    root_dir = Path(__file__).parent.parent
    tutorials_dir = root_dir / "docs" / "tutorials"
    build_dir = root_dir / "docs" / "_build"

    # Define the dummy draft and test notebooks we want to create
    draft_files = [
        tutorials_dir / "CRM_draft.ipynb",
        tutorials_dir / "EffTox_test.ipynb",
        tutorials_dir / "temp_WATU.ipynb",
    ]

    # Minimal valid Jupyter Notebook JSON with a broken Sphinx/RST style reference
    # to ensure strict mode compilation won't fail because they are excluded.
    dummy_notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Draft Tutorial Test\n",
                    "\n",
                    "This is a draft/test tutorial notebook. It has a broken reference :ref:`non-existent-reference-99999` which would fail in strict mode if not excluded.",
                ],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Ensure clean state before starting
    for file_path in draft_files:
        if file_path.exists():
            file_path.unlink()

    # Clear previous build so we test a fresh compilation
    if build_dir.exists():
        shutil.rmtree(build_dir)

    try:
        # Create the temporary files
        for file_path in draft_files:
            file_path.write_text(json.dumps(dummy_notebook_content), encoding="utf-8")

        # Run sphinx build with strict mode enabled (-W is warnings as errors)
        # We run it via 'poetry run' to ensure the exact environment is used
        result = subprocess.run(
            [
                "poetry",
                "run",
                "sphinx-build",
                "-W",
                "-b",
                "html",
                "docs",
                "docs/_build/html",
            ],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
        )

        # Check that the build succeeded
        assert result.returncode == 0, (
            f"Sphinx compilation failed in strict mode:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # Assert no HTML output files corresponding to draft/test notebooks exist
        html_dir = build_dir / "html"
        assert html_dir.exists(), "HTML build directory was not created!"

        for draft_file in draft_files:
            html_name = f"{draft_file.stem}.html"
            compiled_html_path = html_dir / "tutorials" / html_name
            assert not compiled_html_path.exists(), (
                f"Draft notebook {draft_file.name} was compiled to {compiled_html_path}!"
            )

            # Also check pagefind index specifically to be sure it does not contain the stem
            pagefind_dir = html_dir / "pagefind"
            if pagefind_dir.exists():
                # Search for any reference to the draft names inside pagefind files
                # Pagefind index files are binary or JSON, but we can search in files recursively
                for f in pagefind_dir.rglob("*"):
                    if f.is_file():
                        content = f.read_bytes()
                        assert draft_file.stem.encode("utf-8") not in content, (
                            f"Pagefind index contains references to {draft_file.stem} in {f}!"
                        )

    finally:
        # Clean up temporary files created for the test
        for file_path in draft_files:
            if file_path.exists():
                file_path.unlink()
