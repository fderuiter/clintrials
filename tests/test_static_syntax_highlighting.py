# SPDX-License-Identifier: MIT

import subprocess
from pathlib import Path


def test_static_syntax_highlighting_exists():
    """Verify that build-time static syntax highlighting is properly compiled and offline-ready."""
    root_dir = Path(__file__).resolve().parent.parent
    dist_dir = root_dir / "docs" / "_build" / "html"

    # 1. Compile the documentation if it is not already built
    if not dist_dir.exists() or not list(dist_dir.glob("**/*.html")):
        import shutil
        import pytest

        # Check for system-level dependencies and python-level extras required to build
        if not shutil.which("pandoc"):
            pytest.skip("Skipping because pandoc is not installed and documentation is not pre-built.")

        try:
            import sphinx  # type: ignore # noqa: F401
            import nbsphinx  # type: ignore # noqa: F401
            import furo  # type: ignore # noqa: F401
        except ImportError:
            pytest.skip("Skipping because doc-building extras (sphinx, nbsphinx, furo) are not installed.")

        subprocess.run(
            ["poetry", "run", "sphinx-build", "-b", "html", "docs", "docs/_build/html"],
            cwd=str(root_dir),
            check=True
        )

    assert dist_dir.exists(), "Docs build directory was not created."

    # 2. Check that compiled HTML files contain static syntax highlighting token spans
    html_files = list(dist_dir.glob("**/*.html"))
    assert html_files, "No HTML files were compiled under docs/_build/html."

    python_block_count = 0
    highlight_span_count = 0

    for f in html_files:
        content = f.read_text(encoding="utf-8")

        # Ensure no remote external highlight.js stylesheets/scripts are introduced for highlighting
        assert "cdnjs.cloudflare.com/ajax/libs/highlight.js" not in content

        # Check for python code blocks and pygments span tags
        if 'class="highlight-python"' in content or 'class="highlight"' in content or 'class="language-python"' in content:
            python_block_count += 1
            if (
                'class="k"' in content
                or 'class="kn"' in content
                or 'class="n"' in content
                or 'class="mi"' in content
            ):
                highlight_span_count += 1

    assert python_block_count > 0, (
        "No Python code blocks were found in compiled HTML files."
    )
    assert highlight_span_count > 0, (
        "No Pygments token span classes (e.g. class=\"k\") were found inside code blocks."
    )
