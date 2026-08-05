"""Unit and integration tests for translating Jupyter notebooks and integrating them in the portal UI."""

import json
import subprocess
from pathlib import Path

from scripts.translate_notebooks import extract_title


def test_extract_title_logic(tmp_path: Path) -> None:
    """Test various scenarios of extracting a descriptive title from Jupyter notebooks."""
    # Test metadata title
    nb_data = {
        "metadata": {"title": "My Metadata Title"},
        "cells": []
    }
    assert extract_title(nb_data, tmp_path / "test.ipynb") == "My Metadata Title"

    # Test markdown h1 header
    nb_data_md = {
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# My Header H1\n", "Some text"]
            }
        ]
    }
    assert extract_title(nb_data_md, tmp_path / "test.ipynb") == "My Header H1"

    # Test matchpoint h2 combination
    nb_data_matchpoint = {
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Implementing the EffTox Dose-Finding Design in the Matchpoint Trials\n"]
            },
            {
                "cell_type": "markdown",
                "source": ["## Posterior Utility\n"]
            }
        ]
    }
    assert extract_title(nb_data_matchpoint, tmp_path / "matchpoint" / "Utility.ipynb") == "Implementing the EffTox Dose-Finding Design in the Matchpoint Trials - Posterior Utility"


def test_translation_script_and_build(tmp_path: Path) -> None:
    """Run full translation and HTML build integration tests to verify the landing page and search indexing."""
    root_dir = Path(__file__).resolve().parent.parent
    scripts_dir = root_dir / "scripts"

    # Run the translator
    res_translate = subprocess.run(
        ["poetry", "run", "python", str(scripts_dir / "translate_notebooks.py")],
        capture_output=True,
        text=True
    )
    assert res_translate.returncode == 0

    # Assert converted files exist
    tutorials_dest = root_dir / "docs" / "reference" / "tutorials"
    assert tutorials_dest.exists()

    # Let's check CRM.mdx has frontmatter
    crm_mdx_path = tutorials_dest / "CRM.mdx"
    assert crm_mdx_path.exists()
    content = crm_mdx_path.read_text(encoding="utf-8")
    assert content.startswith("---")
    assert 'title: "Using the CRM class in clintrials"' in content

    # Run build_docs.js
    res_build = subprocess.run(
        ["node", str(scripts_dir / "build_docs.js")],
        capture_output=True,
        text=True
    )
    assert res_build.returncode == 0

    # Check compiled HTML files
    dist_dir = root_dir / "docs" / "dist"
    assert (dist_dir / "tutorials" / "CRM.html").exists()
    assert (dist_dir / "tutorials" / "matchpoint" / "Utility.html").exists()

    # Check index.html landing page has tutorials section
    index_html = (dist_dir / "index.html").read_text(encoding="utf-8")
    assert "Tutorials & Onboarding Guides" in index_html
    assert "Using the CRM class in clintrials" in index_html
    assert "Implementing the EffTox Dose-Finding Design in the Matchpoint Trials - Posterior Utility" in index_html

    # Check search index contains the tutorials
    search_index_path = dist_dir / "search_index.json"
    assert search_index_path.exists()
    with open(search_index_path, "r", encoding="utf-8") as f:
        search_data = json.load(f)

    tutorial_urls = [p["url"] for p in search_data if "tutorials" in p["url"]]
    assert len(tutorial_urls) >= 8
