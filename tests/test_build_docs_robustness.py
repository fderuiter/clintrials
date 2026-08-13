# SPDX-License-Identifier: MIT

import subprocess
import shutil
from pathlib import Path

def test_build_docs_robust_parsing_and_rewriting() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    ref_dir = root_dir / "docs" / "reference"
    dist_dir = root_dir / "docs" / "dist"
    
    # 1. Create a test mdx file with extra frontmatter, varying whitespace, single quotes
    test_mdx_path = ref_dir / "tutorials" / "test_robust.mdx"
    test_mdx_content = """---
title:   'My Robust Tutorial'
author: "Doc Contributor"
date: 2026-08-13
---

# Robust Tutorial
Welcome. Here is a link to a directory: [Tutorials](..) and another [Reference Root](/reference/) and [Relative Dir](../tutorials/).
"""
    test_mdx_path.write_text(test_mdx_content, encoding="utf-8")
    
    try:
        # Run the build
        result = subprocess.run(
            ["node", "scripts/build_docs.js"],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check generated HTML
        test_html_path = dist_dir / "tutorials" / "test_robust.html"
        assert test_html_path.exists(), "Robust HTML was not generated."
        
        html_content = test_html_path.read_text(encoding="utf-8")
        
        # Assert Title is parsed cleanly (quotes and extra whitespace stripped)
        assert "<title>My Robust Tutorial | Clintrials Documentation</title>" in html_content
        
        # Assert no raw YAML or frontmatter tags are exposed
        assert "author:" not in html_content
        assert "date:" not in html_content
        assert "---" not in html_content
        
        # Assert path rewrites for directories and root links are correct
        assert 'href="../index.html"' in html_content
        assert 'href="/reference/index.html"' in html_content
        assert 'href="../tutorials/index.html"' in html_content
        
    finally:
        # Clean up
        if test_mdx_path.exists():
            test_mdx_path.unlink()
        test_html_path = dist_dir / "tutorials" / "test_robust.html"
        if test_html_path.exists():
            test_html_path.unlink()

def test_build_docs_syntax_highlight_fallback() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    ref_dir = root_dir / "docs" / "reference"
    dist_dir = root_dir / "docs" / "dist"
    
    # 1. Create a test mdx file with Python block
    test_mdx_path = ref_dir / "tutorials" / "test_highlight_fallback.mdx"
    test_mdx_content = """---
title: "Highlight Fallback Test"
---

```python
import clintrials
print("Hello")
```
"""
    test_mdx_path.write_text(test_mdx_content, encoding="utf-8")
    
    # Run the build with hljs mocked to throw an error
    node_test_script = """
const fs = require('fs');
const path = require('path');

// Mock highlight.js highlight method to throw an error
const hljs = require('highlight.js');
hljs.highlight = function() {
    throw new Error('Mocked Syntax Highlighting Error');
};

// Run build_docs.js
require('./scripts/build_docs.js');
"""
    node_test_path = root_dir / "test_build_wrapper.js"
    node_test_path.write_text(node_test_script, encoding="utf-8")
    
    try:
        result = subprocess.run(
            ["node", "test_build_wrapper.js"],
            cwd=str(root_dir),
            capture_output=True,
            text=True
        )
        
        # Check stderr or stdout for fallback warning
        combined_output = result.stdout + "\n" + result.stderr
        assert "[Warning] Syntax highlighting failed for python block in code block. Falling back to plain text." in combined_output
        
        # Check generated HTML has plain-text fallback content and isn't empty/erased
        test_html_path = dist_dir / "tutorials" / "test_highlight_fallback.html"
        assert test_html_path.exists()
        
        html_content = test_html_path.read_text(encoding="utf-8")
        assert "import clintrials" in html_content
        assert 'class="language-python"' in html_content
        
    finally:
        if test_mdx_path.exists():
            test_mdx_path.unlink()
        test_html_path = dist_dir / "tutorials" / "test_highlight_fallback.html"
        if test_html_path.exists():
            test_html_path.unlink()
        if node_test_path.exists():
            node_test_path.unlink()

def test_build_docs_missing_static_folder() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    static_dir = root_dir / "docs" / "_static"
    backup_static_dir = root_dir / "docs" / "_static_backup"
    
    # Move original static folder to backup to simulate a fresh checkout
    if static_dir.exists():
        static_dir.rename(backup_static_dir)
        
    try:
        # Run build
        result = subprocess.run(
            ["node", "scripts/build_docs.js"],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
            check=True
        )
        
        # Verify that docs/_static was created dynamically
        assert static_dir.exists(), "docs/_static folder was not created dynamically."
        
    finally:
        # Restore backup
        if backup_static_dir.exists():
            if static_dir.exists():
                shutil.rmtree(static_dir)
            backup_static_dir.rename(static_dir)
