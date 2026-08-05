import subprocess
from pathlib import Path


def test_static_syntax_highlighting_exists():
    """Verify that build-time static syntax highlighting is properly compiled and offline-ready."""
    root_dir = Path(__file__).resolve().parent.parent
    dist_dir = root_dir / "docs" / "dist"

    # 1. Compile the documentation if it is not already built
    if not dist_dir.exists() or not list(dist_dir.glob("**/*.html")):
        subprocess.run(["npm", "run", "prebuild"], cwd=str(root_dir), check=True)
        subprocess.run(["npm", "run", "build"], cwd=str(root_dir), check=True)

    assert dist_dir.exists(), "Docs dist directory was not created."

    # 2. Check that compiled HTML files contain static syntax highlighting token spans
    html_files = list(dist_dir.glob("**/*.html"))
    assert html_files, "No HTML files were compiled under docs/dist."

    python_block_count = 0
    hljs_span_count = 0

    for f in html_files:
        content = f.read_text(encoding="utf-8")

        # Ensure no remote external stylesheets/scripts are introduced for highlighting
        assert "cdnjs.cloudflare.com" not in content
        assert "unpkg.com" not in content
        assert "jsdelivr.net" not in content
        assert "highlight.js" not in content or "highlight.js" in f.name or "class=\"language-" in content or "scripts/build_docs.js" in content

        # Verify that static css highlight rules matching standard light theme are embedded
        if "index.html" not in f.name and "search.html" not in f.name:
            assert ".hljs-keyword" in content, f"Embedded CSS missing .hljs-keyword style in {f.name}"
            assert ".hljs-string" in content, f"Embedded CSS missing .hljs-string style in {f.name}"
            assert ".hljs-comment" in content, f"Embedded CSS missing .hljs-comment style in {f.name}"

        if "class=\"language-python\"" in content:
            python_block_count += 1
            if "hljs-keyword" in content or "hljs-string" in content or "hljs-number" in content:
                hljs_span_count += 1

    assert python_block_count > 0, "No Python code blocks (language-python) were found in compiled HTML files."
    assert hljs_span_count > 0, "No highlight.js token span classes (e.g. hljs-keyword) were found inside code blocks."
