import re
from pathlib import Path


def is_external_or_anchor(link: str) -> bool:
    """Check if the link points to an external URL or is just an anchor."""
    return (
        link.startswith('http://') or
        link.startswith('https://') or
        link.startswith('mailto:') or
        link.startswith('#') or
        '://' in link
    )


def extract_markdown_links(content: str) -> list[str]:
    """
    Extracts paths from markdown links: [text](path)
    Handles nested parentheses within the path correctly.
    """
    links = []
    i = 0
    n = len(content)
    while i < n:
        if content[i] == '[':
            # Find matching ']' by scanning forward
            j = i + 1
            bracket_depth = 1
            found_bracket_close = False
            while j < n:
                if content[j] == '[':
                    bracket_depth += 1
                elif content[j] == ']':
                    bracket_depth -= 1
                    if bracket_depth == 0:
                        found_bracket_close = True
                        break
                j += 1

            if found_bracket_close and j + 1 < n and content[j + 1] == '(':
                # Scan the parenthesis block
                k = j + 2
                paren_depth = 1
                path_chars = []
                found_paren_close = False
                while k < n:
                    if content[k] == '(':
                        paren_depth += 1
                        path_chars.append('(')
                    elif content[k] == ')':
                        paren_depth -= 1
                        if paren_depth == 0:
                            found_paren_close = True
                            break
                        path_chars.append(')')
                    else:
                        path_chars.append(content[k])
                    k += 1

                if found_paren_close:
                    path_str = "".join(path_chars).strip()
                    if path_str:
                        links.append(path_str)
                i = k
            else:
                i += 1
        else:
            i += 1
    return links


def is_file_path(path_str: str, base_dir: Path) -> bool:
    """
    Determine if a string looks like a file path rather than an internal RST label.
    """
    if path_str.startswith('/') or '/' in path_str:
        return True
    path = Path(path_str)
    if path.suffix:
        return True
    if (base_dir / path_str).exists():
        return True
    if (base_dir / (path_str + '.rst')).exists():
        return True
    if (base_dir / (path_str + '.md')).exists():
        return True
    return False


def resolve_and_verify(path_str: str, f: Path, root: Path, is_doc_role: bool = False) -> bool:
    """
    Resolve link relative to the root or current file and verify existence.
    """
    clean_path = path_str.split('#')[0]
    if not clean_path:
        return True

    candidates = []
    if clean_path.startswith('/'):
        # Root-relative path
        candidates.append(root / clean_path.lstrip('/'))
    else:
        # Relative path (relative to the file's parent or repository root)
        candidates.append(f.parent / clean_path)
        candidates.append(root / clean_path)

    for cand in candidates:
        resolved = cand.resolve()
        if resolved.exists():
            return True

        # If the path has no suffix, or if we are explicitly checking a doc reference,
        # check with standard documentation extensions (.rst and .md).
        if not resolved.suffix or is_doc_role:
            if resolved.with_suffix('.rst').exists():
                return True
            if resolved.with_suffix('.md').exists():
                return True

    return False


def validate_docs(root: Path) -> list[str]:
    # Gather all documentation files (.md and .rst)
    doc_files = []
    for ext in ("*.md", "*.rst"):
        for f in root.rglob(ext):
            # Skip virtualenvs, node_modules and hidden directories
            if any(part.startswith('.') or part == 'node_modules' for part in f.parts):
                continue
            doc_files.append(f)

    broken_paths = []

    # Matches inline backticks that contain a path
    # We look for something that contains a slash and an extension, e.g. `docs/index.rst`
    inline_path_regex = re.compile(r'`([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)`')

    # Matches explicit hyperlink targets: .. _target: path or .. _`target`: path
    rst_target_regex = re.compile(r'^\s*\.\.\s+_(?:`([^`]+)`|([^:]+)):\s*(.+)$', re.M)

    # Matches anonymous hyperlink targets: .. __: path or __ path
    rst_anonymous_regex = re.compile(r'^\s*(?:\.\.\s+__:|__)\s+(.+)$', re.M)

    # Matches Sphinx roles (e.g. :doc:`getting_started`, :download:`file.zip`)
    rst_role_regex = re.compile(r':(doc|download):\`(?:[^`<]*<)?([^`>]+)>?\`')

    # Matches standard RST directives (e.g. .. image:: path)
    rst_directive_regex = re.compile(r'^\s*\.\.\s+(image|figure|include|literalinclude)::\s*(.+)$', re.M)

    for f in set(doc_files):
        content = f.read_text(errors='ignore')

        # Check markdown links
        if f.suffix == '.md':
            extracted_links = extract_markdown_links(content)
            for link in extracted_links:
                if not is_external_or_anchor(link):
                    if not resolve_and_verify(link, f, root):
                        broken_paths.append(f"{f.relative_to(root)}: {link}")

        # Check RST-specific links and targets
        elif f.suffix == '.rst':
            # 1. Explicit hyperlink targets
            for match in rst_target_regex.finditer(content):
                target_path = match.group(3).strip()
                if target_path and not is_external_or_anchor(target_path):
                    if is_file_path(target_path, f.parent):
                        if not resolve_and_verify(target_path, f, root):
                            broken_paths.append(f"{f.relative_to(root)}: {target_path}")

            # 2. Anonymous hyperlink targets
            for match in rst_anonymous_regex.finditer(content):
                target_path = match.group(1).strip()
                if target_path and not is_external_or_anchor(target_path):
                    if is_file_path(target_path, f.parent):
                        if not resolve_and_verify(target_path, f, root):
                            broken_paths.append(f"{f.relative_to(root)}: {target_path}")

            # 3. Path-based roles
            for match in rst_role_regex.finditer(content):
                role_name = match.group(1)
                target_path = match.group(2).strip()
                if target_path and not is_external_or_anchor(target_path):
                    is_doc = (role_name == 'doc')
                    if not resolve_and_verify(target_path, f, root, is_doc_role=is_doc):
                        broken_paths.append(f"{f.relative_to(root)}: {target_path}")

            # 4. Directives
            for match in rst_directive_regex.finditer(content):
                target_path = match.group(2).strip()
                if target_path and not is_external_or_anchor(target_path):
                    if not resolve_and_verify(target_path, f, root):
                        broken_paths.append(f"{f.relative_to(root)}: {target_path}")

        # Check inline backticks (both MD and RST)
        for match in inline_path_regex.finditer(content):
            path_str = match.group(1)
            if '/' in path_str and ' ' not in path_str and not path_str.startswith('<'):
                if not resolve_and_verify(path_str, f, root):
                    broken_paths.append(f"{f.relative_to(root)}: {path_str}")

    return sorted(list(set(broken_paths)))


def test_documentation_internal_paths() -> None:
    root = Path(__file__).parent.parent
    broken_paths = validate_docs(root)
    assert not broken_paths, f"Found broken internal paths in documentation: {broken_paths}"


def test_is_external_or_anchor() -> None:
    assert is_external_or_anchor('http://example.com')
    assert is_external_or_anchor('https://example.com/foo')
    assert is_external_or_anchor('mailto:test@example.com')
    assert is_external_or_anchor('#some-anchor')
    assert is_external_or_anchor('ftp://files.org')
    assert not is_external_or_anchor('docs/getting_started.rst')
    assert not is_external_or_anchor('/docs/index.rst')


def test_extract_markdown_links() -> None:
    content = (
        "Check out [this link](docs/intro.md), and also [parenthesized link]"
        "(docs/tutorials/EffTox - Nuts and Bolts (Simplified).ipynb) and "
        "another [nested (parens) link](foo/bar(baz(qux))pkg.md)."
    )
    extracted = extract_markdown_links(content)
    assert extracted == [
        "docs/intro.md",
        "docs/tutorials/EffTox - Nuts and Bolts (Simplified).ipynb",
        "foo/bar(baz(qux))pkg.md"
    ]


def test_is_file_path(tmp_path: Path) -> None:
    # 1. Starts with /
    assert is_file_path('/docs/getting_started', tmp_path)
    # 2. Contains /
    assert is_file_path('docs/getting_started', tmp_path)
    # 3. Has suffix
    assert is_file_path('file.txt', tmp_path)
    # 4. Exists in base_dir
    (tmp_path / 'existing_file').touch()
    assert is_file_path('existing_file', tmp_path)
    # 5. Exists as .rst or .md
    (tmp_path / 'doc.rst').touch()
    assert is_file_path('doc', tmp_path)
    # 6. Not a file path
    assert not is_file_path('some-label', tmp_path)


def test_resolve_and_verify_root_relative(tmp_path: Path) -> None:
    root = tmp_path
    f = root / "docs" / "index.rst"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch()

    # Create target
    target_dir = root / "tutorials"
    target_dir.mkdir(exist_ok=True)
    target_file = target_dir / "tutorial.md"
    target_file.touch()

    # Root-relative
    assert resolve_and_verify('/tutorials/tutorial.md', f, root)
    # Relative
    assert resolve_and_verify('../tutorials/tutorial.md', f, root)
    # Missing extension for doc role
    assert resolve_and_verify('/tutorials/tutorial', f, root, is_doc_role=True)
    # Nonexistent
    assert not resolve_and_verify('/tutorials/nonexistent.md', f, root)


def test_validate_docs_scenarios(tmp_path: Path) -> None:
    root = tmp_path

    # Create target files
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    getting_started = docs_dir / "getting_started.rst"
    getting_started.touch()

    index_md = docs_dir / "index.md"
    index_md.touch()

    tutorials_dir = docs_dir / "tutorials"
    tutorials_dir.mkdir(exist_ok=True)
    tutorial_with_parens = tutorials_dir / "EffTox - Nuts and Bolts (Simplified).ipynb"
    tutorial_with_parens.touch()

    # Create an MD file with valid and invalid links
    md_test_file = docs_dir / "test.md"
    md_content = """
# Test Markdown
Here is a root-relative link: [Getting Started](/docs/getting_started.rst)
Here is a parenthesized link: [Tutorial](/docs/tutorials/EffTox - Nuts and Bolts (Simplified).ipynb)
Here is an external link: [Google](https://google.com)
Here is an anchor link: [Section](#section)
Here is a broken root-relative link: [Broken](/docs/does-not-exist.rst)
Here is a broken parenthesized link: [Broken Parens](/docs/tutorials/EffTox - (Broken).ipynb)
    """
    md_test_file.write_text(md_content)

    # Create an RST file with valid and invalid links/roles
    rst_test_file = docs_dir / "test.rst"
    rst_content = """
Test RST
========

Explicit hyperlink target:
.. _valid-target: /docs/getting_started.rst
.. _broken-target: /docs/nonexistent.rst
.. _label-only-target:

Role tests:
- Valid doc role: :doc:`getting_started`
- Valid download role: :download:`/docs/index.md`
- Broken doc role: :doc:`nonexistent`
- Broken download role: :download:`/docs/nonexistent.zip`

Directive tests:
- Valid image directive:
  .. image:: /docs/index.md
- Broken image directive:
  .. image:: /docs/nonexistent.png
    """
    rst_test_file.write_text(rst_content)

    broken = validate_docs(root)
    broken_str = "\n".join(broken)

    assert "docs/does-not-exist.rst" in broken_str
    assert "EffTox - (Broken).ipynb" in broken_str
    assert "nonexistent.rst" in broken_str
    assert "nonexistent" in broken_str
    assert "nonexistent.zip" in broken_str
    assert "nonexistent.png" in broken_str

    # Verify that valid links do NOT show up in broken
    assert "getting_started.rst" not in broken_str
    assert "EffTox - Nuts and Bolts (Simplified).ipynb" not in broken_str
    assert "https://google.com" not in broken_str
    assert "label-only-target" not in broken_str
