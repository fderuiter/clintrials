import re
from pathlib import Path


def test_documentation_internal_paths() -> None:
    root = Path(__file__).parent.parent

    # Gather all documentation files (.md and .rst)
    doc_files = []
    for ext in ("*.md", "*.rst"):
        for f in root.rglob(ext):
            # Skip virtualenvs and hidden directories
            if any(part.startswith('.') for part in f.parts):
                continue
            doc_files.append(f)

    broken_paths = []

    # Matches markdown links: [text](path)
    md_link_regex = re.compile(r'\[[^\]]+\]\(([^)]+)\)')

    # Matches inline backticks that contain a path
    # We look for something that contains a slash and an extension, e.g. `docs/index.rst`
    inline_path_regex = re.compile(r'`([a-zA-Z0-9_\-\./]+\.[a-zA-Z0-9]+)`')

    for f in set(doc_files):
        content = f.read_text(errors='ignore')

        # Check markdown links
        if f.suffix == '.md':
            for match in md_link_regex.finditer(content):
                link = match.group(1)
                if not link.startswith('http') and not link.startswith('#') and not link.startswith('mailto:'):
                    # Clean anchor tags from path
                    link = link.split('#')[0]
                    if link:
                        target = (f.parent / link).resolve()
                        if not target.exists():
                            broken_paths.append(f"{f.relative_to(root)}: {link}")

        # Check inline backticks (both MD and RST)
        for match in inline_path_regex.finditer(content):
            path_str = match.group(1)
            if '/' in path_str and ' ' not in path_str and not path_str.startswith('<'):
                target = root / path_str
                if not target.exists():
                    broken_paths.append(f"{f.relative_to(root)}: {path_str}")

    assert not broken_paths, f"Found broken internal paths in documentation: {broken_paths}"

