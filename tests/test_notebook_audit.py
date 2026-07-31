import ast
import json
from pathlib import Path

from scripts.audit_notebooks import (
    NotebookASTVisitor,
    audit_single_notebook,
    clean_ipython_magics,
    find_notebook_files,
    load_public_elements,
    resolve_node,
)


def test_find_notebook_files(tmp_path: Path) -> None:
    """Test finding notebook files in a directory."""
    # Create mock notebooks and checkpoint paths
    valid_nb = tmp_path / "valid.ipynb"
    valid_nb.touch()

    checkpoint_dir = tmp_path / ".ipynb_checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_nb = checkpoint_dir / "valid-checkpoint.ipynb"
    checkpoint_nb.touch()

    draft_nb = tmp_path / "~draft.ipynb"
    draft_nb.touch()

    other_file = tmp_path / "other.txt"
    other_file.touch()

    found = find_notebook_files(tmp_path)
    assert len(found) == 1
    assert found[0].name == "valid.ipynb"


def test_clean_ipython_magics() -> None:
    """Test cleaning IPython magics and shell commands."""
    code = (
        "!pip install clintrials\n"
        "import numpy as np\n"
        "%matplotlib inline\n"
        "x = 10"
    )
    cleaned = clean_ipython_magics(code)
    expected = (
        "\n"
        "import numpy as np\n"
        "\n"
        "x = 10"
    )
    assert cleaned == expected


def test_resolve_node() -> None:
    """Test resolving AST Name and Attribute nodes to full paths."""
    # Test Name resolve
    bindings = {"x": "clintrials.core.Protocol"}
    node_name = ast.Name(id="x", ctx=ast.Load())
    assert resolve_node(node_name, bindings) == "clintrials.core.Protocol"

    node_clintrials = ast.Name(id="clintrials", ctx=ast.Load())
    assert resolve_node(node_clintrials, bindings) == "clintrials"

    # Test Attribute resolve
    node_attr = ast.Attribute(
        value=ast.Name(id="clintrials", ctx=ast.Load()),
        attr="core",
        ctx=ast.Load(),
    )
    assert resolve_node(node_attr, bindings) == "clintrials.core"


def test_notebook_ast_visitor_private_imports() -> None:
    """Test NotebookASTVisitor private import and attribute check."""
    manifest_elements = {"clintrials.core.Protocol", "clintrials.core.Protocol.__init__"}

    # Test private module import
    tree1 = ast.parse("import clintrials._private")
    visitor1 = NotebookASTVisitor(manifest_elements)
    visitor1.visit(tree1)
    assert len(visitor1.private_violations) == 1
    assert "Import of private module: clintrials._private" in visitor1.private_violations[0][2]

    # Test private name import
    tree2 = ast.parse("from clintrials.core import _private_helper")
    visitor2 = NotebookASTVisitor(manifest_elements)
    visitor2.visit(tree2)
    assert len(visitor2.private_violations) == 1
    assert "Import of private name: _private_helper" in visitor2.private_violations[0][2]

    # Test private attribute access
    code3 = (
        "from clintrials.core import Protocol\n"
        "p = Protocol()\n"
        "p._private_method()"
    )
    tree3 = ast.parse(code3)
    visitor3 = NotebookASTVisitor(manifest_elements)
    visitor3.visit(tree3)
    assert len(visitor3.private_violations) == 1
    assert "Access to private attribute/method" in visitor3.private_violations[0][2]


def test_load_public_elements(tmp_path: Path) -> None:
    """Test loading public API elements from JSON manifest."""
    manifest_data = {
        "clintrials.core": {
            "Protocol": {
                "type": "class",
                "methods": {
                    "__init__": {"type": "method"},
                    "run": {"type": "method"}
                }
            }
        },
        "clintrials.math": {
            "logit": {
                "type": "function"
            }
        }
    }
    manifest_file = tmp_path / "test_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f)

    elements = load_public_elements(manifest_file)
    expected = {
        "clintrials.core",
        "clintrials.core.Protocol",
        "clintrials.core.Protocol.__init__",
        "clintrials.core.Protocol.run",
        "clintrials.math",
        "clintrials.math.logit"
    }
    assert elements == expected


def test_audit_single_notebook(tmp_path: Path) -> None:
    """Test auditing a single notebook file for links and violations."""
    # Create mock notebooks with various links and imports
    valid_referenced_file = tmp_path / "referenced.md"
    valid_referenced_file.touch()

    notebook_data = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Tutorial\n",
                    "This links to [good](referenced.md) and [bad](missing.md) and [external](https://google.com)\n"
                ]
            },
            {
                "cell_type": "code",
                "source": [
                    "from clintrials.core import Protocol\n",
                    "from clintrials.core import _private\n"
                ]
            }
        ]
    }

    notebook_file = tmp_path / "test.ipynb"
    with open(notebook_file, "w") as f:
        json.dump(notebook_data, f)

    manifest_elements = {"clintrials.core.Protocol"}
    issues, referenced = audit_single_notebook(notebook_file, manifest_elements)

    # Should flag the bad relative link
    assert any("Broken link" in issue and "missing.md" in issue for issue in issues)
    # Should not flag the good relative link or the external google.com link
    assert not any("referenced.md" in issue for issue in issues)
    assert not any("google.com" in issue for issue in issues)

    # Should flag the private import
    assert any("Private API violation" in issue and "_private" in issue for issue in issues)

    # Should track clintrials.core.Protocol
    assert "clintrials.core.Protocol" in referenced
