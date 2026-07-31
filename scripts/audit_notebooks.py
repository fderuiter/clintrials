#!/usr/bin/env python3
"""Unified Notebook Audit and Standards CLI Tool.

Parses Jupyter notebooks to validate relative markdown links, detect private imports/references,
and compute public API coverage against a JSON manifest.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

MARKDOWN_LINK_RE = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')


def find_notebook_files(path: Union[str, Path]) -> List[Path]:
    """Find all valid .ipynb files recursively, excluding checkpoints and drafts."""
    p = Path(path)
    if p.is_file():
        if p.suffix == ".ipynb":
            return [p]
        return []
    elif p.is_dir():
        notebooks = []
        for sub_p in p.rglob("*.ipynb"):
            parts = sub_p.parts
            if any(part.startswith(".") or "checkpoint" in part.lower() or part.startswith("~") for part in parts):
                continue
            notebooks.append(sub_p)
        return sorted(notebooks)
    return []


def clean_ipython_magics(code_str: str) -> str:
    """Strip or comment out IPython magics/shell commands to prevent AST SyntaxErrors."""
    cleaned_lines = []
    for line in code_str.splitlines():
        stripped = line.strip()
        if stripped.startswith("!") or stripped.startswith("%"):
            cleaned_lines.append("")
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def resolve_node(node: Optional[ast.AST], bindings: Dict[str, str]) -> Optional[str]:
    """Statically resolve the full path name of a Name or Attribute AST node."""
    if isinstance(node, ast.Name):
        if node.id in bindings:
            return bindings[node.id]
        if node.id == "clintrials":
            return "clintrials"
        return None
    elif isinstance(node, ast.Attribute):
        val_path = resolve_node(node.value, bindings)
        if val_path:
            return f"{val_path}.{node.attr}"
    return None


class NotebookASTVisitor(ast.NodeVisitor):
    """AST visitor to find private API violations and collect referenced clintrials elements."""

    def __init__(self, manifest_elements: Set[str], bindings: Optional[Dict[str, str]] = None) -> None:
        """Initialize the visitor with manifest elements and bindings."""
        self.manifest_elements: Set[str] = manifest_elements
        self.bindings: Dict[str, str] = bindings if bindings is not None else {}
        self.referenced: Set[str] = set()
        self.private_violations: List[Tuple[int, str, str]] = []  # list of tuples (lineno, name, msg)

    def visit_Import(self, node: ast.Import) -> None:
        """Analyze imports to find private module imports."""
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            # Check for private imports
            if any(part.startswith("_") for part in name.split(".")):
                self.private_violations.append(
                    (node.lineno, name, f"Import of private module: {name}")
                )

            if name.startswith("clintrials"):
                self.bindings[asname] = name
                self.referenced.add(name)
                # Register all subparts
                parts = name.split(".")
                for i in range(1, len(parts) + 1):
                    sub_mod = ".".join(parts[:i])
                    self.referenced.add(sub_mod)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Analyze ImportFrom statements for private source or name imports."""
        module = node.module or ""
        # Check for private module in from import
        if any(part.startswith("_") for part in module.split(".")):
            self.private_violations.append(
                (node.lineno, module, f"Import from private module: {module}")
            )

        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            # Check for private name import (excluding dunder like __version__)
            if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
                self.private_violations.append(
                    (node.lineno, name, f"Import of private name: {name}")
                )

            if module.startswith("clintrials") or module == "clintrials":
                full_path = f"{module}.{name}"
                self.bindings[asname] = full_path
                self.referenced.add(module)
                self.referenced.add(full_path)
                # Register all subparts of module
                parts = module.split(".")
                for i in range(1, len(parts) + 1):
                    sub_mod = ".".join(parts[:i])
                    self.referenced.add(sub_mod)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Analyze assignments to track bindings of resolved paths."""
        # Visit the value first to resolve nested assignments and references
        self.visit(node.value)

        val_path = resolve_node(node.value, self.bindings)
        if isinstance(node.value, ast.Call):
            val_path = resolve_node(node.value.func, self.bindings)

        if val_path:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.bindings[target.id] = val_path
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.bindings[elt.id] = val_path

        # Visit targets
        for target in node.targets:
            self.visit(target)

    def visit_Call(self, node: ast.Call) -> None:
        """Analyze function calls to identify which API elements are referenced."""
        func_path = resolve_node(node.func, self.bindings)
        if func_path:
            self.referenced.add(func_path)
            init_path = f"{func_path}.__init__"
            if init_path in self.manifest_elements:
                self.referenced.add(init_path)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Track Name reference if it matches a bound clintrials element."""
        name = node.id
        path = self.bindings.get(name)
        if path:
            self.referenced.add(path)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Analyze attributes to detect private attribute access or track paths."""
        attr = node.attr
        val_path = resolve_node(node.value, self.bindings)

        if attr.startswith("_") and not (attr.startswith("__") and attr.endswith("__")):
            if val_path and val_path.startswith("clintrials"):
                self.private_violations.append(
                    (node.lineno, attr, f"Access to private attribute/method: {val_path}.{attr}")
                )

        path = resolve_node(node, self.bindings)
        if path:
            self.referenced.add(path)
            parts = path.split(".")
            for i in range(1, len(parts) + 1):
                sub = ".".join(parts[:i])
                if sub.startswith("clintrials"):
                    self.referenced.add(sub)
        self.generic_visit(node)


def load_public_elements(manifest_path: Path) -> Set[str]:
    """Load the set of all public API elements from the manifest JSON."""
    if not manifest_path.exists():
        return set()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    elements = set()
    for module_name, exports in manifest.items():
        elements.add(module_name)
        for export_name, export_info in exports.items():
            full_export_name = f"{module_name}.{export_name}"
            elements.add(full_export_name)
            if export_info.get("type") == "class":
                for method_name in export_info.get("methods", {}):
                    elements.add(f"{full_export_name}.{method_name}")
    return elements


def audit_single_notebook(notebook_path: Path, manifest_elements: Set[str]) -> Tuple[List[str], Set[str]]:
    """Audit a single notebook and return any issues found and the referenced API elements."""
    issues: List[str] = []
    referenced_elements: Set[str] = set()

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except Exception as e:
        issues.append(f"Failed to parse notebook JSON: {e}")
        return issues, referenced_elements

    cells = notebook.get("cells", [])
    bindings: Dict[str, str] = {}

    for cell_idx, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])
        source_str = "".join(source) if isinstance(source, list) else source

        if cell_type == "markdown":
            # Extract links and images
            lines = source_str.splitlines()
            for line_idx, line in enumerate(lines, start=1):
                for match in MARKDOWN_LINK_RE.finditer(line):
                    target = match.group(1).strip()
                    # Strip query parameters and anchors
                    clean_target = target.split("?")[0].split("#")[0]
                    if not clean_target:
                        continue

                    # Exclusion Rules
                    if "://" in clean_target or clean_target.startswith("//") or clean_target.startswith("mailto:"):
                        continue

                    # Validation relative to the notebook file directory
                    notebook_dir = notebook_path.parent
                    resolved_path = (notebook_dir / clean_target).resolve()

                    # Exclude checkpoints/drafts if we somehow matched them
                    if any(part.startswith(".") or "checkpoint" in part.lower() for part in resolved_path.parts):
                        continue

                    if not resolved_path.exists():
                        issues.append(
                            f"Broken link in Cell {cell_idx}, line {line_idx}: target file does not exist on disk: '{target}'"
                        )

        elif cell_type == "code":
            if not source_str.strip():
                continue

            # Filter IPython magics
            cleaned_code = clean_ipython_magics(source_str)
            try:
                tree = ast.parse(cleaned_code)
            except SyntaxError as se:
                issues.append(f"Syntax error in Cell {cell_idx}: {se}")
                continue

            visitor = NotebookASTVisitor(manifest_elements, bindings=bindings)
            visitor.visit(tree)

            # Record violations
            for lineno, name, msg in visitor.private_violations:
                # Find the offending line of code if possible
                code_lines = source_str.splitlines()
                offending_code = code_lines[lineno - 1].strip() if 0 <= lineno - 1 < len(code_lines) else ""
                issues.append(
                    f"Private API violation in Cell {cell_idx}, Line {lineno}: {msg} -> `{offending_code}`"
                )

            # Accumulate referenced elements
            referenced_elements.update(visitor.referenced)
            bindings = visitor.bindings

    return issues, referenced_elements


def main() -> None:
    """Audit Jupyter notebooks for quality standards."""
    parser = argparse.ArgumentParser(description="Audit Jupyter notebooks for quality standards.")
    parser.add_argument("path", help="Path to notebook file or directory of notebooks to audit")
    parser.add_argument("--manifest", default="api_manifest.json", help="Path to the public API JSON manifest")
    parser.add_argument("--min-coverage", type=float, default=0.0, help="Minimum acceptable coverage percentage")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest file '{manifest_path}' does not exist.")  # noqa: T201
        sys.exit(1)

    manifest_elements = load_public_elements(manifest_path)
    if not manifest_elements:
        print(f"Error: Loaded 0 public elements from '{manifest_path}'.")  # noqa: T201
        sys.exit(1)

    notebook_files = find_notebook_files(args.path)
    if not notebook_files:
        print(f"No notebooks found at path '{args.path}'.")  # noqa: T201
        sys.exit(0)

    total_issues_count = 0
    all_referenced_elements = set()

    print(f"Auditing {len(notebook_files)} notebooks against {len(manifest_elements)} public API elements...\n")  # noqa: T201

    for nb_path in notebook_files:
        relative_path = nb_path.relative_to(Path.cwd()) if nb_path.is_relative_to(Path.cwd()) else nb_path
        print(f"Auditing: {relative_path}")  # noqa: T201
        issues, referenced = audit_single_notebook(nb_path, manifest_elements)
        all_referenced_elements.update(referenced)

        if issues:
            total_issues_count += len(issues)
            for issue in issues:
                print(f"  [FAIL] {issue}")  # noqa: T201
        else:
            print("  [OK] No issues found.")  # noqa: T201

    # Calculate coverage
    covered_elements = manifest_elements.intersection(all_referenced_elements)
    coverage_pct = (len(covered_elements) / len(manifest_elements)) * 100 if manifest_elements else 100.0

    print("\n" + "="*40)  # noqa: T201
    print("Notebook Standards and API Coverage Report")  # noqa: T201
    print("="*40)  # noqa: T201
    print(f"Total issues found: {total_issues_count}")  # noqa: T201
    print(f"Public API elements in manifest: {len(manifest_elements)}")  # noqa: T201
    print(f"Unique public API elements referenced: {len(covered_elements)}")  # noqa: T201
    print(f"Computed API Coverage: {coverage_pct:.2f}%")  # noqa: T201

    # Check coverage threshold
    coverage_failed = False
    if coverage_pct < args.min_coverage:
        print(f"  [FAIL] Coverage {coverage_pct:.2f}% is below the required minimum of {args.min_coverage:.2f}%")  # noqa: T201
        coverage_failed = True
    else:
        print(f"  [OK] Coverage matches or exceeds target of {args.min_coverage:.2f}%")  # noqa: T201

    print("="*40)  # noqa: T201

    if total_issues_count > 0 or coverage_failed:
        print("\nAudit failed. Fix the issues or raise coverage to pass.")  # noqa: T201
        sys.exit(1)
    else:
        print("\nAudit passed successfully!")  # noqa: T201
        sys.exit(0)


if __name__ == "__main__":
    main()
