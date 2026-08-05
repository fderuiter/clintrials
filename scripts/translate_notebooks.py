#!/usr/bin/env python3
"""Translate Jupyter notebooks to MDX files for clinical trial design tutorials."""

import json
import shutil
import sys
from pathlib import Path


def extract_title(notebook_data: dict, file_path: Path) -> str:
    """Extract a descriptive title from notebook metadata or first markdown headers.

    Checks metadata title first, then looks for standard h1/h2 headers.
    """
    # 1. Try metadata.title
    title = notebook_data.get("metadata", {}).get("title")
    if title:
        return str(title).strip()

    # 2. Extract '#' and '##' headers from markdown cells
    h1_header = None
    h2_header = None
    for cell in notebook_data.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", [])
            if isinstance(source, str):
                lines = source.splitlines()
            else:
                lines = source
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("# "):
                    if not h1_header:
                        h1_header = stripped.lstrip("#").strip()
                elif stripped.startswith("## "):
                    if not h2_header:
                        h2_header = stripped.lstrip("#").strip()

    if h1_header:
        if h2_header and (
            "matchpoint" in str(file_path).lower()
            or h1_header.lower().startswith("implementing the efftox")
        ):
            return f"{h1_header} - {h2_header}"
        return h1_header

    # 3. Try kernelspec display_name if it is not generic
    kernelspec = notebook_data.get("metadata", {}).get("kernelspec", {})
    display_name = kernelspec.get("display_name")
    if display_name and "python" not in display_name.lower():
        return str(display_name).strip()

    # 4. Fallback to stem
    return file_path.stem


def convert_notebook_to_mdx(input_path: Path, output_path: Path) -> bool:
    """Read a Jupyter notebook and compile its cells to an MDX document."""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {input_path}: {e}", file=sys.stderr)  # noqa: T201
        return False

    title = extract_title(data, input_path)

    # Format the exact frontmatter required by build_docs.js regex
    mdx_lines = ["---", f'title: "{title}"', "---", ""]

    cells = data.get("cells", [])
    for cell in cells:
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = source

        if cell_type == "markdown":
            mdx_lines.append(source_text)
            mdx_lines.append("")
        elif cell_type == "code":
            if source_text.strip():
                mdx_lines.append("```python")
                mdx_lines.append(source_text.rstrip())
                mdx_lines.append("```")
                mdx_lines.append("")

            # Handle outputs
            outputs = cell.get("outputs", [])
            output_blocks = []
            for out in outputs:
                out_type = out.get("output_type")
                if out_type == "stream":
                    text = out.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    output_blocks.append(text)
                elif out_type in ("execute_result", "display_data"):
                    out_data = out.get("data", {})
                    # Prefer text/plain
                    text_plain = out_data.get("text/plain", "")
                    if isinstance(text_plain, list):
                        text_plain = "".join(text_plain)
                    elif not isinstance(text_plain, str):
                        text_plain = str(text_plain)
                    if text_plain:
                        output_blocks.append(text_plain)
                elif out_type == "error":
                    tb = out.get("traceback", [])
                    if isinstance(tb, list):
                        tb = "\n".join(tb)
                    output_blocks.append(tb)

            if output_blocks:
                combined_output = "\n".join(output_blocks).rstrip()
                if combined_output:
                    mdx_lines.append("```text")
                    mdx_lines.append(combined_output)
                    mdx_lines.append("```")
                    mdx_lines.append("")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mdx_lines))
        print(f"Successfully converted {input_path} -> {output_path}")  # noqa: T201
        return True
    except Exception as e:
        print(f"Error writing to {output_path}: {e}", file=sys.stderr)  # noqa: T201
        return False


def main() -> None:
    """Find all Jupyter notebooks in tutorials/ and convert them to MDX."""
    root_dir = Path(__file__).resolve().parent.parent
    tutorials_src = root_dir / "docs" / "tutorials"
    tutorials_dest = root_dir / "docs" / "reference" / "tutorials"

    if not tutorials_src.exists():
        print(f"Tutorials directory not found at {tutorials_src}", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    if tutorials_dest.exists():
        shutil.rmtree(tutorials_dest)
    tutorials_dest.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0

    for ipynb_path in sorted(tutorials_src.rglob("*.ipynb")):
        if (
            ".ipynb_checkpoints" in ipynb_path.parts
            or ipynb_path.name.startswith("~")
            or ipynb_path.name.startswith(".")
        ):
            continue

        rel_path = ipynb_path.relative_to(tutorials_src)
        mdx_path = tutorials_dest / rel_path.with_suffix(".mdx")

        if convert_notebook_to_mdx(ipynb_path, mdx_path):
            success_count += 1
        else:
            failure_count += 1

    print(  # noqa: T201
        f"Finished conversion: {success_count} succeeded, {failure_count} failed."
    )
    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
