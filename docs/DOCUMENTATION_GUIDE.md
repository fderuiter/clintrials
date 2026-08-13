# Robust Documentation Guidelines & Contributor Guide

This dedicated guide details how to build, preview, and verify the documentation for **Clintrials** on your local machine. We support a **dual-build documentation setup** designed to meet diverse development and publication needs.

---

## 1. Dual-Build Architecture Overview

Our repository maintains two independent documentation pipelines:

| Feature | 1. Sphinx Pipeline | 2. Custom Node Pipeline |
| :--- | :--- | :--- |
| **Source Format** | reStructuredText (`.rst`), Sphinx Sphinx config | MDX / Markdown (`.mdx`, `.md`) |
| **Scope** | Traditional reference manual & developer docs | Modern, fast, and fully searchable API reference + notebook tutorials |
| **Output Location** | `docs/_build/html` | `docs/dist/` |
| **Search Engine** | Pagefind static search | Fast local JSON-indexed search |

---

## 2. Sphinx Documentation Pipeline

The Sphinx pipeline processes `.rst` files and generates the traditional reference documentation.

### Setup Requirements
1. **Python**: Ensure you have Python 3.9+ (excluding 3.9.7) installed.
2. **Poetry**: Make sure Poetry is installed and configured.

### Commands

To install dependencies and compile the Sphinx documentation, run the following:

```bash
# 1. Install all dependencies including documentation and visualization extras
poetry install --all-extras

# 2. Navigate to the documentation directory
cd docs

# 3. Clean any existing build artifacts
poetry run make clean

# 4. Compile the HTML files
poetry run make html
```

Once the compilation completes, open docs/_build/html/index.html in your browser to view the documentation portal.

---

## 3. Custom Node API Documentation Pipeline

The Node pipeline translates Jupyter notebooks into MDX, generates API manifests from code docstrings, and compiles these files into a fast, searchable static site.

### Setup Requirements
1. **Node.js & NPM**: Ensure Node.js (v18+) is installed on your machine.
2. **Python Environment**: The compiler's prebuild phase imports the `clintrials` package to extract docstrings. Therefore, the Python virtual environment must be active or accessible via Poetry.

### Commands

```bash
# 1. Ensure Python dependencies are installed and the current package is built
poetry install --all-extras

# 2. Install Node dependencies in the repository root
npm install

# 3. Run the prebuild script (translates tutorials and generates MDX reference files)
npm run prebuild

# 4. Build the static site (compiles MDX files and writes to docs/dist/)
npm run build
```

The resulting HTML site will be written to `docs/dist/`. To preview, open `docs/dist/index.html` in any web browser.

---

## 4. Automated Build Validation & Quality Checks

Our custom Node compiler features built-in verification checks that audit the documentation during build execution. These checks output directly to your terminal.

### What is validated?
1. **Frontmatter Integrity**:
   - Matches blocks enclosed by `---` bounds:
     ```yaml
     ---
     title: "My Custom Guide"
     author: "Contributor Name"
     ---
     ```
   - Checks for the presence of a `title` key. If a frontmatter block or title is missing or unparseable, a warning is printed.
2. **Link Verification**:
   - Relative directory links (e.g. `../tutorials/`, `.`, `..`) and root reference links (e.g. `/reference/`) are verified on disk relative to the source directory and resolved to functional index targets (e.g. `index.html`) rather than appending `.html` directly to folders.
   - Any broken internal pathways will trigger warnings.
3. **Syntax Highlighting Robustness**:
   - Python code blocks are compiled using `highlight.js`.
   - If highlight compilation fails, the block gracefully falls back to plain-text rendering without erasing any code.

### Warning Message Reference

Keep an eye out for these warnings in your build terminal:
- `[Warning] Missing frontmatter block in <file>`
- `[Warning] Unparseable frontmatter block in <file>`
- `[Warning] Missing or unparseable title in frontmatter of <file>`
- `[Warning] Broken relative link found in <file>: "<link>"`
- `[Warning] Syntax highlighting failed for python block in code block. Falling back to plain text.`

---

## 5. Local Quality Assurance and Tests

Before submitting a pull request, please run our automated test suite to ensure that your modifications do not break pathing or rendering logic:

```bash
# Run documentation link checking
poetry run pytest tests/test_docs_links.py

# Run compilation robustness verification
poetry run pytest tests/test_build_docs_robustness.py

# Run static syntax highlighting verification
poetry run pytest tests/test_static_syntax_highlighting.py

# Run braces preservation verification
poetry run pytest tests/test_docs_braces.py
```

By verifying these locally, you can guarantee a 100% green build on our automated CI pipeline and speed up the review of your pull request!
