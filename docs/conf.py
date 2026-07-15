import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "clintrials"
author = "Kristian Brock"
release = importlib.metadata.version("clintrials")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "nbsphinx",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

nbsphinx_execute = "never"


autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "tutorials/environment_test.ipynb",
    "tutorials/sequential_design_draft.ipynb",
]

html_theme = "furo"
html_static_path = ["_static"]
html_extra_path = ["_extra"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

# Warnings as errors when running in CI
if os.environ.get("SPHINX_STRICT", "0") == "1":
    nitpicky = True
    nitpick_ignore = []

def setup(app):
    """Register custom extensions and hooks with Sphinx."""
    def replace_python_blocks_with_testcode(app, docname, source):
        if docname == "README" or docname.endswith(".md") or (app.env.doc2path(docname) and app.env.doc2path(docname).endswith(".md")):
            import re
            def repl(match):
                code = match.group(1)
                return f"```{{testcode}}\n{code}```\n\n```{{testoutput}}\n:options: +ELLIPSIS\n\n...\n```"
            source[0] = re.sub(r"^```python\s*\n(.*?)^```\s*$", repl, source[0], flags=re.MULTILINE | re.DOTALL)

    app.connect('source-read', replace_python_blocks_with_testcode)


