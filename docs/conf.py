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
    "myst_parser",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable", {}),
    "scipy": ("https://docs.scipy.org/doc/scipy", {}),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", {}),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

# Warnings as errors when running in CI
if os.environ.get("SPHINX_STRICT", "0") == "1":
    nitpicky = True
    nitpick_ignore = []
