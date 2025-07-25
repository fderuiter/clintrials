import os
import sys
import importlib

sys.path.insert(0, os.path.abspath('..'))

project = 'clintrials'
author = 'Kristian Brock'
release = importlib.import_module('clintrials').__version__ if hasattr(importlib.import_module('clintrials'), '__version__') else '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'myst_parser',
]

autosummary_generate = True
napoleon_google_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']

