"""Sphinx configuration for the saltjax documentation."""
import datetime
import os
import sys

# Document the local source tree (on readthedocs the package is also pip-installed).
sys.path.insert(0, os.path.abspath('../src'))

import saltjax # noqa: E402

# -- Project information -----------------------------------------------------

project = "saltjax"
author = "Mickael Rigault"
copyright = f"2026-{datetime.date.today().year}, Mickael Rigault"
release = saltjax.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_nb',
    'sphinx_copybutton',
    ]

# Notebooks are stored executed: they are never run at build time.
nb_execution_mode = "off"

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# -- API documentation -------------------------------------------------------

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True
napoleon_use_rtype = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
    'sncosmo': ('https://sncosmo.readthedocs.io/en/stable/', None),
    'skysurvey': ('https://skysurvey.readthedocs.io/en/latest/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

source_suffix = {'.rst': 'restructuredtext',
                 '.ipynb': 'myst-nb',
                 '.md': 'myst-nb'}

# -- Options for HTML output -------------------------------------------------

html_title = f"saltjax {version}"
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    'show_toc_level': 2,
    'home_page_in_toc': True,
    'navigation_with_keys': False,
    'repository_url': 'https://github.com/MickaelRigault/saltjax',
    'repository_branch': 'main',
    'path_to_docs': 'docs',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_edit_page_button': True,
    'use_download_button': True,
}
