"""Sphinx configuration for mononet documentation."""

from __future__ import annotations

import os
from pathlib import Path

# -- Project information ---------------------------------------------------
project = "mononet"
author = "Davor Runje"
copyright = "2026, Davor Runje"
html_baseurl = "https://davorrunje.github.io/mononet/"

# -- Source layout ---------------------------------------------------------
# Temporary: source files still live in docs/docs/ (existing MkDocs layout).
# After Task 7, content moves up one level and this file becomes the source
# directory itself.
_HERE = Path(__file__).resolve().parent
_SOURCE_SUFFIX_DIR = _HERE / "docs"
if _SOURCE_SUFFIX_DIR.is_dir():
    # Phase 1-2: point at docs/docs/
    master_doc = "index"

# -- General configuration -------------------------------------------------
extensions = [
    "myst_nb",
    "autodoc2",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
]
exclude_patterns = ["_build", "site", "navigation_template.txt", "SUMMARY.md"]
templates_path = ["_templates"]
source_suffix = {
    ".md": "markdown",
    ".ipynb": "myst-nb",
}

# -- HTML output -----------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "mononet"
html_theme_options = {
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "github_url": "https://github.com/davorrunje/mononet",
    "use_edit_page_button": True,
    "navbar_align": "left",
    "show_version_warning_banner": True,
    "switcher": {
        "json_url": "https://davorrunje.github.io/mononet/versions.json",
        "version_match": os.environ.get("DOCS_VERSION", "latest"),
    },
    "show_toc_level": 2,
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}
html_context = {
    "github_user": "davorrunje",
    "github_repo": "mononet",
    "github_version": "main",
    "doc_path": "docs",
}
html_css_files = ["extra.css"]
html_js_files = ["extra.js"]

# -- MyST configuration ----------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "fieldlist",
    "dollarmath",
    "amsmath",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3

# -- MyST-NB (notebooks) ---------------------------------------------------
nb_execution_mode = "off"

# -- sphinx-autodoc2 -------------------------------------------------------
autodoc2_packages = [
    {"path": "../mononet", "auto_mode": True},
]
autodoc2_render_plugin = "myst"
autodoc2_docstring_parser_regexes = [
    (r".*", "google"),
]
autodoc2_hidden_objects = ["private", "dunder"]
autodoc2_index_template = None  # let Sphinx handle the index via toctree

# -- intersphinx -----------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "jax": ("https://docs.jax.dev", None),
    "keras": ("https://keras.io/api", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
intersphinx_disabled_reftypes = ["std:doc"]

# -- sphinx-multiversion ---------------------------------------------------
smv_branch_whitelist = r"^main$"
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_released_pattern = r"^refs/tags/v\d+\.\d+\.\d+$"
smv_remote_whitelist = r"^origin$"
smv_outputdir_format = "{ref.name}"
