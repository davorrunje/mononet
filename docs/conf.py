# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration for mononet documentation."""

from __future__ import annotations

import os

# -- Project information ---------------------------------------------------
project = "mononet"
author = "Davor Runje"
copyright = "2026, Davor Runje"
html_baseurl = "https://davorrunje.github.io/mononet/"

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
exclude_patterns = [
    "_build",
    "superpowers",
    "references",
]
templates_path = ["_templates"]
source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- HTML output -----------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_favicon = "_static/favicon.png"
html_title = "mononet"
html_theme_options = {
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
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
    # srcdir is docs/; resolves to <repo>/mononet.
    {"path": "../mononet", "auto_mode": True},
]
autodoc2_render_plugin = "myst"
autodoc2_docstring_parser_regexes = [
    (r".*", "myst"),
]
autodoc2_hidden_objects = ["private", "dunder"]
autodoc2_index_template = None  # let Sphinx handle the index via toctree

# -- intersphinx -----------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "flax": ("https://flax.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    # keras.io is not a Sphinx site (no objects.inv); keras.* refs are
    # nitpick-ignored below rather than resolved.
}
intersphinx_disabled_reftypes = ["std:doc"]

# -- nitpicky cross-reference checking -------------------------------------
nitpicky = True
nitpick_ignore_regex = [
    # No intersphinx inventory exists for these external namespaces, so their
    # cross-refs cannot be resolved. Ignore by NAMESPACE (not exact target) so a
    # dependency version bump cannot turn the strict -W docs gate red.
    ("py:.*", r"typing_extensions\..*"),  # no published objects.inv
    ("py:.*", r"keras\..*"),  # keras.io is not a Sphinx site
    # torch inventory does not publish these private/internal symbols (leading
    # underscore or bare TypeVars leaked into public signatures by autodoc2):
    ("py:.*", r"torch\.nn\.modules\.module\.(_grad_t|T)"),
    ("py:.*", r"torch\._prims_common\..*"),
    # torch's nn.Module.{cuda,ipu,xpu,mtia} accept a `device` parameter typed
    # `torch.device`; autodoc2 resolves the bare `device` annotation relative to
    # the enclosing method instead of qualifying it globally. Known autodoc2
    # limitation with inherited torch.nn.Module methods, not a mononet docstring.
    ("py:.*", r"torch\.nn\.modules\.module\.Module\.\w+\.device"),
    # Not published in torch's objects.inv despite being a real public class.
    ("py:.*", r"torch\.utils\.hooks\.RemovableHandle"),
    # flax.nnx generic TypeVars (Module[A, B]) and internal typing aliases are
    # implementation details, not documented public objects.
    ("py:.*", r"flax\.nnx\.module\.[AB]"),
    ("py:.*", r"flax\.nnx\.filterlib\.Filter"),
    ("py:.*", r"flax\.typing\..*"),
    # jax.numpy.ndarray is documented as a py:attribute (deprecated alias for
    # jax.Array), so it never matches the py:class role autodoc2 emits.
    ("py:class", r"jax\.numpy\.ndarray"),
    # numpy.typing.{NDArray,DTypeLike} are documented as py:data and
    # numpy.int8 as py:attribute; autodoc2 always emits py:class for
    # annotations, so these never match on objtype despite being resolvable.
    ("py:class", r"numpy\.typing\.(NDArray|DTypeLike)"),
    ("py:class", r"numpy\.int8"),
]

# -- linkcheck -------------------------------------------------------------
linkcheck_ignore = [
    # Bot-blocked (HTTP 403) but valid in a browser:
    r"https://patents\.justia\.com/patent/11551063",
    # pytorch docs use JS-generated anchors linkcheck cannot verify:
    r"https://docs\.pytorch\.org/docs/stable/.*#torch\..*",
]
