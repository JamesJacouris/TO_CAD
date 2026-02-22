# Configuration file for the Sphinx documentation builder.
# Branch scope: Main_V2 only.
#
# Build:  cd docs && make clean html
# Output: docs/_build/html/index.html

import os
import sys

# ── Repo root on path so autodoc can import src.* ──────────────────────────
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "TO_CAD Pipeline"
copyright = "2026, James Jacouris"
author = "James Jacouris"
version = "1.0"
release = "1.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    # Core autodoc
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # Docstring style support (Google / NumPy style)
    "sphinx.ext.napoleon",
    # Source-code links on API pages
    "sphinx.ext.viewcode",
    # Cross-links to Python stdlib & NumPy docs
    "sphinx.ext.intersphinx",
    # Automatic section labels for :ref: cross-links
    "sphinx.ext.autosectionlabel",
    # Inline math
    "sphinx.ext.mathjax",
    # Pipeline dependency diagrams (requires system graphviz / dot)
    "sphinx.ext.graphviz",
]

# Graphviz output format (SVG is crisp and zoomable in HTML)
graphviz_output_format = "svg"

# ---------------------------------------------------------------------------
# Autodoc settings
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Skip FreeCAD / Open3D modules that can't be imported in CI
autodoc_mock_imports = [
    "FreeCAD", "FreeCADGui", "Part", "Mesh",
    "PySide", "PySide2", "PySide6",
    "open3d",
    "numba",
    "optuna",
]

# ---------------------------------------------------------------------------
# Napoleon (Google / NumPy docstring parsing)
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# ---------------------------------------------------------------------------
# Autosummary
# ---------------------------------------------------------------------------
autosummary_generate = True

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# ---------------------------------------------------------------------------
# Auto-section labels
# ---------------------------------------------------------------------------
# Prefix labels with document name to avoid collisions
autosectionlabel_prefix_document = True

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress specific known warnings that can't be fixed
suppress_warnings = [
    "autosectionlabel.*",   # duplicate label warnings from long headings
    "ref.python",           # broken intersphinx for private stdlib
]

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

html_title = "TO_CAD Pipeline Docs"
html_short_title = "TO_CAD"

# ---------------------------------------------------------------------------
# Multi-version (single branch build for Main_V2)
# ---------------------------------------------------------------------------
# sphinx_multiversion kept but branch list restricted to Main_V2.
# To build: cd docs && sphinx-multiversion . _build/html
# To build single version (faster): cd docs && make html
try:
    import sphinx_multiversion  # noqa: F401
    extensions.append("sphinx_multiversion")
    smv_branch_whitelist = r"^Main_V2$"
    smv_remote_whitelist = r"^origin$"
    smv_tag_whitelist = r"^$"   # no tags
    html_sidebars = {
        "**": [
            "versions.html",
            "globaltoc.html",
            "relations.html",
            "sourcelink.html",
            "searchbox.html",
        ]
    }
except ImportError:
    pass
