# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TO_CAD Pipeline'
copyright = '2026, James Jacouris'
author = 'James Jacouris'

version = '1.0'
release = '1.0'

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_multiversion',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Multiversion configuration ----------------------------------------------
smv_branch_whitelist = r'Top3D_Yin_Pipeline_V2|Top3D_Yin_Pipeline_Hybrid_Integrated|main'
smv_remote_whitelist = r'^origin$'

html_sidebars = {
    '**': [
        'versioning.html',
    ],
}
