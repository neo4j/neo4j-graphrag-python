#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import sys
import typing

# Add the root of the project to the path so that Sphinx can find the Python sources
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "neo4j-genai-python"
copyright = "2024, Neo4j, Inc."
author = "Neo4j, Inc."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "neo4j"
html_theme_path = ["themes"]


# 01-nav.js is a copy of a js file of the same name that is included in the
# docs-ui bundle
def setup(app):
    app.add_js_file("https://neo4j.com/docs/assets/js/site.js", loading_method="defer")
    app.add_js_file("js/12-fragment-jumper.js", loading_method="defer")


# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {"gentree": "gentree.html"}

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = "sphinx"
pygments_style = "friendly"

# Don't include type hints in function signatures
autodoc_typehints = "description"

autodoc_type_aliases = {
    # The code-base uses `import typing_extensions as te`.
    # Re-write these to use `typing` instead, as Sphinx always resolves against
    # the latest version of the `typing` module.
    # This is a work-around to make Sphinx resolve type hints correctly, even
    # though we're using `from __future__ import annotations`.
    "te": typing,
    # Type alias that's only defined and imported if `typing.TYPE_CHECKING`
    # is `True`.
    "_TAuth": "typing.Tuple[typing.Any, typing.Any] | Auth | None",
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "sidebar_includehidden": True,
    "sidebar_collapse": True,
}

autodoc_default_options = {
    "member-order": "bysource",
    # 'special-members': '__init__',
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
