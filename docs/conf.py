# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#

import os
from datetime import datetime, timezone
from importlib import metadata

# Load all of the global Astropy configuration
from sphinx_astropy.conf import *

# -- General configuration ----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "3.0"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append("_templates")


# -- Project information ------------------------------------------------------

project = "halotools"
author = "Andrew Hearin"
copyright = f"2017–{datetime.now(tz=timezone.utc).year}, " + author

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

release = metadata.version(project)
version = ".".join(release.split(".")[:2])


# -- Options for HTML output --------------------------------------------------

# A NOTE ON HTML THEMES
html_theme_options = {
    "logotext1": "halo",  # white,  semi-bold
    "logotext2": "tools",  # red, light
    "logotext3": ":docs",  # white,  light
}

# The global astropy configuration uses a custom theme, 'bootstrap-astropy',
# which is installed along with astropy. A different theme can be used or
# the options for this theme can be modified by overriding some of the
# variables set in the global configuration. The variables set in the
# global configuration are listed below, commented out.

# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
# html_theme_path = []

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
# html_theme = None

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = ''

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f"{project} v{release}"

# Output file base name for HTML help builder.
htmlhelp_basename = project + "doc"

# Static files to copy after template files
html_static_path = ["_static"]
html_style = "halotools.css"

# Set canonical URL from the Read the Docs Domain
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

html_context = {
    "default_mode": "light",
    "to_be_indexed": ["stable", "latest"],
    "is_development": False,
    "github_user": "astropy",
    "github_repo": "halotools",
    "github_version": "master",
    "doc_path": "docs",
    # Tell Jinja2 templates the build is running on Read the Docs
    "READTHEDOCS": os.environ.get("READTHEDOCS", "") == "True",
}
# -- Options for LaTeX output -------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual])
latex_documents = [
    ("index", project + ".tex", project + " Documentation", author, "manual")
]


# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", project.lower(), project + " Documentation", [author], 1)]

# -- Options for the edit_on_github extension ---------------------------------
