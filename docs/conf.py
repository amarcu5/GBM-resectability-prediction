import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'GBM resectability prediction'
copyright = '2018, Adam Marcus'
author = 'Adam Marcus'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
  'breathe',
  'exhale',
  'sphinx_rtd_theme'
]

# Setup the breathe extension
breathe_projects = {}
breathe_projects[project] = './doxyoutput/xml'
breathe_default_project = project

# Setup the exhale extension
exhale_args = {
  'containmentFolder':     './api',
  'rootFileName':          'root.rst',
  'rootFileTitle':         project + ' documentation',
  'doxygenStripFromPath':  '..',
  'createTreeView':        True,
  'exhaleExecutesDoxygen': True,
  'exhaleDoxygenStdin':    'INPUT = ../src\nEXCLUDE_PATTERNS = *.cc *.c'
}

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
