# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dieltools'
copyright = '2024, Tomohito Amano, Tamio Yamazaki'
author = 'Tomohito Amano, Tamio Yamazaki'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# !! original option
html_theme = 'alabaster'
html_static_path = ['_static']

# !! https://rcmdnk.com/blog/2016/05/01/computer-brew-file-github/
# !! You need ::  pip install sphinx_rtd_theme
# html_theme = "sphinx_rtd_theme"
