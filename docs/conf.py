# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MLWC'
copyright = '2024, Tomohito Amano, Tamio Yamazaki'
author = 'Tomohito Amano, Tamio Yamazaki'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# https://qiita.com/futakuchi0117/items/4d3997c1ca1323259844
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx_rtd_theme',
              'sphinx.ext.todo']

todo_include_todos = True

# https://hesma2.hatenablog.com/entry/2021/04/03/180206#3-%E8%A8%AD%E5%AE%9A%E3%81%AE%E7%B7%A8%E9%9B%86
autodoc_default_flags = [
  'members',
  'private-members'
]

# https://stackoverflow.com/questions/1149280/how-can-i-use-sphinx-autodoc-extension-for-private-methods
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# !! original option
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# !! https://rcmdnk.com/blog/2016/05/01/computer-brew-file-github/
# !! You need ::  pip install sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
