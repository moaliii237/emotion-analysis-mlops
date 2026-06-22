import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

project = 'EmotionScope'
copyright = '2026, NLP1'
author = 'NLP1'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

html_theme = 'furo'

templates_path = ['_templates']
exclude_patterns = []