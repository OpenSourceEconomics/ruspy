#!/usr/bin/env python
""" This script builds the Appendix.
"""

# standard library
import os

for type_ in ['pdflatex', 'bibtex', 'pdflatex', 'pdflatex']:
    os.system(type_ + ' project_description')
