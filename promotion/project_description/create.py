#!/usr/bin/env python

import os
import yaml
from material.latex_descriptives import create_desc
from material.figure_1 import plot_convergence


with open('material/init_paper.yml') as y:
    init_dict = yaml.load(y)
create_desc()
plot_convergence(init_dict)


for type_ in ['pdflatex', 'bibtex', 'pdflatex', 'pdflatex']:
    os.system(type_ + ' project_description')


