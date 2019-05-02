#!/usr/bin/env python

import os
import yaml
from material.latex_descriptives import create_desc
from material.figure_1 import plot_convergence
from material.replication import get_table

with open("material/init_paper.yml") as y:
    init_dict = yaml.safe_load(y)
if not os.path.isfile("figures/descr_2a.txt"):
    create_desc()
if not os.path.isfile("figures/replication.txt"):
    get_table(init_dict)
if not os.path.isfile("figures/figure_1.png"):
    plot_convergence(init_dict)


for type_ in ["pdflatex", "bibtex", "pdflatex", "pdflatex"]:
    os.system(type_ + " project_description")
