#!/usr/bin/env python

import os

os.chdir("documentation/")
os.system("python create_documentation.py")
os.chdir("../promotion/project_description")
os.system("python create_project_description.py")
