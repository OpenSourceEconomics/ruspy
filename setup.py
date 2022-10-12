from setuptools import find_packages
from setuptools import setup

# Package meta-data.
NAME = "ruspy"
DESCRIPTION = (
    "An open-source package for the simulation and estimation of a prototypical"
    " infinite-horizon dynamic discrete choice model based on Rust (1987)."
)
URL = ""
EMAIL = "maximilian.blesch@hu-berlin.de"
AUTHOR = "Maximilian Blesch"


setup(
    name=NAME,
    version="1.1",
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    license="MIT",
    include_package_data=True,
)
