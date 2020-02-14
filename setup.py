import os
import sys
from shutil import rmtree

from setuptools import Command
from setuptools import find_packages
from setuptools import setup

# Package meta-data.
NAME = "ruspy"
DESCRIPTION = "Hier sollte eine Beschreibung stehen."
URL = ""
EMAIL = "s6mables@uni-bonn.de"
AUTHOR = "Maximilian Blesch"

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy",
    "pytest",
    "pandas",
    "scipy",
    "pyyaml",
    "matplotlib",
    "numba",
]


here = os.path.abspath(os.path.dirname(__file__))

about = {}


class PublishCommand(Command):
    """Support setup.py publish."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print(f"\033[1m{s}\033[0m")

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except FileNotFoundError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system(f"{sys.executable} setup.py sdist bdist_wheel --universal")

        self.status("Uploading the package to PyPi via Twine…")
        os.system("twine upload dist/*")

        sys.exit()


setup(
    name=NAME,
    version="0.1.0",
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    license="MIT",
    include_package_data=True,
    cmdclass={"publish": PublishCommand},
)
