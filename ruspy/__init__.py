"""The module allows to run tests from inside the interpreter."""
import os

import pytest

import ruspy.ruspy_config
from ruspy.estimation.estimation import estimate
from ruspy.ruspy_config import PACKAGE_DIR
from ruspy.simulation.simulation import simulate


def test():
    """The function allows to run the tests from inside the interpreter."""
    current_directory = os.getcwd()
    os.chdir(PACKAGE_DIR)
    pytest.main()
    os.chdir(current_directory)
