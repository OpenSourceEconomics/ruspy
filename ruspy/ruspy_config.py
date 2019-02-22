"""This module provides some configuration for the package."""
import sys
import os

import numpy as np

# We only support modern Python.
np.testing.assert_equal(sys.version_info[0], 3)
np.testing.assert_equal(sys.version_info[1] >= 5, True)

# We rely on relative paths throughout the package.
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = ROOT_DIR
TEST_RESOURCES_DIR = PACKAGE_DIR + "/test/resources/"
