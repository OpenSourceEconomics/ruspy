"""This module provides some configuration for the package."""
import sys
import os

import numpy as np

# We only support modern Python.
np.testing.assert_equal(sys.version_info[:2] >= (3, 6), True)

# We rely on relative paths throughout the package.
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_RESOURCES_DIR = PACKAGE_DIR + "/test/resources/"
