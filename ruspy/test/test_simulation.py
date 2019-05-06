"""
This module contains tests for the simulation process. The setting and the true
results of the simulation are saved in resources/simulation_test/linear_5_agents.pkl.
This module first takes the settings and simulates a dataset. Then the results are
compared to the true saved results.
"""

import yaml
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ruspy.simulation.simulation import simulate
from ruspy.ruspy_config import TEST_RESOURCES_DIR


with open(TEST_RESOURCES_DIR + "simulation_test/sim_test_init.yml") as y:
    init_dict = yaml.safe_load(y)


@pytest.fixture
def inputs():
    out = {}
    out["df"], out["unobs"], out["utilities"], num_states = simulate(
        init_dict["simulation"]
    )
    return out


@pytest.fixture
def outputs():
    out = dict()
    out["states"] = np.loadtxt(TEST_RESOURCES_DIR + "simulation_test/states.txt")
    out["decision"] = np.loadtxt(TEST_RESOURCES_DIR + "simulation_test/decision.txt")
    out["unobs"] = np.loadtxt(TEST_RESOURCES_DIR + "simulation_test/unobs.txt")
    out["utilities"] = np.loadtxt(TEST_RESOURCES_DIR + "simulation_test/utilites.txt")
    return out


def test_states(inputs, outputs):
     assert_array_equal(outputs['states'], inputs['df'].state)


def test_decision(inputs, outputs):
    assert_array_equal(outputs['decision'], inputs['df'].decision)


def test_unobs(inputs, outputs):
    assert_array_equal(np.ndarray.flatten(inputs["unobs"]), outputs["unobs"])


def test_utilities(inputs, outputs):
    assert_array_equal(np.ndarray.flatten(inputs["utilities"]), outputs["utilities"])