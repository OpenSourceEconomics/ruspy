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
from ruspy.estimation.estimation_cost_parameters import calc_fixp, cost_func, lin_cost
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix


with open(TEST_RESOURCES_DIR + "simulation_test/sim_test_init.yml") as y:
    init_dict = yaml.safe_load(y)


@pytest.fixture
def inputs():
    out = {}
    out["df"], out["unobs"], out["utilities"], num_states = simulate(
        init_dict, seed=7023
    )
    costs = cost_func(num_states, lin_cost, init_dict["params"])
    trans_mat = create_transition_matrix(num_states, np.array(init_dict["known trans"]))
    init_dict["ev_known"] = calc_fixp(num_states, trans_mat, costs, init_dict["beta"])
    out["df_known"], out["unobs_known"], out["utilities_known"], num_states = simulate(
        init_dict, seed=7023
    )
    return out


@pytest.fixture
def outputs():
    out = dict()
    out["states"] = np.loadtxt(TEST_RESOURCES_DIR + "simulation_test/states.txt")
    out["decision"] = np.loadtxt(TEST_RESOURCES_DIR + "simulation_test/decision.txt")
    out["unobs"] = np.loadtxt(TEST_RESOURCES_DIR + "simulation_test/unobs.txt")
    out["utilities"] = np.loadtxt(TEST_RESOURCES_DIR + "simulation_test/utilities.txt")
    return out


def test_states(inputs, outputs):
    assert_array_equal(outputs["states"], inputs["df"].state)


def test_decision(inputs, outputs):
    assert_array_equal(outputs["decision"], inputs["df"].decision)


def test_unobs(inputs, outputs):
    assert_array_equal(inputs["unobs"].flatten(), outputs["unobs"])


def test_utilities(inputs, outputs):
    assert_array_equal(inputs["utilities"].flatten(), outputs["utilities"])


def test_states_known(inputs):
    assert_array_equal(inputs["df_known"].state, inputs["df"].state)


def test_decision_known(inputs):
    assert_array_equal(inputs["df_known"].decision, inputs["df"].decision)


def test_unobs_known(inputs):
    assert_array_equal(inputs["unobs_known"], inputs["unobs"])


def test_utilities_known(inputs):
    assert_array_equal(inputs["utilities_known"], inputs["utilities"])
