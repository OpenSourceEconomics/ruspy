"""
This module contains tests for the simulation process. The setting and the true
results of the simulation are saved in resources/simulation_test/linear_5_agents.pkl.
This module first takes the settings and simulates a dataset. Then the results are
compared to the true saved results.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ruspy.simulation.simulation import simulate
from ruspy.estimation.estimation_cost_parameters import (
    calc_fixp,
    cost_func,
    lin_cost,
    create_transition_matrix,
)
from ruspy.test.ranodm_init import random_init


@pytest.fixture
def inputs():
    init_dict = random_init()["simulation"]
    out = {}
    out["df"], out["unobs"], out["utilities"], num_states = simulate(init_dict)
    costs = cost_func(num_states, lin_cost, np.array(init_dict["params"]))
    trans_mat = create_transition_matrix(num_states, np.array(init_dict["known_trans"]))
    ev = calc_fixp(num_states, trans_mat, costs, init_dict["beta"])
    out["df_known"], out["unobs_known"], out["utilities_known"], num_states = simulate(
        init_dict, ev_known=ev
    )
    return out


def test_states_known(inputs):
    assert_array_equal(inputs["df_known"].state, inputs["df"].state)


def test_decision_known(inputs):
    assert_array_equal(inputs["df_known"].decision, inputs["df"].decision)


def test_unobs_known(inputs):
    assert_array_equal(inputs["unobs_known"], inputs["unobs"])


def test_utilities_known(inputs):
    assert_array_equal(inputs["utilities_known"], inputs["utilities"])
