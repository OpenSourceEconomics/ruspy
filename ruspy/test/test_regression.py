"""
This module contains a regression test for the simulation, discounting and some
estimation functions. The test first draws a random dictionary with some constraints
to ensure enough observations. It then simulates a dataset and the compares the
discounted utility average over all buses, with the theoretical expected value
calculated by the NFXP.
"""

import pytest
from numpy.testing import assert_allclose
import numpy as np
from ruspy.test.ranodm_init import random_init
from ruspy.simulation.simulation import simulate
from ruspy.simulation.value_zero import discount_utility
from ruspy.estimation.estimation_cost_parameters import calc_fixp
from ruspy.estimation.estimation_cost_parameters import cost_func
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix
from ruspy.estimation.estimation_cost_parameters import lin_cost


@pytest.fixture
def inputs():
    constraints = {"PERIODS": 70000, "BUSES": 200, "BETA": 0.9999}
    return constraints


def test_regression_simulation(inputs):
    init_dict = random_init(inputs)
    num_periods = init_dict["simulation"]["periods"]
    beta = init_dict["simulation"]["beta"]
    params = np.array(init_dict["simulation"]["params"])
    probs = np.array(init_dict["simulation"]["known_trans"])
    num_states = init_dict["simulation"]["states"]

    trans_mat = create_transition_matrix(num_states, probs)
    costs = cost_func(num_states, lin_cost, params)
    ev = calc_fixp(num_states, trans_mat, costs, beta)

    df = simulate(init_dict["simulation"], ev, trans_mat)

    utilities = (
        df["utilities"]
        .to_numpy()
        .reshape(init_dict["simulation"]["buses"], num_periods)
    )
    v_disc = discount_utility(utilities, num_periods, beta)

    assert_allclose(v_disc / ev[0], 1, rtol=1e-02)
