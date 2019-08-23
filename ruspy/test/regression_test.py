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
from ruspy.simulation.value_zero import discount_utility, calc_ev_0
from ruspy.estimation.estimation_cost_parameters import calc_fixp
from ruspy.estimation.estimation_cost_parameters import cost_func
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix
from ruspy.estimation.estimation_cost_parameters import lin_cost


@pytest.fixture
def inputs():
    constraints = {"PERIODS": 70000, "BUSES": 100, "BETA": 0.9999}
    return constraints


def test_regression_simulation(inputs):
    init_dict = random_init(inputs)
    num_buses = init_dict["simulation"]["buses"]
    num_periods = init_dict["simulation"]["periods"]
    beta = init_dict["simulation"]["beta"]
    params = np.array(init_dict["simulation"]["params"])
    probs = np.array(init_dict["simulation"]["known_trans"])
    num_states = 200

    trans_mat = create_transition_matrix(num_states, probs)
    costs = cost_func(num_states, lin_cost, params)
    ev = calc_fixp(num_states, trans_mat, costs, beta)

    df, unobs, utilities = simulate(init_dict["simulation"], ev, trans_mat)
    ev_calc = calc_ev_0(ev, unobs, num_buses)

    v_disc_ = np.array([0.0, 0.0])
    v_disc = discount_utility(v_disc_, num_periods, utilities, beta)

    assert_allclose(v_disc[1] / ev_calc, 1, rtol=1e-02)
