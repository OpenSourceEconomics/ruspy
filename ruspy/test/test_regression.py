"""
This module contains a regression test for the simulation, discounting and some
estimation functions. The test first draws a random dictionary with some constraints
to ensure enough observations. It then simulates a dataset and the compares the
discounted utility average over all buses, with the theoretical expected value
calculated by the NFXP.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.simulation.simulation import simulate
from ruspy.test.ranodm_init import random_init


@pytest.fixture
def inputs():
    constraints = {"PERIODS": 70000, "BUSES": 200, "BETA": 0.9999}
    return constraints


def test_regression_simulation(inputs):
    init_dict = random_init(inputs)
    beta = init_dict["simulation"]["beta"]
    params = np.array(init_dict["simulation"]["params"])
    probs = np.array(init_dict["simulation"]["known_trans"])
    num_states = init_dict["simulation"]["states"]

    trans_mat = create_transition_matrix(num_states, probs)
    costs = calc_obs_costs(num_states, lin_cost, params, 0.001)
    ev = calc_fixp(trans_mat, costs, beta)

    df = simulate(init_dict["simulation"], ev, trans_mat)

    v_disc = discount_utility(df, beta)

    assert_allclose(v_disc / ev[0], 1, rtol=1e-02)


def discount_utility(df, beta):
    v = 0.0
    for i in df.index.levels[0]:
        v += np.sum(np.multiply(beta ** df.xs([i]).index, df.xs([i])["utilities"]))
    return v / len(df.index.levels[0])
