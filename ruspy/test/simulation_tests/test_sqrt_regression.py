"""
This module contains a regression test for the simulation, discounting and some
estimation functions. The test first draws a random dictionary with some constraints
to ensure enough observations. It then simulates a dataset and the compares the
discounted utility average over all buses, with the theoretical expected value
calculated by the NFXP.
"""
import numpy as np
from numpy.testing import assert_allclose

from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import sqrt_costs
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.simulation.simulation import simulate
from ruspy.test.ranodm_init import random_init
from ruspy.test.simulation_tests.regression_aux import discount_utility


def test_regression_simulation(inputs):
    init_dict = random_init(inputs)

    # Draw parameter
    param_1 = np.random.normal(11.5, 0.2)
    param_2 = np.random.normal(3.2, 0.5)
    params = np.array([param_1, param_2])

    beta = init_dict["simulation"]["beta"]
    probs = np.array(init_dict["simulation"]["known_trans"])
    num_states = 500

    trans_mat = create_transition_matrix(num_states, probs)

    costs = calc_obs_costs(num_states, sqrt_costs, params, 0.01)

    ev = calc_fixp(trans_mat, costs, beta)

    df = simulate(init_dict["simulation"], ev, costs, trans_mat)

    v_disc = discount_utility(df, beta)

    assert_allclose(v_disc / ev[0], 1, rtol=1e-02)
