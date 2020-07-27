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
from ruspy.model_code.cost_functions import hyperbolic_costs
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.simulation.simulation import simulate
from ruspy.test.ranodm_init import random_init
from ruspy.test.regression_sim_tests.regression_aux import discount_utility


def test_regression_simulation(inputs):
    init_dict = random_init(inputs)

    # Draw parameter
    param_1 = np.random.normal(8, 1)
    param_2 = np.random.normal(23, 2)
    params = np.array([param_1, param_2])

    disc_fac = init_dict["simulation"]["discount_factor"]
    probs = np.array(init_dict["simulation"]["known_trans"])
    num_states = 600

    trans_mat = create_transition_matrix(num_states, probs)

    costs = calc_obs_costs(num_states, hyperbolic_costs, params, 0.1)

    ev = calc_fixp(trans_mat, costs, disc_fac)[0]

    df = simulate(init_dict["simulation"], ev, costs, trans_mat)

    v_disc = discount_utility(df, disc_fac)

    assert_allclose(v_disc / ev[0], 1, rtol=1e-01)
