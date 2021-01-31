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
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.simulation.simulation import read_init_dict
from ruspy.simulation.simulation import simulate
from ruspy.test.ranodm_init import random_init
from ruspy.test.regression_sim_tests.regression_aux import disc_ut_loop
from ruspy.test.regression_sim_tests.regression_aux import discount_utility


def test_regression_simulation(inputs):
    init_dict = random_init(inputs)

    # Draw parameter
    param1 = np.random.normal(10.0, 2)
    param2 = np.random.normal(2.3, 0.5)
    params = np.array([param1, param2])

    disc_fac = init_dict["simulation"]["discount_factor"]
    probs = np.array(init_dict["simulation"]["known_trans"])
    num_states = 300

    trans_mat = create_transition_matrix(num_states, probs)
    costs = calc_obs_costs(num_states, lin_cost, params, 0.001)
    ev = calc_fixp(trans_mat, costs, disc_fac)[0]

    df = simulate(init_dict["simulation"], ev, costs, trans_mat)

    v_disc = discount_utility(df, disc_fac)

    assert_allclose(v_disc / ev[0], 1, rtol=1e-02)


def test_regression_simulation_reduced_data(inputs):
    init_dict = random_init(inputs)

    # Draw parameter
    param1 = np.random.normal(10.0, 2)
    param2 = np.random.normal(2.3, 0.5)
    params = np.array([param1, param2])

    num_buses, disc_fac, num_periods = read_init_dict(init_dict["simulation"])
    probs = np.array(init_dict["simulation"]["known_trans"])
    num_states = 300

    trans_mat = create_transition_matrix(num_states, probs)
    costs = calc_obs_costs(num_states, lin_cost, params, 0.001)
    ev = calc_fixp(trans_mat, costs, disc_fac)[0]

    utilities = simulate(
        init_dict["simulation"], ev, costs, trans_mat, reduced_data="utility"
    )

    v_disc = disc_ut_loop(utilities, disc_fac,)

    assert_allclose(v_disc / ev[0], 1, rtol=1e-02)
