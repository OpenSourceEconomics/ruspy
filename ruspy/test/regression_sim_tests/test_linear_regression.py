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
from numpy.testing import assert_array_equal

from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.simulation.simulation import simulate
from ruspy.test.ranodm_init import random_init
from ruspy.test.regression_sim_tests.regression_aux import discount_utility


@pytest.fixture(scope="module")
def inputs_sim(inputs):
    out = {}
    out["init_dict"] = init_dict = random_init(inputs)["simulation"]

    # Draw parameter
    param1 = np.random.normal(10.0, 2)
    param2 = np.random.normal(2.3, 0.5)
    params = np.array([param1, param2])

    disc_fac = init_dict["discount_factor"]
    probs = np.array(init_dict["known_trans"])
    num_states = 300

    out["trans_mat"] = trans_mat = create_transition_matrix(num_states, probs)
    out["costs"] = costs = calc_obs_costs(num_states, lin_cost, params, 0.001)
    out["ev"] = ev = calc_fixp(trans_mat, costs, disc_fac)[0]
    out["df"] = simulate(
        init_dict,
        ev,
        costs,
        trans_mat,
    )
    return out


def test_regression_simulation(inputs_sim):

    v_disc = discount_utility(
        inputs_sim["df"], inputs_sim["init_dict"]["discount_factor"]
    )

    assert_allclose(v_disc / inputs_sim["ev"][0], 1, rtol=1e-02)
