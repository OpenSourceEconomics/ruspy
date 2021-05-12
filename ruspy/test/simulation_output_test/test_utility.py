from numpy.testing import assert_array_equal, assert_almost_equal
import pytest
import numpy as np
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
    out["df"] = df = simulate(
        init_dict,
        ev,
        costs,
        trans_mat,
    )
    out["v_disc"] = discount_utility(
        df, disc_fac
    )
    return out


def test_regression_simulation_reduced_data_discounted_utility(inputs_sim):

    utility = simulate(
        inputs_sim["init_dict"],
        inputs_sim["ev"],
        inputs_sim["costs"],
        inputs_sim["trans_mat"],
        reduced_data="discounted utility",
    )

    assert_almost_equal(
        utility,
        inputs_sim["v_disc"],
        decimal=4
    )


def test_regression_simulation_reduced_data_utility(inputs_sim):

    utilities = simulate(
        inputs_sim["init_dict"],
        inputs_sim["ev"],
        inputs_sim["costs"],
        inputs_sim["trans_mat"],
        reduced_data="utility",
    )

    assert_array_equal(
        utilities,
        inputs_sim["df"]["utilities"]
        .to_numpy()
        .reshape(
            (inputs_sim["init_dict"]["buses"], inputs_sim["init_dict"]["periods"])
        ),
    )