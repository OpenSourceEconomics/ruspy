"""

test get_criterion_function for MPEC and linear cost function
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.criterion_function import get_criterion_function
from ruspy.estimation.est_cost_params import get_ev
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    disc_fac = 0.9999
    num_states = 90
    scale = 1e-3
    alg_details = {}
    init_dict = {
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
            "maint_cost_func": "linear",
            "cost_scale": scale,
        },
        "method": "MPEC",
        "alg_details": alg_details,
    }

    df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")

    criterion_func, criterion_dev = get_criterion_function(init_dict, df)
    out["criterion_function"] = criterion_func
    out["criterion_derivative"] = criterion_dev
    out["df"] = df
    out["scale"] = scale
    out["disc_fac"] = disc_fac
    out["num_states"] = num_states
    out["alg_details"] = alg_details
    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["params_base"] = np.loadtxt(TEST_FOLDER + "repl_params_linear.txt")
    out["cost_ll"] = 163.584
    return out


def test_criterion_function(inputs, outputs):
    criterion_function = inputs["criterion_function"]
    true_params = outputs["params_base"]
    transition_results = estimate_transitions(inputs["df"])
    trans_mat = create_transition_matrix(
        inputs["num_states"], np.array(transition_results["x"])
    )
    obs_costs = calc_obs_costs(
        num_states=inputs["num_states"],
        maint_func=lin_cost,
        params=true_params,
        scale=inputs["scale"],
    )
    ev = get_ev(
        true_params, trans_mat, obs_costs, inputs["disc_fac"], inputs["alg_details"]
    )
    true_mpec_params = np.concatenate((ev[0], true_params))
    assert_array_almost_equal(
        criterion_function(mpec_params=true_mpec_params),
        outputs["cost_ll"],
        decimal=2,
    )


# def test_criterion_derivative(inputs, outputs):
#     criterion_derivative = inputs["criterion_derivative"]
#     true_params = outputs["params_base"]
#     num_states = inputs["num_states"]
#     transition_results = estimate_transitions(inputs["df"])
#     trans_mat = create_transition_matrix(
#         inputs["num_states"], np.array(transition_results["x"])
#     )
#     obs_costs = calc_obs_costs(
#         num_states=inputs["num_states"],
#         maint_func=lin_cost, params=true_params,
#         scale=inputs["scale"],
#     )
#     ev = get_ev(
#         true_params, trans_mat, obs_costs, inputs["disc_fac"], inputs["alg_details"]
#     )
#     true_mpec_params = np.concatenate((ev[0],true_params))
#     assert_array_almost_equal(
#         criterion_derivative(mpec_params=true_mpec_params)[num_states:],
#         np.array([0, 0]),
#         decimal=2,
#     )
