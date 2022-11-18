"""
This module contains unit tests for the function get_criterion_function of
ruspy.estimation.criterion_function for the MPEC method and different
cost functions.The true values of the parameters and the likelihood are saved
in resources/estimation_test.
The criterion function is tested by calculating the true expected value,
inserting the true expected value and the true parameters in the
criterion function and comparing the result to the true likelihood.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.criterion_function import get_criterion_function
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.nfxp import get_ev
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import cubic_costs
from ruspy.model_code.cost_functions import hyperbolic_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.cost_functions import quadratic_costs
from ruspy.model_code.cost_functions import sqrt_costs

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    disc_fac = 0.9999
    num_states = 90
    alg_details = {}
    init_dict = {
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
        },
        "method": "MPEC",
        "alg_details": alg_details,
    }

    df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")

    out["input data"] = df
    out["init_dict"] = init_dict
    out["num_states"] = num_states
    out["disc_fac"] = disc_fac
    out["alg_details"] = alg_details
    return out


# true outputs
@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["cost_ll_linear"] = 163.584
    out["cost_ll_quad"] = 163.402
    out["cost_ll_cubic"] = 164.632939
    out["cost_ll_hyper"] = 165.11428
    out["cost_ll_sqrt"] = 163.390

    return out


TEST_SPECIFICATIONS = [
    ("linear", "linear", lin_cost, 1e-3, 2),
    ("quadratic", "quad", quadratic_costs, 1e-5, 3),
    ("cubic", "cubic", cubic_costs, 1e-8, 4),
    ("hyperbolic", "hyper", hyperbolic_costs, 1e-1, 2),
    ("square_root", "sqrt", sqrt_costs, 0.01, 2),
]


@pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
def test_criterion_function(inputs, outputs, specification):
    cost_func_name, cost_func_name_short, cost_func, scale, num_params = specification
    df = inputs["input data"]

    init_dict = inputs["init_dict"]
    init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
    init_dict["model_specifications"]["cost_scale"] = scale

    criterion_func, criterion_dev, transition_results = get_criterion_function(
        init_dict, df
    )
    true_params = np.loadtxt(TEST_FOLDER + f"repl_params_{cost_func_name_short}.txt")
    trans_mat = create_transition_matrix(
        inputs["num_states"], np.array(transition_results["x"])
    )
    obs_costs = calc_obs_costs(
        num_states=inputs["num_states"],
        maint_func=cost_func,
        params=true_params,
        scale=scale,
    )
    ev = get_ev(
        true_params, trans_mat, obs_costs, inputs["disc_fac"], inputs["alg_details"]
    )
    true_mpec_params = np.concatenate((ev[0], true_params))
    assert_array_almost_equal(
        criterion_func(mpec_params=true_mpec_params),
        outputs["cost_ll_" + cost_func_name_short],
        decimal=3,
    )


# @pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
# def test_criterion_derivative(inputs, outputs, specification):
#     cost_func_name, cost_func_name_short, cost_func, scale, num_params = specification
#     df = inputs["input data"]
#
#     init_dict = inputs["init_dict"]
#     init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
#     init_dict["model_specifications"]["cost_scale"] = scale
#
#     criterion_func, criterion_dev, transition_results = get_criterion_function(
#         init_dict, df
#     )
#     true_params = np.loadtxt(TEST_FOLDER + f"repl_params_{cost_func_name_short}.txt")
#     trans_mat = create_transition_matrix(
#         inputs["num_states"], np.array(transition_results["x"])
#     )
#     obs_costs = calc_obs_costs(
#         num_states=inputs["num_states"],
#         maint_func=cost_func,
#         params=true_params,
#         scale=scale,
#     )
#     ev = get_ev(
#         true_params, trans_mat, obs_costs, inputs["disc_fac"], inputs["alg_details"]
#     )
#     true_mpec_params = np.concatenate((ev[0], true_params))
#     assert_array_almost_equal(
#         criterion_dev(mpec_params=true_mpec_params)[90:], np.zeros(num_params),
#         decimal=2,
#     )
