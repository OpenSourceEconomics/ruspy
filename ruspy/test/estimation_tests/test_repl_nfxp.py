"""
This module contains unit tests for the estimation process for different
cost functions using the NFXP method.
The parameters and likelihood are estimated by minimizing the criterion
function using the minimize function from estimagic with scipy_lbgfsb algorithm.
The estimated parameters and likelihood are compared to the true parameters and
the true likelihood saved in resources/estimation_test.
Moreover, the convergence of the algorithm is tested.
"""
import numpy as np
import pandas as pd
import pytest
from estimagic import minimize
from numpy.testing import assert_allclose

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.criterion_function import get_criterion_function

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    disc_fac = 0.9999
    num_states = 90
    init_dict = {
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
        },
        "method": "NFXP",
        "alg_details": {},
    }

    df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")

    out["input data"] = df
    out["init_dict"] = init_dict
    return out


# true outputs
@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["params_linear"] = np.loadtxt(
        TEST_FOLDER + "repl_params_linear.txt"
    )  # 10.0750, 2.2930
    out["params_quad"] = np.loadtxt(
        TEST_FOLDER + "repl_params_quad.txt"
    )  # 11.48129539, 476.30640207, -2.31414426
    out["params_cubic"] = np.loadtxt(
        TEST_FOLDER + "repl_params_cubic.txt"
    )  # 8.26859, 1.00489, 0.494566, 26.3475
    out["params_hyper"] = np.loadtxt(
        TEST_FOLDER + "repl_params_hyper.txt"
    )  # 8.05929601, 22.96685649
    out["params_sqrt"] = np.loadtxt(
        TEST_FOLDER + "repl_params_sqrt.txt"
    )  # 11.42995702, 3.2308913

    out["cost_ll_linear"] = 163.584
    out["cost_ll_quad"] = 163.402
    out["cost_ll_cubic"] = 164.632939  # 162.885
    out["cost_ll_hyper"] = 165.11428  # 165.423
    out["cost_ll_sqrt"] = 163.390  # 163.395.

    return out


TEST_SPECIFICATIONS = [
    ("linear", "linear", np.array([2, 10]), 1e-3, 2),
    ("quadratic", "quad", np.array([11, 476.3, -2.3]), 1e-5, 3),
    ("cubic", "cubic", np.array([8.3, 1, 0.5, 26]), 1e-8, 4),
    ("hyperbolic", "hyper", np.array([8, 23]), 1e-1, 2),
    ("square_root", "sqrt", np.array([11, 3]), 0.01, 2),
]


# test parameters
@pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
def test_repl_params(inputs, outputs, specification):
    cost_func_name, cost_func_name_short, init_params, scale, num_params = specification
    # specify init_dict with cost function and cost scale as well as input data
    df = inputs["input data"]
    init_dict = inputs["init_dict"]
    init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
    init_dict["model_specifications"]["cost_scale"] = scale
    # specify criterion function
    criterion_func, criterion_dev, result_trans = get_criterion_function(init_dict, df)
    # minimize criterion function
    result_fixp = minimize(
        criterion=criterion_func,
        params=init_params,
        algorithm="scipy_lbfgsb",
        derivative=criterion_dev,
    )
    # compare estimated cost parameters to true parameters
    assert_allclose(
        result_fixp.params, outputs["params_" + cost_func_name_short], atol=1e-1
    )


# test cost likelihood
@pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
def test_cost_ll(inputs, outputs, specification):
    cost_func_name, cost_func_name_short, init_params, scale, num_params = specification
    # specify init_dict with cost function and cost scale as well as input data
    df = inputs["input data"]
    init_dict = inputs["init_dict"]
    init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
    init_dict["model_specifications"]["cost_scale"] = scale
    # specify criterion function
    criterion_func, criterion_dev, result_trans = get_criterion_function(init_dict, df)
    # minimize criterion function
    result_fixp = minimize(
        criterion=criterion_func,
        params=init_params,
        algorithm="scipy_lbfgsb",
        derivative=criterion_dev,
    )
    # compare computed minimum neg log-likelihood to true minimum neg log-likelihood
    assert_allclose(
        result_fixp.criterion, outputs["cost_ll_" + cost_func_name_short], atol=1e-3
    )


# test convergence of algorithm
@pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
def test_success(inputs, specification):
    cost_func_name, cost_func_name_short, init_params, scale, num_params = specification
    init_dict = inputs["init_dict"]
    init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
    init_dict["model_specifications"]["cost_scale"] = scale
    # specify init_dict with cost function and cost scale as well as input data
    df = inputs["input data"]
    # specify criterion function
    criterion_func, criterion_dev, result_trans = get_criterion_function(init_dict, df)
    # minimize criterion function
    result_fixp = minimize(
        criterion=criterion_func,
        params=init_params,
        algorithm="scipy_lbfgsb",
        derivative=criterion_dev,
    )

    # test success of algorithm
    assert result_fixp.success
