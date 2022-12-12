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
            "num_states": num_states,
        },
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
    out["params_quadratic"] = np.loadtxt(
        TEST_FOLDER + "repl_params_quad.txt"
    )  # 11.48129539, 476.30640207, -2.31414426
    out["params_cubic"] = np.loadtxt(
        TEST_FOLDER + "repl_params_cubic.txt"
    )  # 8.26859, 1.00489, 0.494566, 26.3475
    out["params_hyperbolic"] = np.loadtxt(
        TEST_FOLDER + "repl_params_hyper.txt"
    )  # 8.05929601, 22.96685649
    out["params_square_root"] = np.loadtxt(
        TEST_FOLDER + "repl_params_sqrt.txt"
    )  # 11.42995702, 3.2308913

    out["cost_ll_linear"] = 163.584
    out["cost_ll_quadratic"] = 163.402
    out["cost_ll_cubic"] = 164.632939  # 162.885
    out["cost_ll_hyperbolic"] = 165.11428  # 165.423
    out["cost_ll_square_root"] = 163.390  # 163.395.

    return out


TEST_SPECIFICATIONS = [
    ("linear", np.array([2, 10]), 1e-3),
    # ("quadratic", np.array([11, 476.3, -2.3]), 1e-5),
    # ("cubic", np.array([8.3, 1, 0.5, 26]), 1e-8),
    ("hyperbolic", np.array([8, 23]), 1e-1),
    ("square_root", np.array([11, 3]), 0.01),
]


# test parameters
@pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
def test_nfxp(inputs, outputs, specification):
    cost_func_name, init_params, scale = specification
    # specify init_dict with cost function and cost scale as well as input data
    df = inputs["input data"]
    init_dict = inputs["init_dict"]
    init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
    init_dict["model_specifications"]["cost_scale"] = scale
    init_dict["method"] = "NFXP"
    # specify criterion function
    criterion_func, criterion_dev, result_trans = get_criterion_function(init_dict, df)
    # minimize criterion function
    result_fixp = minimize(
        criterion=criterion_func,
        params=init_params,
        algorithm="scipy_lbfgsb",
        derivative=criterion_dev,
    )
    breakpoint()
    # compare estimated cost parameters to true parameters
    assert_allclose(result_fixp.params, outputs["params_" + cost_func_name], atol=1e-1)
    # compare computed minimum neg log-likelihood to true minimum neg log-likelihood
    assert_allclose(
        result_fixp.criterion, outputs["cost_ll_" + cost_func_name], atol=1e-3
    )

    # test success of algorithm
    assert result_fixp.success


@pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
def test_bhhh(inputs, outputs, specification):
    cost_func_name, init_params, scale = specification
    # specify init_dict with cost function and cost scale as well as input data
    df = inputs["input data"]
    init_dict = inputs["init_dict"]
    init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
    init_dict["model_specifications"]["cost_scale"] = scale
    init_dict["method"] = "NFXP_BHHH"
    # specify criterion function
    criterion_func, criterion_dev, result_trans = get_criterion_function(init_dict, df)
    # minimize criterion function
    # ToDo: We start at optimum. This can't be the solution!
    result_fixp = minimize(
        criterion=criterion_func,
        params=outputs["params_" + cost_func_name],
        algorithm="bhhh",
        derivative=criterion_dev,
    )
    # compare estimated cost parameters to true parameters
    assert_allclose(result_fixp.params, outputs["params_" + cost_func_name], atol=1e-1)
    # compare computed minimum neg log-likelihood to true minimum neg log-likelihood
    assert_allclose(
        result_fixp.criterion, outputs["cost_ll_" + cost_func_name], atol=1e-3
    )
    #
    # # test success of algorithm
    # assert result_fixp.success


@pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
def test_mpec(inputs, outputs, specification):
    cost_func_name, init_params, scale = specification
    # specify init_dict with cost function and cost scale as well as input data
    df = inputs["input data"]
    init_dict = inputs["init_dict"]
    init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
    init_dict["model_specifications"]["cost_scale"] = scale
    num_states = init_dict["model_specifications"]["num_states"]
    init_dict["method"] = "MPEC"
    # specify criterion function
    (
        criterion_func,
        criterion_dev,
        constraint,
        constraint_dev,
        result_trans,
    ) = get_criterion_function(init_dict, df)

    x0 = np.zeros(num_states + init_params.shape[0], dtype=float)
    x0[num_states:] = init_params
    # minimize criterion function
    result_mpec = minimize(
        criterion=criterion_func,
        params=x0,
        algorithm="nlopt_slsqp",
        derivative=criterion_dev,
        constraints={
            "type": "nonlinear",
            "func": constraint,
            "derivative": constraint_dev,
            "value": np.zeros(num_states, dtype=float),
        },
    )
    # compare computed minimum neg log-likelihood to true minimum neg log-likelihood
    assert_allclose(
        result_mpec.criterion, outputs["cost_ll_" + cost_func_name], atol=1e-3
    )

    # compare estimated cost parameters to true parameters
    assert_allclose(
        result_mpec.params[90:], outputs["params_" + cost_func_name], atol=1e-1
    )

    # test success of algorithm
    assert result_mpec.success
