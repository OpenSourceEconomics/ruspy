"""
This module contains unit tests for the function get_criterion_function of
ruspy.estimation.criterion_function for the NFXP method and different
cost functions.The true values of the parameters and the likelihood are saved
in resources/estimation_test.
The criterion function is tested by inserting the true parameters in the
criterion function and comparing the result to the true likelihood.
Its derivative is tested by inserting the true parameters in the derivative
of the criterion function and comparing the result to zero.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

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
    out["cost_ll_linear"] = 163.584
    out["cost_ll_quad"] = 163.402
    out["cost_ll_cubic"] = 164.632939  # 162.885
    out["cost_ll_hyper"] = 165.11428  # 165.423
    out["cost_ll_sqrt"] = 163.390  # 163.395.

    return out


TEST_SPECIFICATIONS = [
    ("linear", "linear", 1e-3, 2),
    ("quadratic", "quad", 1e-5, 3),
    ("cubic", "cubic", 1e-8, 4),
    ("hyperbolic", "hyper", 1e-1, 2),
    ("square_root", "sqrt", 0.01, 2),
]


@pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
def test_criterion_function(inputs, outputs, specification):
    cost_func_name, cost_func_name_short, scale, num_params = specification
    df = inputs["input data"]

    init_dict = inputs["init_dict"]
    init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
    init_dict["model_specifications"]["cost_scale"] = scale

    criterion_func, criterion_dev, transition_results = get_criterion_function(
        init_dict, df
    )

    assert_array_almost_equal(
        criterion_func(
            np.loadtxt(TEST_FOLDER + f"repl_params_{cost_func_name_short}.txt")
        ),
        outputs["cost_ll_" + cost_func_name_short],
        decimal=3,
    )


@pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
def test_criterion_derivative(inputs, outputs, specification):
    cost_func_name, cost_func_name_short, scale, num_params = specification
    df = inputs["input data"]

    init_dict = inputs["init_dict"]
    init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
    init_dict["model_specifications"]["cost_scale"] = scale

    criterion_func, criterion_dev, transition_results = get_criterion_function(
        init_dict, df
    )

    assert_array_almost_equal(
        criterion_dev(
            np.loadtxt(TEST_FOLDER + f"repl_params_{cost_func_name_short}.txt")
        ),
        np.zeros(num_params),
        decimal=2,
    )
