"""
test get_criterion_function for NFXP and quadratic cost function
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
    scale = 1e-5
    init_dict = {
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
            "maint_cost_func": "quadratic",
            "cost_scale": scale,
        },
        "method": "NFXP",
        "alg_details": {},
    }

    df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")
    (
        criterion_func,
        criterion_kwargs,
        criterion_dev,
        criterion_dev_kwargs,
    ) = get_criterion_function(init_dict, df)

    out["criterion_function"] = criterion_func
    out["criterion_derivative"] = criterion_dev
    out["criterion_kwargs"] = criterion_kwargs
    out["criterion_derivative_kwargs"] = criterion_dev_kwargs

    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["params_base"] = np.loadtxt(TEST_FOLDER + "repl_params_quad.txt")
    out["cost_ll"] = 163.402
    return out


def test_criterion_function(inputs, outputs):
    criterion_function = inputs["criterion_function"]
    criterion_kwargs = inputs["criterion_kwargs"]
    # true_params = np.loadtxt(TEST_FOLDER + "repl_params_linear.txt")
    true_params = outputs["params_base"]

    assert_array_almost_equal(
        criterion_function(true_params, **criterion_kwargs),
        outputs["cost_ll"],
        decimal=2,
    )


# here we test the derivative of the criterion function
# at the optimum , the derivative should be equal to zero


def test_criterion_derivative(inputs, outputs):
    criterion_derivative = inputs["criterion_derivative"]
    criterion_derivative_kwargs = inputs["criterion_derivative_kwargs"]
    true_params = outputs["params_base"]

    assert_array_almost_equal(
        criterion_derivative(true_params, **criterion_derivative_kwargs),
        np.array([0, 0, 0]),
        decimal=2,
    )
