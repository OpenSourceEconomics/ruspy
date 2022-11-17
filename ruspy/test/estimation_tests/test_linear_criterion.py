"""
test get_criterion_function for NFXP and linear cost function
"""
"""
This module contains tests for the data and estimation code of the ruspy project. The
settings for this tests is specified in resources/replication_test/init_replication.yml.
The test first reads the original data, then processes the data to a pandas DataFrame
suitable for the estimation process. After estimating all the relevant parameters,
they are compared to the results, from the paper. As this test runs the complete
data_reading, data processing and runs several times the NFXP it is the one with the
longest test time.
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
    scale = 1e-3
    init_dict = {
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
            "maint_cost_func": "linear",
            "cost_scale": scale,
        },
        "method": "NFXP",
        "alg_details": {},
    }

    df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")

    criterion_func, criterion_dev = get_criterion_function(init_dict, df)
    out["criterion_function"] = criterion_func
    out["criterion_derivative"] = criterion_dev
    return out


# true outputs
@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["params_base"] = np.loadtxt(TEST_FOLDER + "repl_params_linear.txt")
    out["cost_ll"] = 163.584

    return out


def test_criterion_function(inputs, outputs):
    criterion_function = inputs["criterion_function"]
    true_params = outputs["params_base"]

    assert_array_almost_equal(
        criterion_function(true_params),
        outputs["cost_ll"],
        decimal=3,
    )


def test_criterion_derivative(inputs, outputs):
    criterion_derivative = inputs["criterion_derivative"]
    true_params = outputs["params_base"]
    assert_array_almost_equal(
        criterion_derivative(true_params),
        np.array([0, 0]),
        decimal=2,
    )
