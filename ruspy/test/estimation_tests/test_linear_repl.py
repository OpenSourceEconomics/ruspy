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
from estimagic import minimize
from numpy.testing import assert_allclose
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

    criterion_func, criterion_dev, result_trans = get_criterion_function(init_dict, df)

    result_fixp = minimize(
        criterion=criterion_func,
        params=np.array([2, 10]),
        algorithm="scipy_bfgs",
        derivative=criterion_dev,
    )
    out["params_est"] = result_fixp.params

    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["trans_base"] = np.loadtxt(TEST_FOLDER + "repl_test_trans.txt")
    out["params_base"] = np.loadtxt(TEST_FOLDER + "repl_params_linear.txt")
    out["transition_count"] = np.loadtxt(TEST_FOLDER + "transition_count.txt")
    out["trans_ll"] = 3140.570557
    out["cost_ll"] = 163.584

    return out


def test_repl_params(inputs, outputs):
    # This is as precise as the paper gets
    assert_array_almost_equal(inputs["params_est"], outputs["params_base"], decimal=4)


def test_cost_ll(inputs, outputs):
    # This is as precise as the paper gets
    assert_allclose(inputs["cost_ll"], outputs["cost_ll"], atol=1e-3)


def test_success(inputs):
    assert inputs["status"] == 1
