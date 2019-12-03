import pickle as pkl

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal

from ruspy.estimation.estimation import estimate
from ruspy.ruspy_config import TEST_RESOURCES_DIR

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    init_dict = {
        "groups": "group_4",
        "binsize": 5000,
        "beta": 0.9999,
        "states": 90,
        "maint_cost_func": "quadratic",
    }
    df = pkl.load(open(TEST_FOLDER + "group_4.pkl", "rb"))
    result_trans, result_fixp = estimate(init_dict, df)
    out["trans_est"] = result_trans["x"]
    out["params_est"] = result_fixp["x"]
    out["trans_ll"] = result_trans["fun"]
    out["cost_ll"] = result_fixp["fun"]
    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["trans_base"] = np.loadtxt(TEST_FOLDER + "repl_test_trans.txt")
    # out["params_base"] = np.loadtxt(TEST_FOLDER + "repl_test_params.txt")
    out["transition_count"] = np.loadtxt(TEST_FOLDER + "transition_count.txt")
    out["trans_ll"] = 3140.570557
    out["cost_ll"] = 163.584284  # 163.402,
    return out


def test_repl_trans(inputs, outputs):
    assert_array_almost_equal(inputs["trans_est"], outputs["trans_base"])


def test_trans_ll(inputs, outputs):
    assert_allclose(inputs["trans_ll"], outputs["trans_ll"])


def test_cost_ll(inputs, outputs):
    assert_allclose(inputs["cost_ll"], outputs["cost_ll"])
