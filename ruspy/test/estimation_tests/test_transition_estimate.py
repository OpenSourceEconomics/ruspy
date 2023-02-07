"""
This module contains unit tests for the function estimate_transitions from
ruspy.estimation.estimation_transitions. The values to compare the results with
are saved in resources/estimation_test. The setting of the test is documented in the
inputs section in test module.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.estimation_transitions import estimate_transitions

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")

    transition_results = estimate_transitions(df)
    out = {
        "params_est": transition_results["x"],
        "trans_count": transition_results["trans_count"],
        "fun": transition_results["fun"],
    }
    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["trans_base"] = np.loadtxt(TEST_FOLDER + "repl_test_trans.txt")
    out["transition_count"] = np.loadtxt(TEST_FOLDER + "transition_count.txt")
    out["trans_ll"] = 3140.570557
    return out


def test_repl_trans(inputs, outputs):
    assert_array_almost_equal(inputs["params_est"], outputs["trans_base"])


def test_trans_ll(inputs, outputs):
    assert_allclose(inputs["fun"], outputs["trans_ll"])


def test_transcount(inputs, outputs):
    assert_allclose(inputs["trans_count"], outputs["transition_count"])
