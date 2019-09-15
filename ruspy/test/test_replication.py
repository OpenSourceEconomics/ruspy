"""
This module contains tests for the data and estimation code of the ruspy project. The
settings for this tests is specified in resources/replication_test/init_replication.yml.
The test first reads the original data, then processes the data to a pandas DataFrame
suitable for the estimation process. After estimating all the relevant parameters,
they are compared to the results, from the paper. As this test runs the complete
data_reading, data processing and runs several times the NFXP it is the one with the
longest test time.
"""

import pytest
import numpy as np
import pickle as pkl
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from ruspy.estimation.estimation import estimate
from ruspy.ruspy_config import TEST_RESOURCES_DIR
from ruspy.estimation.estimation_transitions import create_increases


TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"

init_dict = {"groups": "group_4", "binsize": 5000, "beta": 0.9999, "states": 90}

df = pkl.load(open(TEST_FOLDER + "group_4.pkl", "rb"))
result_trans, result_fixp = estimate(init_dict, df, repl_4=True)


@pytest.fixture
def inputs():
    out = dict()
    out["trans_est"] = result_trans["x"]
    out["params_est"] = result_fixp["x"]
    out["trans_ll"] = result_trans["fun"]
    out["cost_ll"] = result_fixp["fun"]
    return out


@pytest.fixture
def outputs():
    out = dict()
    out["trans_base"] = np.loadtxt(TEST_FOLDER + "repl_test_trans.txt")
    out["params_base"] = np.loadtxt(TEST_FOLDER + "repl_test_params.txt")
    out["transition_count"] = np.loadtxt(TEST_FOLDER + "transition_count.txt")
    out["trans_ll"] = 3140.570557
    out["cost_ll"] = 163.585839
    return out


def test_repl_params(inputs, outputs):
    assert_array_almost_equal(inputs["params_est"], outputs["params_base"], decimal=2)


def test_repl_trans(inputs, outputs):
    assert_array_almost_equal(inputs["trans_est"], outputs["trans_base"], decimal=4)


def test_trans_ll(inputs, outputs):
    assert_allclose(inputs["trans_ll"], outputs["trans_ll"])


def test_cost_ll(inputs, outputs):
    assert_allclose(inputs["cost_ll"], outputs["cost_ll"])


def test_transition_count(outputs):
    num_bus = len(df["Bus_ID"].unique())
    num_periods = int(df.shape[0] / num_bus)
    states = df["state"].values.reshape(num_bus, num_periods)
    decisions = df["decision"].values.reshape(num_bus, num_periods)
    space_state = states.max() + 1
    state_count = np.zeros(shape=(space_state, space_state), dtype=int)
    increases = np.zeros(shape=(num_bus, num_periods - 1), dtype=int)
    increases, state_count = np.array(
        create_increases(
            increases, state_count, num_bus, num_periods, states, decisions, repl_4=True
        )
    )
    assert_array_equal(np.bincount(increases.flatten()), outputs["transition_count"])
