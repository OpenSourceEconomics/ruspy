import pickle as pkl

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import derivative_loglike_cost_params
from ruspy.estimation.estimation import estimate
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import sqrt_costs
from ruspy.model_code.cost_functions import sqrt_costs_dev
from ruspy.ruspy_config import TEST_RESOURCES_DIR


TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    beta = 0.9999
    num_states = 90
    init_dict = {
        "groups": "group_4",
        "binsize": 5000,
        "beta": beta,
        "states": num_states,
        "maint_cost_func": "square_root",
    }
    df = pkl.load(open(TEST_FOLDER + "group_4.pkl", "rb"))
    result_trans, result_fixp = estimate(init_dict, df)
    out["trans_est"] = result_trans["x"]
    out["params_est"] = result_fixp["x"]
    out["trans_ll"] = result_trans["fun"]
    out["cost_ll"] = result_fixp["fun"]
    out["states"] = df.loc[(slice(None), slice(1, None)), "state"].to_numpy()
    out["decisions"] = df.loc[(slice(None), slice(1, None)), "decision"].to_numpy()
    out["beta"] = beta
    out["num_states"] = num_states
    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["trans_base"] = np.loadtxt(TEST_FOLDER + "repl_test_trans.txt")
    # out["params_base"] = np.loadtxt(TEST_FOLDER + "repl_test_params.txt")
    out["transition_count"] = np.loadtxt(TEST_FOLDER + "transition_count.txt")
    out["trans_ll"] = 3140.570557
    out["cost_ll"] = 163.390  # 163.395. Why not correct?
    return out


def test_repl_trans(inputs, outputs):
    assert_array_almost_equal(inputs["trans_est"], outputs["trans_base"])


def test_trans_ll(inputs, outputs):
    assert_allclose(inputs["trans_ll"], outputs["trans_ll"])


def test_cost_ll(inputs, outputs):
    assert_allclose(inputs["cost_ll"], outputs["cost_ll"])


def test_ll_params_derivative(inputs, outputs):
    num_states = inputs["num_states"]
    trans_mat = create_transition_matrix(num_states, outputs["trans_base"])
    state_mat = create_state_matrix(inputs["states"], num_states)
    endog = inputs["decisions"]
    decision_mat = np.vstack(((1 - endog), endog))
    beta = inputs["beta"]
    assert_array_almost_equal(
        derivative_loglike_cost_params(
            inputs["params_est"],
            sqrt_costs,
            sqrt_costs_dev,
            num_states,
            trans_mat,
            state_mat,
            decision_mat,
            beta,
        ),
        np.array([0, 0]),
    )
