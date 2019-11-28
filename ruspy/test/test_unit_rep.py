import pickle as pkl

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import derivative_loglike_cost_params
from ruspy.estimation.est_cost_params import loglike_cost_params
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import lin_cost
from ruspy.ruspy_config import TEST_RESOURCES_DIR

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    df = pkl.load(open(TEST_FOLDER + "group_4.pkl", "rb"))
    out["num_bus"] = len(df["Bus_ID"].unique())
    out["num_periods"] = int(df.shape[0] / out["num_bus"])
    out["states"] = df["state"].to_numpy()
    out["decisions"] = df["decision"].to_numpy()
    out["beta"] = 0.9999
    out["num_states"] = 90
    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["trans_base"] = np.loadtxt(TEST_FOLDER + "repl_test_trans.txt")
    out["params_base"] = np.loadtxt(TEST_FOLDER + "repl_test_params.txt")
    out["transition_count"] = np.loadtxt(TEST_FOLDER + "transition_count.txt")
    out["trans_ll"] = 3140.570557
    out["cost_ll"] = 163.585839
    return out


def test_unit_ll_params(inputs, outputs):
    num_states = inputs["num_states"]
    trans_mat = create_transition_matrix(num_states, outputs["trans_base"])
    state_mat = create_state_matrix(inputs["states"], num_states)
    endog = inputs["decisions"]
    decision_mat = np.vstack(((1 - endog), endog))
    beta = inputs["beta"]
    assert_allclose(
        loglike_cost_params(
            outputs["params_base"],
            lin_cost,
            num_states,
            trans_mat,
            state_mat,
            decision_mat,
            beta,
        ),
        outputs["cost_ll"],
    )


def test_ll_params_derivative(inputs, outputs):
    num_states = inputs["num_states"]
    trans_mat = create_transition_matrix(num_states, outputs["trans_base"])
    state_mat = create_state_matrix(inputs["states"], num_states)
    endog = inputs["decisions"]
    decision_mat = np.vstack(((1 - endog), endog))
    beta = inputs["beta"]
    assert_array_almost_equal(
        derivative_loglike_cost_params(
            outputs["params_base"],
            lin_cost,
            num_states,
            trans_mat,
            state_mat,
            decision_mat,
            beta,
        ),
        np.array([0, 0]),
    )
