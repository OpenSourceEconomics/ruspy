import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import derivative_loglike_cost_params
from ruspy.estimation.estimation import estimate
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import cubic_costs
from ruspy.model_code.cost_functions import cubic_costs_dev
from ruspy.ruspy_config import TEST_RESOURCES_DIR

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    disc_fac = 0.9999
    num_states = 90
    scale = 1e-8
    init_dict = {
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
            "maint_cost_func": "cubic",
            "cost_scale": scale,
        },
        "optimizer": {
            "optimizer_name": "BFGS",
            "use_gradient": "yes",
            "use_search_bounds": "no",
        },
    }
    df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")
    result_trans, result_fixp = estimate(init_dict, df)
    out["trans_est"] = result_trans["x"]
    out["params_est"] = result_fixp["x"]
    out["trans_ll"] = result_trans["fun"]
    out["cost_ll"] = result_fixp["fun"]
    out["states"] = df.loc[:, "state"].to_numpy(int)
    out["decisions"] = df.loc[:, "decision"].to_numpy(int)
    out["disc_fac"] = disc_fac
    out["num_states"] = num_states
    out["scale"] = scale
    out["message"] = result_fixp["message"]
    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["trans_base"] = np.loadtxt(TEST_FOLDER + "repl_test_trans.txt")
    out["params_base"] = np.loadtxt(TEST_FOLDER + "repl_params_cubic.txt")
    out["transition_count"] = np.loadtxt(TEST_FOLDER + "transition_count.txt")
    out["trans_ll"] = 3140.570557
    out["cost_ll"] = 163.624044  # 162.885
    return out


def test_repl_params(inputs, outputs):
    assert_array_almost_equal(inputs["params_est"], outputs["params_base"], decimal=3)


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
    disc_fac = inputs["disc_fac"]
    assert_array_almost_equal(
        derivative_loglike_cost_params(
            inputs["params_est"],
            cubic_costs,
            cubic_costs_dev,
            num_states,
            trans_mat,
            state_mat,
            decision_mat,
            disc_fac,
            inputs["scale"],
            {},
        ),
        np.array([0, 0, 0, 0]),
        decimal=2,
    )


def test_success(inputs):
    assert inputs["message"] == "Optimization terminated successfully."
