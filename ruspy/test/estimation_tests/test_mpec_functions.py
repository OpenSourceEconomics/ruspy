import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.estimation.mpec import mpec_constraint
from ruspy.estimation.mpec import mpec_constraint_derivative
from ruspy.estimation.mpec import mpec_loglike_cost_params
from ruspy.estimation.mpec import mpec_loglike_cost_params_derivative
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.cost_functions import lin_cost_dev

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture
def inputs():
    out = {}
    df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")
    transition_results = estimate_transitions(df)
    num_states = 90
    endog = df.loc[(slice(None), slice(1, None)), "decision"].to_numpy(int)
    states = df.loc[(slice(None), slice(1, None)), "state"].to_numpy(int)
    out["maint_func"] = lin_cost
    out["maint_func_dev"] = lin_cost_dev
    out["num_states"] = num_states
    out["num_params"] = 2
    out["trans_matrix"] = create_transition_matrix(
        num_states, np.array(transition_results["x"])
    )
    out["state_matrix"] = create_state_matrix(states, out["num_states"])
    out["decision_matrix"] = np.vstack(((1 - endog), endog))
    out["disc_fac"] = 0.9999
    out["scale"] = 0.001
    out["derivative"] = "Yes"
    out["params"] = np.ones(out["num_states"] + out["num_params"])
    return out


@pytest.fixture
def outputs():
    out = {}
    out["mpec_likelihood"] = np.loadtxt(
        TEST_RESOURCES_DIR + "estimation_test/mpec_like.txt"
    )
    out["mpec_constraint"] = np.loadtxt(
        TEST_RESOURCES_DIR + "estimation_test/mpec_constraint.txt"
    )
    out["mpec_like_dev"] = np.loadtxt(
        TEST_RESOURCES_DIR + "estimation_test/mpec_like_dev.txt"
    )
    out["mpec_constr_dev"] = np.loadtxt(
        TEST_RESOURCES_DIR + "estimation_test/mpec_constr_dev.txt"
    )
    return out


def test_mpec_likelihood(inputs, outputs):
    assert_almost_equal(
        mpec_loglike_cost_params(
            inputs["maint_func"],
            inputs["maint_func_dev"],
            inputs["num_states"],
            inputs["num_params"],
            inputs["state_matrix"],
            inputs["decision_matrix"],
            inputs["disc_fac"],
            inputs["scale"],
            inputs["derivative"],
            inputs["params"],
            np.zeros(inputs["num_states"] + inputs["num_params"]),
        ),
        outputs["mpec_likelihood"],
    )


def test_like_dev(inputs, outputs):
    assert_array_almost_equal(
        mpec_loglike_cost_params_derivative(
            inputs["maint_func"],
            inputs["maint_func_dev"],
            inputs["num_states"],
            inputs["num_params"],
            inputs["disc_fac"],
            inputs["scale"],
            inputs["decision_matrix"],
            inputs["state_matrix"],
            inputs["params"],
        ),
        outputs["mpec_like_dev"],
    )


def test_mpec_constraint(inputs, outputs):
    assert_array_almost_equal(
        mpec_constraint(
            inputs["maint_func"],
            inputs["maint_func_dev"],
            inputs["num_states"],
            inputs["num_params"],
            inputs["trans_matrix"],
            inputs["disc_fac"],
            inputs["scale"],
            inputs["derivative"],
            np.zeros(inputs["num_states"]),
            inputs["params"],
            np.zeros(
                (inputs["num_states"], inputs["num_states"] + inputs["num_params"])
            ),
        ),
        outputs["mpec_constraint"],
    )


def test_mpec_constraint_dev(inputs, outputs):
    assert_array_almost_equal(
        mpec_constraint_derivative(
            inputs["maint_func"],
            inputs["maint_func_dev"],
            inputs["num_states"],
            inputs["num_params"],
            inputs["disc_fac"],
            inputs["scale"],
            inputs["trans_matrix"],
            inputs["params"],
        ),
        outputs["mpec_constr_dev"],
    )
