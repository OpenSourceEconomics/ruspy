import numpy as np
import pytest
from numpy.testing import assert_allclose

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.estimation import estimate
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.simulation.simulation import simulate

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    disc_fac = 0.9999
    num_states = 300
    num_buses = 200
    num_periods = 1000
    scale = 0.001
    init_dict = {
        "groups": "group_4",
        "binsize": 5000,
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
            "maint_cost_func": "linear",
            "cost_scale": scale,
        },
        "optimizer": {"approach": "NFXP", "algorithm": "scipy_L-BFGS-B"},
        "simulation": {
            "discount_factor": disc_fac,
            "seed": 123,
            "buses": num_buses,
            "periods": num_periods,
        },
    }
    out["trans_base"] = np.loadtxt(TEST_FOLDER + "repl_test_trans.txt")
    out["params_base"] = np.loadtxt(TEST_FOLDER + "repl_params_linear.txt")
    trans_mat = create_transition_matrix(num_states, out["trans_base"])
    costs = calc_obs_costs(num_states, lin_cost, out["params_base"], scale)
    ev_known = calc_fixp(trans_mat, costs, disc_fac)[0]
    df = simulate(init_dict["simulation"], ev_known, costs, trans_mat)
    result_trans, result_fixp = estimate(init_dict, df)
    out["trans_est"] = result_trans["x"]
    out["params_est"] = result_fixp["x"]
    out["status"] = result_fixp["status"]
    return out


def test_repl_rc(inputs):
    # This is as precise as the paper gets
    assert_allclose(inputs["params_est"][0], inputs["params_base"][0], atol=0.5)


def test_repl_params(inputs):
    # This is as precise as the paper gets
    assert_allclose(inputs["params_est"][1], inputs["params_base"][1], atol=0.25)


def test_repl_trans(inputs):
    assert_allclose(inputs["trans_est"], inputs["trans_base"], atol=1e-2)
