import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.estimation import estimate
from ruspy.model_code.demand_function import get_demand


TEST_FOLDER = TEST_RESOURCES_DIR


@pytest.fixture(scope="module")
def inputs():
    out = {}
    disc_fac = 0.9999
    num_states = 90
    num_params = 2
    scale = 1e-3
    lb = np.concatenate((np.full(num_states, -np.inf), np.full(num_params, 0.0)))
    ub = np.concatenate((np.full(num_states, 50.0), np.full(num_params, np.inf)))
    init_dict = {
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
            "maint_cost_func": "linear",
            "cost_scale": scale,
        },
        "optimizer": {
            "approach": "MPEC",
            "algorithm": "LD_SLSQP",
            "gradient": "Yes",
            "params": np.concatenate(
                (np.full(num_states, 0.0), np.array([4.0]), np.ones(num_params - 1))
            ),
            "set_ftol_abs": 1e-15,
            "set_xtol_rel": 1e-15,
            "set_xtol_abs": 1e-3,
            "set_lower_bounds": lb,
            "set_upper_bounds": ub,
        },
    }
    demand_dict = {
        "RC_lower_bound": 2,
        "RC_upper_bound": 13,
        "demand_evaluations": 100,
        "tolerance": 1e-10,
        "num_periods": 12,
        "num_buses": 1,
    }
    df = pd.read_pickle(TEST_FOLDER + "/replication_test/group_4.pkl")
    result_trans, result_fixp = estimate(init_dict, df)
    demand_params = np.concatenate((result_trans["x"], result_fixp["x"][-2:]))
    demand = get_demand(init_dict, demand_dict, demand_params)
    out["demand_estimate"] = demand["demand"].astype(float).to_numpy()
    return out


@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["demand_base"] = np.loadtxt(TEST_FOLDER + "/demand_test/get_demand.txt")
    return out


def test_get_demand(inputs, outputs):
    assert_array_almost_equal(
        inputs["demand_estimate"], outputs["demand_base"], decimal=3
    )
