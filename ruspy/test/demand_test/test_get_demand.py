import numpy as np
import pandas as pd
import pytest
from estimagic import minimize
from numpy.testing import assert_array_almost_equal

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.criterion_function import get_criterion_function
from ruspy.model_code.demand_function import get_demand


TEST_FOLDER = TEST_RESOURCES_DIR


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
    criterion_func, criterion_dev, result_trans = get_criterion_function(init_dict, df)
    result_fixp = minimize(
        criterion=criterion_func,
        params=np.zeros(2, dtype=float),
        algorithm="scipy_lbfgsb",
        derivative=criterion_dev,
    )
    demand_params = np.concatenate((result_trans["x"], result_fixp.params))
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
