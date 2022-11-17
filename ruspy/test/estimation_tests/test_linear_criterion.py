"""
test get_criterion_function for NFXP and linear cost function
"""
"""
This module contains tests for the data and estimation code of the ruspy project. The
settings for this tests is specified in resources/replication_test/init_replication.yml.
The test first reads the original data, then processes the data to a pandas DataFrame
suitable for the estimation process. After estimating all the relevant parameters,
they are compared to the results, from the paper. As this test runs the complete
data_reading, data processing and runs several times the NFXP it is the one with the
longest test time.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.criterion_function import get_criterion_function

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"


@pytest.fixture(scope="module")
def inputs():
    out = {}
    disc_fac = 0.9999
    num_states = 90
    init_dict = {
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
        },
        "method": "NFXP",
        "alg_details": {},
    }

    df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")

    out["input data"] = df
    out["init_dict"] = init_dict
    return out


# true outputs
@pytest.fixture(scope="module")
def outputs():
    out = {}
    out["cost_ll_linear"] = 163.584
    out["cost_ll_cubic"] = 164.632939  # 162.885

    return out


TEST_SPECIFICATIONS = [("linear", 1e-3, 2), ("cubic", 1e-8, 4)]


@pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
def test_criterion_function(inputs, outputs, specification):
    cost_func_name, scale, num_params = specification

    df = inputs["input data"]
    init_dict = inputs["init_dict"]

    init_dict["model_specifications"]["maint_cost_func"] = cost_func_name
    init_dict["model_specifications"]["cost_scale"] = scale

    criterion_func, criterion_dev = get_criterion_function(init_dict, df)

    assert_array_almost_equal(
        criterion_func(np.loadtxt(TEST_FOLDER + f"repl_params_{cost_func_name}.txt")),
        outputs["cost_ll_" + cost_func_name],
        decimal=3,
    )


#
# @pytest.mark.parametrize("specification", TEST_SPECIFICATIONS)
# def test_criterion_derivative(inputs, outputs, specification):
#     criterion_derivative = inputs["criterion_derivative"]
#     true_params = outputs["params_base"]
#     assert_array_almost_equal(
#         criterion_derivative(true_params),
#         np.array([0, 0]),
#         decimal=2,
#     )
