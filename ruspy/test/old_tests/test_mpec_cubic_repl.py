# import numpy as np
# import pandas as pd
# import pytest
# from numpy.testing import assert_almost_equal
# from numpy.testing import assert_array_almost_equal
#
# from ruspy.config import TEST_RESOURCES_DIR
# from ruspy.estimation.estimation import estimate
#
#
# TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"
#
#
# @pytest.fixture(scope="module")
# def inputs():
#     out = {}
#     disc_fac = 0.9999
#     num_states = 90
#     scale = 1e-8
#     num_params = 4
#     lb = np.concatenate((np.full(num_states, -np.inf), np.full(num_params, 0.0)))
#     ub = np.concatenate((np.full(num_states, 50.0), np.full(num_params, np.inf)))
#     init_dict = {
#         "model_specifications": {
#             "discount_factor": disc_fac,
#             "number_states": num_states,
#             "maint_cost_func": "cubic",
#             "cost_scale": scale,
#         },
#         "optimizer": {
#             "approach": "MPEC",
#             "algorithm": "LD_SLSQP",
#             "derivative": "Yes",
#             "params": np.concatenate(
#                 (np.full(num_states, 0.0), np.array([4.0]), np.ones(num_params - 1))
#             ),
#             "set_ftol_abs": 1e-15,
#             "set_xtol_rel": 1e-15,
#             "set_xtol_abs": 1e-3,
#             "set_lower_bounds": lb,
#             "set_upper_bounds": ub,
#         },
#     }
#     df = pd.read_pickle(TEST_FOLDER + "group_4.pkl")
#     result_trans, result_fixp = estimate(init_dict, df)
#     out["params_est"] = result_fixp["x"][num_states:].round(8)
#     out["cost_ll"] = result_fixp["fun"]
#     out["status"] = result_fixp["status"]
#     return out
#
#
# @pytest.fixture(scope="module")
# def outputs():
#     out = {}
#     out["params_base"] = np.array([10.07494318, 229309.298, 0.0, 0.0])
#     out["cost_ll"] = 163.584283  # 162.885
#     return out
#
#
# def test_repl_params(inputs, outputs):
#     assert_array_almost_equal(inputs["params_est"], outputs["params_base"], decimal=3)
#
#
# def test_cost_ll(inputs, outputs):
#     assert_almost_equal(inputs["cost_ll"], outputs["cost_ll"], decimal=3)
#
#
# def test_success(inputs):
#     assert inputs["status"] is True
