"""
This module contains unit tests, for the most important functions of
ruspy.estimation.estimation_cost_parameters. The values to compare the results with
are saved in resources/estimation_test. The setting of the test is documented in the
inputs section in test module.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ruspy.config import TEST_RESOURCES_DIR
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.test.ranodm_init import random_init


@pytest.fixture
def inputs():
    out = {}
    out["nstates"] = 90
    out["cost_fct"] = lin_cost
    out["params"] = np.array([10, 2])
    out["trans_prob"] = np.array([0.2, 0.3, 0.15, 0.35])
    out["disc_fac"] = 0.9999
    return out


@pytest.fixture
def outputs():
    out = {}
    out["costs"] = np.loadtxt(TEST_RESOURCES_DIR + "estimation_test/myop_cost.txt")
    out["trans_mat"] = np.loadtxt(TEST_RESOURCES_DIR + "estimation_test/trans_mat.txt")
    out["fixp"] = np.loadtxt(TEST_RESOURCES_DIR + "estimation_test/fixp.txt")
    out["choice_probs"] = np.loadtxt(
        TEST_RESOURCES_DIR + "estimation_test/choice_prob.txt"
    )
    return out


def test_cost_func(inputs, outputs):
    assert_array_almost_equal(
        calc_obs_costs(inputs["nstates"], inputs["cost_fct"], inputs["params"], 0.001),
        outputs["costs"],
    )


def test_create_trans_mat(inputs, outputs):
    assert_array_almost_equal(
        create_transition_matrix(inputs["nstates"], inputs["trans_prob"]),
        outputs["trans_mat"],
    )


def test_fixp(inputs, outputs):
    assert_array_almost_equal(
        calc_fixp(outputs["trans_mat"], outputs["costs"], inputs["disc_fac"])[0],
        outputs["fixp"],
    )


def test_choice_probs(inputs, outputs):
    assert_array_almost_equal(
        choice_prob_gumbel(outputs["fixp"], outputs["costs"], inputs["disc_fac"]),
        outputs["choice_probs"],
    )


def test_trans_mat_rows_one():
    rand_dict = random_init()
    control = np.ones(rand_dict["estimation"]["states"])
    assert_array_almost_equal(
        create_transition_matrix(
            rand_dict["estimation"]["states"],
            np.array(rand_dict["simulation"]["known_trans"]),
        ).sum(axis=1),
        control,
    )
