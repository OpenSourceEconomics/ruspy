"""
This module contains unit tests, for the most important functions of
ruspy.estimation.estimation_cost_parameters. The values to compare the results with
are saved in resources/estimation_test. The setting of the test is documented in the
inputs section in test module.
"""

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest


from ruspy.estimation.estimation_cost_parameters import (
    cost_func,
    create_transition_matrix,
    calc_fixp,
    choice_prob,
    lin_cost,
    converge_choice,
)
from ruspy.test.ranodm_init import random_init
from ruspy.ruspy_config import TEST_RESOURCES_DIR


@pytest.fixture
def inputs():
    out = dict()
    out["nstates"] = 90
    out["cost_fct"] = lin_cost
    out["params"] = np.array([10, 2])
    out["trans_prob"] = np.array([0.2, 0.3, 0.15, 0.35])
    out["beta"] = 0.9999
    return out


@pytest.fixture
def outputs():
    out = dict()
    out["costs"] = np.loadtxt(TEST_RESOURCES_DIR + "estimation_test/myop_cost.txt")
    out["trans_mat"] = np.loadtxt(TEST_RESOURCES_DIR + "estimation_test/trans_mat.txt")
    out["fixp"] = np.loadtxt(TEST_RESOURCES_DIR + "estimation_test/fixp.txt")
    out["choice_probs"] = np.loadtxt(
        TEST_RESOURCES_DIR + "estimation_test/choice_prob.txt"
    )
    return out


def test_cost_func(inputs, outputs):
    assert_array_almost_equal(
        cost_func(inputs["nstates"], inputs["cost_fct"], inputs["params"]),
        outputs["costs"],
    )


def test_create_trans_mat(inputs, outputs):
    assert_array_almost_equal(
        create_transition_matrix(inputs["nstates"], inputs["trans_prob"]),
        outputs["trans_mat"],
    )


def test_calc_fixp(inputs, outputs):
    assert_array_almost_equal(
        calc_fixp(
            inputs["nstates"], outputs["trans_mat"], outputs["costs"], inputs["beta"]
        ),
        outputs["fixp"],
    )


def test_choice_probs(inputs, outputs):
    assert_array_almost_equal(
        choice_prob(outputs["fixp"], outputs["costs"], inputs["beta"]),
        outputs["choice_probs"],
    )


def test_choice_convergence(inputs, outputs):
    assert_array_almost_equal(
        converge_choice(
            inputs["nstates"], outputs["trans_mat"], outputs["costs"], inputs["beta"]
        ),
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
