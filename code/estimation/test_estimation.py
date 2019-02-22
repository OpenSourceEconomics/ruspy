import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest


from estimation.estimation_auxiliary import myopic_costs
from estimation.estimation_auxiliary import lin_cost
from estimation.estimation_auxiliary import create_transition_matrix
from estimation.estimation_auxiliary import calc_fixp
from estimation.estimation_auxiliary import choice_prob

@pytest.fixture
def inputs():
    out = dict()
    out['nstates'] = 90
    out['cost_fct'] = lin_cost
    out['params'] = [10,  2]
    out['trans_prob'] = np.array([0.2, 0.3, 0.15, 0.35])
    out['beta'] = 0.9999
    return out

@pytest.fixture
def outputs():
    out = dict()
    out['myop_costs'] = np.loadtxt('test_cases/myop_costs_90_10_2.txt')
    out['trans_mat'] = np.loadtxt('test_cases/trans_mat_90_2_3_15_35.txt')
    out['fixp'] = np.loadtxt('test_cases/fixp_90_2_3_15_35_lin_10_2_9999.txt')
    out['choice_probs'] = np.loadtxt('test_cases/choice_probs_90_10_2_9999.txt')
    return out


def test_myopic_costs(inputs, outputs):
    assert_array_equal(myopic_costs(inputs['nstates'], inputs['cost_fct'], inputs['params']),
                       outputs['myop_costs'])


def test_create_trans_mat(inputs, outputs):
    assert_array_equal(create_transition_matrix(inputs['nstates'], inputs['trans_prob']),
                       outputs['trans_mat'])


def test_calc_fixp(inputs, outputs):
    assert_array_almost_equal(calc_fixp(inputs['nstates'], outputs['trans_mat'], inputs['cost_fct'],
                                  inputs['params'], inputs['beta'], threshold=1e-6),
                       outputs['fixp'])


def test_choice_probs(inputs, outputs):
    assert_array_equal(choice_prob(outputs['fixp'], inputs['params'], inputs['beta']),
                       outputs['choice_probs'])




