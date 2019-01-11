import numpy as np
from numpy.testing import assert_array_equal
import pytest
from estimation_auxiliary import myopic_costs
from estimation_auxiliary import lin_cost

@pytest.fixture
def inputs():
    out = dict()
    out['nstates'] = 90
    out['cost_fct'] = lin_cost
    out['params'] = [10.07778082,  2.29416295]
    return out

@pytest.fixture
def outputs():
    out = dict()
    out['myop_costs'] = np.loadtxt('test_cases/myop_costs_90.txt')
    return out

def test_myopic_costs(inputs, outputs):
    assert_array_equal(myopic_costs(inputs['nstates'], inputs['cost_fct'], inputs['params']),
                       outputs['myop_costs'])