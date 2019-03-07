import pytest
from numpy.testing import assert_allclose
import numpy as np
from ruspy.test.ranodm_init import random_init
from ruspy.simulation.simulation import simulate
from ruspy.simulation.simulation_auxiliary import discount_utility
from ruspy.estimation.estimation_auxiliary import calc_fixp
from ruspy.estimation.estimation_auxiliary import myopic_costs
from ruspy.estimation.estimation_auxiliary import create_transition_matrix
from ruspy.estimation.estimation_auxiliary import lin_cost


@pytest.fixture
def inputs():
    constraints = {'PERIODS': 70000, 'BUSES': 100, 'BETA': 0.9999}
    return constraints


def test_regression_simulation(inputs):
    """

    :param inputs:
    :return:
    """
    init_dict = random_init(inputs)
    df, unobs, utilities, num_states = simulate(init_dict['simulation'])
    num_buses = init_dict['simulation']['buses']
    num_periods = init_dict['simulation']['periods']
    beta = init_dict['simulation']['beta']
    params = np.array(init_dict['simulation']['params'])
    probs = np.array(init_dict['simulation']['probs'])
    v_disc_ = [0., 0.]
    v_disc = discount_utility(v_disc_, num_buses, num_periods, 2, utilities, beta)
    trans_mat = create_transition_matrix(num_states, probs)
    costs = myopic_costs(num_states, lin_cost, params)
    v_calc = calc_fixp(num_states, trans_mat, costs, beta)
    un_ob_av = 0
    for bus in range(num_buses):
        un_ob_av += unobs[bus, 0, 0]
    un_ob_av = un_ob_av/num_buses
    assert_allclose(v_disc[1] / (v_calc[0] + un_ob_av), 1, rtol=1e-02)
