"""
This module contains a regression test for the simulation, discounting and some
estimation functions.
"""

import pytest
from numpy.testing import assert_allclose
import numpy as np
from ruspy.test.ranodm_init import random_init
from ruspy.simulation.simulation import simulate
from ruspy.plotting.discounting import discount_utility
from ruspy.estimation.estimation_cost_parameters import calc_fixp
from ruspy.estimation.estimation_cost_parameters import myopic_costs
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix
from ruspy.estimation.estimation_cost_parameters import lin_cost


@pytest.fixture
def inputs():
    """
    The test will simulate a dataset. The constraints defined below, assure that
    there are enough number of relevant ovservations to ensure convergence.
    :return: A dictionary with the constraints.
    """
    constraints = {'PERIODS': 70000, 'BUSES': 100, 'BETA': 0.9999}
    return constraints


def test_regression_simulation(inputs):
    """
    This test first draws a random dictionary with the constraints defined in the
    inputs. It then simulates a dataset and the compares the discounted utility
    average over all buses, with the theoretical expected value calculated by the
    NFXP.
    :param inputs: A dictionary with constraints for the random dictionary.
    :return: The True/False value of the test.
    """
    init_dict = random_init(inputs)
    df, unobs, utilities, num_states = simulate(init_dict['simulation'])
    num_buses = init_dict['simulation']['buses']
    num_periods = init_dict['simulation']['periods']
    beta = init_dict['simulation']['beta']
    params = np.array(init_dict['simulation']['params'])
    probs = np.array(init_dict['simulation']['probs'])
    v_disc_ = [0., 0.]
    v_disc = discount_utility(v_disc_, num_buses, num_periods, num_periods, utilities,
                              beta)
    trans_mat = create_transition_matrix(num_states, probs)
    costs = myopic_costs(num_states, lin_cost, params)
    v_calc = calc_fixp(num_states, trans_mat, costs, beta)
    un_ob_av = 0
    for bus in range(num_buses):
        un_ob_av += unobs[bus, 0, 0]
    un_ob_av = un_ob_av/num_buses
    assert_allclose(v_disc[1] / (v_calc[0] + un_ob_av), 1, rtol=1e-02)
