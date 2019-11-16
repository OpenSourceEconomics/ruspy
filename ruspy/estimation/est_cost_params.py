"""
This module contains functions for estimating the parameters shaping the cost
function. Therefore it also contains the heart of this project. The python
implementation of fix point algorithm developed by John Rust.
"""
import numba
import numpy as np

from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.cost_functions import lin_cost_dev
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.model_code.fix_point_alg import fixp_point_dev


def loglike_cost_params(
    params, maint_func, num_states, trans_mat, state_mat, decision_mat, beta,
):
    """
    This is the logliklihood function for the estimation of the cost parameters.

    :param params:       A numpy array containing the parameters shaping the cost
                         function.
    :param maint_func:   The maintenance cost function. Only linear implemented so
                         far.
    :param num_states:   The size of the state space s.
    :type num_states:    int
    :param trans_mat:    A two dimensional numpy array contlinaining a s x s markov
                         transition matrix.
    :param state_mat:    A two dimensional numpy array containing n x s matrix
                         with TRUE in each row at the column in which the bus was in
                         that observation.
    :param decision_mat: A two dimensional numpy array contaning  a n x 2 vector
                         with 1 in the first row for maintaining and 1 in the second
                         for replacement.
    :param beta:         The discount factor.
    :type beta:          float

    :return: The negative loglikelihood value for minimizing the objective function.
    """
    costs = calc_obs_costs(num_states, maint_func, params)
    ev = calc_fixp(trans_mat, costs, beta)
    p_choice = choice_prob_gumbel(ev, costs, beta)
    ll_prob = np.log(np.dot(p_choice.T, state_mat))
    return -np.sum(decision_mat * ll_prob)


def derivative_loglike_cost_params(
    params, maint_func, num_states, trans_mat, state_mat, decision_mat, beta,
):
    costs = calc_obs_costs(num_states, maint_func, params)
    ev = calc_fixp(trans_mat, costs, beta)
    cost_dev = lin_cost_dev(num_states)
    p_choice = choice_prob_gumbel(ev, costs, beta)
    ev_dev = fixp_point_dev(ev, trans_mat, costs, beta)
    ll_values = np.multiply(1 - p_choice, cost_dev + beta * ev_dev - ev_dev[0])
    ll_dev = np.dot(ll_values.T, state_mat)
    return np.sum(decision_mat * ll_dev)


@numba.jit(nopython=True)
def create_state_matrix(states, num_states, num_obs):
    """
    This function constructs a auxiliary matrix for the likelihood.

    :param states:      A numpy array containing the observed states.
    :param num_states:  The size of the state space s.
    :type num_states:   int
    :param num_obs:     The total number of observations n.
    :type num_obs:      int

    :return:            A two dimensional numpy array containing n x s matrix
                        with TRUE in each row at the column in which the bus was in
                        that observation.
    """
    state_mat = np.full((num_states, num_obs), 0.0)
    for i, value in enumerate(states):
        state_mat[value, i] = 1.0
    return state_mat
