"""
This module contains functions for estimating the parameters shaping the cost
function. Therefore it also contains the heart of this project. The python
implementation of fix point algorithm developed by John Rust.
"""
import numpy as np

from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.fix_point_alg import calc_fixp


def loglike_opt_rule(
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
    :param trans_mat:    A two dimensional numpy array containing a s x s markov
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
