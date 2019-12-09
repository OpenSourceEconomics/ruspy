"""
This module contains functions for estimating the parameters shaping the cost
function. Therefore it also contains the heart of this project. The python
implementation of fix point algorithm developed by John Rust.
"""
import numba
import numpy as np

from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.model_code.fix_point_alg import contr_op_dev_wrt_params
from ruspy.model_code.fix_point_alg import contr_op_dev_wrt_rc
from ruspy.model_code.fix_point_alg import solve_equ_system_fixp


ev_intermed = None
current_params = None


def loglike_cost_params(
    params,
    maint_func,
    maint_func_dev,
    num_states,
    trans_mat,
    state_mat,
    decision_mat,
    beta,
    scale,
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
    costs = calc_obs_costs(num_states, maint_func, params, scale)

    ev = get_ev(params, trans_mat, costs, beta)

    p_choice = choice_prob_gumbel(ev, costs, beta)
    ll = like_hood_data(np.log(p_choice), decision_mat, state_mat)
    return ll


def derivative_loglike_cost_params(
    params,
    maint_func,
    maint_func_dev,
    num_states,
    trans_mat,
    state_mat,
    decision_mat,
    beta,
    scale,
):
    dev = np.zeros_like(params)
    costs = calc_obs_costs(num_states, maint_func, params, scale)

    ev = get_ev(params, trans_mat, costs, beta)

    p_choice = choice_prob_gumbel(ev, costs, beta)
    maint_cost_dev = maint_func_dev(num_states, scale)

    lh_values_rc = like_hood_vaules_rc(ev, costs, p_choice, trans_mat, beta)
    like_dev_rc = like_hood_data(lh_values_rc, decision_mat, state_mat)
    dev[0] = like_dev_rc

    for i in range(len(params) - 1):
        if len(params) == 2:
            cost_dev_param = maint_cost_dev
        else:
            cost_dev_param = maint_cost_dev[:, i]

        like_values_params = like_hood_values_param(
            ev, costs, p_choice, trans_mat, cost_dev_param, beta
        )
        like_dev_params = like_hood_data(like_values_params, decision_mat, state_mat)
        dev[i + 1] = like_dev_params

    return dev


def get_ev(params, trans_mat, costs, beta):
    global ev_intermed
    global current_params
    if (ev_intermed is not None) & np.array_equal(current_params, params):
        ev = ev_intermed
        ev_intermed = None
    else:
        ev = calc_fixp(trans_mat, costs, beta)
        ev_intermed = ev
        current_params = params
    return ev


def like_hood_values_param(ev, costs, p_choice, trans_mat, cost_dev, beta):
    dev_contr_op_params = contr_op_dev_wrt_params(trans_mat, p_choice[:, 0], cost_dev)
    dev_ev_params = solve_equ_system_fixp(
        dev_contr_op_params, ev, trans_mat, costs, beta
    )
    dev_value_maint_params = chain_rule_param(cost_dev, dev_ev_params, beta)
    lh_values_param = like_hood_dev_values(p_choice, dev_value_maint_params)
    return lh_values_param


def like_hood_vaules_rc(ev, costs, p_choice, trans_mat, beta):
    dev_contr_op_rc = contr_op_dev_wrt_rc(trans_mat, p_choice[:, 0])
    dev_ev_rc = solve_equ_system_fixp(dev_contr_op_rc, ev, trans_mat, costs, beta)
    dev_value_maint_rc = 1 + beta * dev_ev_rc - beta * dev_ev_rc[0]
    lh_values_rc = like_hood_dev_values(p_choice, dev_value_maint_rc)
    return lh_values_rc


def chain_rule_param(cost_dev, dev_ev_param, beta):
    chain_value = cost_dev[0] - beta * dev_ev_param[0] + beta * dev_ev_param - cost_dev
    return chain_value


def like_hood_data(l_values, decision_mat, state_mat):
    return -np.sum(decision_mat * np.dot(l_values.T, state_mat))


def like_hood_dev_values(p_choice, dev_values):
    l_values = np.empty_like(p_choice)
    l_values[:, 0] = np.multiply(1 - p_choice[:, 0], dev_values)
    l_values[:, 1] = np.multiply(1 - p_choice[:, 1], -dev_values)

    return l_values


@numba.jit(nopython=True)
def create_state_matrix(states, num_states):
    """
    This function constructs a auxiliary matrix for the likelihood.

    :param states:      A numpy array containing the observed states.
    :param num_states:  The size of the state space s.
    :type num_states:   int

    :return:            A two dimensional numpy array containing n x s matrix
                        with TRUE in each row at the column in which the bus was in
                        that observation.
    """
    num_obs = states.shape[0]
    state_mat = np.full((num_states, num_obs), 0.0)
    for i, value in enumerate(states):
        state_mat[value, i] = 1.0
    return state_mat
