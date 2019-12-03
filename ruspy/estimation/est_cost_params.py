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
from ruspy.model_code.fix_point_alg import cont_op_dev_wrt_fixp
from ruspy.model_code.fix_point_alg import contr_op_dev_wrt_params
from ruspy.model_code.fix_point_alg import contr_op_dev_wrt_rc


def loglike_cost_params(
    params,
    maint_func,
    maint_func_dev,
    num_states,
    trans_mat,
    state_mat,
    decision_mat,
    beta,
    scale=0.001,
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
    costs = calc_obs_costs(num_states, maint_func, params, scale=scale)
    ev = calc_fixp(trans_mat, costs, beta)
    p_choice = choice_prob_gumbel(ev, costs, beta)
    ll_prob = np.log(np.dot(p_choice.T, state_mat))
    return -np.sum(decision_mat * ll_prob)


def derivative_loglike_cost_params(
    params,
    maint_func,
    maint_func_dev,
    num_states,
    trans_mat,
    state_mat,
    decision_mat,
    beta,
    scale=0.001,
):

    costs = calc_obs_costs(num_states, maint_func, params)
    ev = calc_fixp(trans_mat, costs, beta)
    cost_dev = maint_func_dev(num_states, scale=scale)
    t_prime = cont_op_dev_wrt_fixp(ev, trans_mat, costs, beta)

    p_choice = choice_prob_gumbel(ev, costs, beta)

    partial_fixp_wrt_params = contr_op_dev_wrt_params(
        trans_mat, p_choice[:, 0], maint_func_dev
    )
    partial_fixp_wrt_rc = contr_op_dev_wrt_rc(trans_mat, p_choice[:, 0])

    partial_ev_wrt_params = np.linalg.lstsq(
        np.eye(num_states) - t_prime, partial_fixp_wrt_params, rcond=None
    )[0]
    partial_ev_wrt_rc = np.linalg.lstsq(
        np.eye(num_states) - t_prime, partial_fixp_wrt_rc, rcond=None
    )[0]

    dev_value_maint_params = (
        cost_dev[0]
        - beta * partial_ev_wrt_params[0]
        + beta * partial_ev_wrt_params
        - cost_dev
    )

    ll_values_params = np.empty_like(p_choice)

    ll_values_params[:, 0] = np.multiply(1 - p_choice[:, 0], dev_value_maint_params)
    ll_values_params[:, 1] = np.multiply(1 - p_choice[:, 1], -dev_value_maint_params)

    dev_value_maint_rc = 1 + beta * partial_ev_wrt_rc - beta * partial_ev_wrt_rc[0]

    ll_values_rc = np.empty_like(p_choice)

    ll_values_rc[:, 0] = np.multiply(1 - p_choice[:, 0], dev_value_maint_rc)
    ll_values_rc[:, 1] = np.multiply(1 - p_choice[:, 1], -dev_value_maint_rc)

    ll_dev_params = -np.sum(decision_mat * np.dot(ll_values_params.T, state_mat))
    ll_dev_rc = -np.sum(decision_mat * np.dot(ll_values_rc.T, state_mat))

    dev = np.array([ll_dev_rc, ll_dev_params])
    print(dev)
    return dev


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
