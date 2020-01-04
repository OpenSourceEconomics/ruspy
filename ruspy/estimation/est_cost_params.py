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
    disc_fac,
    scale,
):
    """
    This is the logliklihood function for the estimation of the cost parameters.


    Parameters
    ----------
    params : numpy.array
        see :ref:`params`
    maint_func: :func:
        see :ref: `maint_func`
    num_states : int
        The size of the state space.
    disc_fac : numpy.float
        see :ref:`disc_fac`
    trans_mat : numpy.array
        see :ref:`trans_mat`
    state_mat : numpy.array
        see :ref:`state_mat`
    decision_mat : numpy.array
        see :ref:`decision_mat`

    Returns
    -------
    log_like : numpy.float
        The negative log-likelihood value of the cost parameters.


    """
    costs = calc_obs_costs(num_states, maint_func, params, scale)

    ev = get_ev(params, trans_mat, costs, disc_fac)

    p_choice = choice_prob_gumbel(ev, costs, disc_fac)
    log_like = like_hood_data(np.log(p_choice), decision_mat, state_mat)
    return log_like


def derivative_loglike_cost_params(
    params,
    maint_func,
    maint_func_dev,
    num_states,
    trans_mat,
    state_mat,
    decision_mat,
    disc_fac,
    scale,
):
    """

    :param params:
    :param maint_func:
    :param maint_func_dev:
    :param num_states:
    :param trans_mat:
    :param state_mat:
    :param decision_mat:
    :param disc_fac:
    :param scale:
    :return:
    """
    dev = np.zeros_like(params)
    costs = calc_obs_costs(num_states, maint_func, params, scale)

    ev = get_ev(params, trans_mat, costs, disc_fac)

    p_choice = choice_prob_gumbel(ev, costs, disc_fac)
    maint_cost_dev = maint_func_dev(num_states, scale)

    lh_values_rc = like_hood_vaules_rc(ev, costs, p_choice, trans_mat, disc_fac)
    like_dev_rc = like_hood_data(lh_values_rc, decision_mat, state_mat)
    dev[0] = like_dev_rc

    for i in range(len(params) - 1):
        if len(params) == 2:
            cost_dev_param = maint_cost_dev
        else:
            cost_dev_param = maint_cost_dev[:, i]

        like_values_params = like_hood_values_param(
            ev, costs, p_choice, trans_mat, cost_dev_param, disc_fac
        )
        like_dev_params = like_hood_data(like_values_params, decision_mat, state_mat)
        dev[i + 1] = like_dev_params

    return dev


def get_ev(params, trans_mat, costs, disc_fac):
    global ev_intermed
    global current_params
    if (ev_intermed is not None) & np.array_equal(current_params, params):
        ev = ev_intermed
        ev_intermed = None
    else:
        ev = calc_fixp(trans_mat, costs, disc_fac)
        ev_intermed = ev
        current_params = params
    return ev


def like_hood_values_param(ev, costs, p_choice, trans_mat, cost_dev, disc_fac):
    dev_contr_op_params = contr_op_dev_wrt_params(trans_mat, p_choice[:, 0], cost_dev)
    dev_ev_params = solve_equ_system_fixp(
        dev_contr_op_params, ev, trans_mat, costs, disc_fac
    )
    dev_value_maint_params = chain_rule_param(cost_dev, dev_ev_params, disc_fac)
    lh_values_param = like_hood_dev_values(p_choice, dev_value_maint_params)
    return lh_values_param


def like_hood_vaules_rc(ev, costs, p_choice, trans_mat, disc_fac):
    dev_contr_op_rc = contr_op_dev_wrt_rc(trans_mat, p_choice[:, 0])
    dev_ev_rc = solve_equ_system_fixp(dev_contr_op_rc, ev, trans_mat, costs, disc_fac)
    dev_value_maint_rc = 1 + disc_fac * dev_ev_rc - disc_fac * dev_ev_rc[0]
    lh_values_rc = like_hood_dev_values(p_choice, dev_value_maint_rc)
    return lh_values_rc


def chain_rule_param(cost_dev, dev_ev_param, disc_fac):
    chain_value = (
        cost_dev[0] - disc_fac * dev_ev_param[0] + disc_fac * dev_ev_param - cost_dev
    )
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
