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
    alg_details,
):
    """
    This is the logliklihood function for the estimation of the cost parameters.


    Parameters
    ----------
    params : numpy.array
        see :ref:`params`
    maint_func: func
        see :ref:`maint_func`
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
    params = params["value"].to_numpy()
    costs = calc_obs_costs(num_states, maint_func, params, scale)

    ev = get_ev(params, trans_mat, costs, disc_fac, alg_details)

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
    alg_details,
):
    """
    This is the derivative of logliklihood function of the cost
    parameter estimation with respect to all cost parameters.


    Parameters
    ----------
    params : numpy.array
        see :ref:`params`
    maint_func: func
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
    dev : numpy.array
        A dim(params) one dimensional numpy array containing the derivative of the
        cost log-likelihood function for every cost parameter.


    """
    params = params["value"].to_numpy()
    dev = np.zeros_like(params)
    obs_costs = calc_obs_costs(num_states, maint_func, params, scale)

    ev = get_ev(params, trans_mat, obs_costs, disc_fac, alg_details)

    p_choice = choice_prob_gumbel(ev, obs_costs, disc_fac)
    maint_cost_dev = maint_func_dev(num_states, scale)

    lh_values_rc = like_hood_vaules_rc(ev, obs_costs, p_choice, trans_mat, disc_fac)
    like_dev_rc = like_hood_data(lh_values_rc, decision_mat, state_mat)
    dev[0] = like_dev_rc

    for i in range(len(params) - 1):
        if len(params) == 2:
            cost_dev_param = maint_cost_dev
        else:
            cost_dev_param = maint_cost_dev[:, i]

        log_like_values_params = log_like_values_param(
            ev, obs_costs, p_choice, trans_mat, cost_dev_param, disc_fac
        )
        dev[i + 1] = like_hood_data(log_like_values_params, decision_mat, state_mat)

    return dev


def get_ev(params, trans_mat, obs_costs, disc_fac, alg_details):
    """
    A auxiliary function, which allows the log-likelihood function as well as her
    derivative to share the same fix point and to avoid the need to execute the
    computation double.

    Parameters
    ----------
    params : numpy.array
        see :ref:`params`
    trans_mat : numpy.array
        see :ref:`trans_mat`
    obs_costs : numpy.array
        see :ref:`costs`
    disc_fac : numpy.float
        see :ref:`disc_fac`

    Returns
    -------
    ev : numpy.array
        see :ref:`ev`

    """
    global ev_intermed
    global current_params
    if (ev_intermed is not None) & np.array_equal(current_params, params):
        ev = ev_intermed
        ev_intermed = None
    else:
        ev = calc_fixp(trans_mat, obs_costs, disc_fac, **alg_details)
        ev_intermed = ev
        current_params = params
    return ev


def log_like_values_param(ev, costs, p_choice, trans_mat, cost_dev, disc_fac):
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
    This function constructs a auxiliary matrix for the log-likelihood of the cost
    parameters.

    Parameters
    ----------
    states : numpy.array
        All mileage state observations.
    num_states : int
        The size of the state space.

    Returns
    -------
    state_mat : numpy.array
        see :ref:`state_mat`

    """
    num_obs = states.shape[0]
    state_mat = np.full((num_states, num_obs), 0.0)
    for i, value in enumerate(states):
        state_mat[value, i] = 1.0
    return state_mat
