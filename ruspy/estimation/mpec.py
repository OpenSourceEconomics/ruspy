"""
This module contains all the key functions used to estimate the model using MPEC.
"""
import numpy as np

from ruspy.estimation.nfxp import like_hood_data
from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from ruspy.model_code.cost_functions import calc_obs_costs


def mpec_loglike_cost_params(
    mpec_params,
    maint_func,
    maint_func_dev,
    num_states,
    disc_fac,
    scale,
    decision_mat,
    state_mat,
):
    """
    Calculate the negative partial log likelihood for MPEC depending on cost parameters
    as well as the discretized expected values.

    Parameters
    ----------
    mpec_params : numpy.ndarray
        see :ref:`mpec_params`
    maint_func: func
        see :ref:`maint_func`
    num_states : int
        The size of the state space.
    state_mat : numpy.ndarray
        see :ref:`state_mat`
    decision_mat : numpy.ndarray
        see :ref:`decision_mat`
    disc_fac : numpy.float
        see :ref:`disc_fac`
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    log_like: float
        Contains the negative partial log likelihood for the given parameters.
    """

    costs = calc_obs_costs(num_states, maint_func, mpec_params[num_states:], scale)
    p_choice = choice_prob_gumbel(mpec_params[0:num_states], costs, disc_fac)
    log_like = like_hood_data(np.log(p_choice), decision_mat, state_mat)
    return float(log_like)


def mpec_loglike_cost_params_derivative(
    mpec_params,
    maint_func,
    maint_func_dev,
    num_states,
    disc_fac,
    scale,
    decision_mat,
    state_mat,
):
    """
    Computing the analytical gradient of the objective function for MPEC.

    Parameters
    ----------
    mpec_params : numpy.ndarray
        see :ref:`mpec_params`
    maint_func: func
        see :ref:`maint_func`
    maint_func_dev: func
        see :ref:`maint_func`
    num_states : int
        The size of the state space.
    disc_fac : numpy.float
        see :ref:`disc_fac`
    scale : numpy.float
        see :ref:`scale`
    decision_mat : numpy.ndarray
        see :ref:`decision_mat`
    state_mat : numpy.ndarray
        see :ref:`state_mat`

    Returns
    -------
    gradient : numpy.ndarray
        Vector that holds the derivative of the negative log likelihood function
        to the parameters.

    """
    num_params = mpec_params[num_states:].shape[0]
    # Calculate choice probabilities
    costs = calc_obs_costs(num_states, maint_func, mpec_params[num_states:], scale)
    p_choice = choice_prob_gumbel(mpec_params[0:num_states], costs, disc_fac)

    # calculate the derivative based on the model
    derivative_both = mpec_loglike_cost_params_derivative_model(
        num_states, num_params, disc_fac, scale, maint_func_dev, p_choice
    )

    # Calculate actual gradient depending on the given data
    # get decision matrix into the needed shape
    decision_mat_temp = np.vstack(
        (
            np.tile(decision_mat[0], (num_states + num_params, 1)),
            np.tile(decision_mat[1], (num_states + num_params, 1)),
        )
    )

    # calculate the gradient
    gradient_temp = -np.sum(
        decision_mat_temp * np.dot(derivative_both, state_mat), axis=1
    )
    # bring the calculated gradient into the correct shape
    gradient = np.reshape(gradient_temp, (num_states + num_params, 2), order="F").sum(
        axis=1
    )

    return gradient


def mpec_constraint(
    mpec_params,
    maint_func,
    maint_func_dev,
    num_states,
    disc_fac,
    scale,
    trans_mat,
):
    """
    Calculate the constraint of MPEC.

    Parameters
    ----------
    mpec_params : numpy.ndarray
        see :ref:`mpec_params`
    maint_func: func
        see :ref:`maint_func`
    maint_func_dev: func
        see :ref:`maint_func`
    num_states : int
        The size of the state space.
    disc_fac : numpy.float
        see :ref:`disc_fac`
    scale : numpy.float
        see :ref:`scale`
    trans_mat : numpy.ndarray
        see :ref:`trans_mat`

    Returns
    -------
    None.

    """
    ev = mpec_params[0:num_states]
    obs_costs = calc_obs_costs(num_states, maint_func, mpec_params[num_states:], scale)

    maint_value = disc_fac * ev - obs_costs[:, 0]
    repl_value = disc_fac * ev[0] - obs_costs[0, 1] - obs_costs[0, 0]

    # Select the minimal absolute value to rescale the value vector for the
    # exponential function.
    ev_max = np.max(np.array(maint_value, repl_value))

    log_sum = ev_max + np.log(
        np.exp(maint_value - ev_max) + np.exp(repl_value - ev_max)
    )

    ev_new = np.dot(trans_mat, log_sum)
    return ev_new - ev


def mpec_constraint_derivative(
    mpec_params,
    maint_func,
    maint_func_dev,
    num_states,
    disc_fac,
    scale,
    trans_mat,
):
    """
    Calculating the analytical Jacobian of the MPEC constraint.

    Parameters
    ----------
    mpec_params : numpy.ndarray
        see :ref:`mpec_params`
    maint_func: func
        see :ref:`maint_func`
    maint_func_dev: func
        see :ref:`maint_func`
    num_states : int
        The size of the state space.
    disc_fac : numpy.float
        see :ref:`disc_fac`
    scale : numpy.float
        see :ref:`scale`
    trans_mat : numpy.ndarray
        see :ref:`trans_mat`

    Returns
    -------
    jacobian : numpy.ndarray
        Jacobian of the MPEC constraint.

    """
    # Calculate a vector representing 1 divided by the right hand side of the MPEC
    # constraint
    num_params = mpec_params[num_states:].shape[0]
    ev = mpec_params[0:num_states]
    obs_costs = calc_obs_costs(num_states, maint_func, mpec_params[num_states:], scale)

    maint_value = disc_fac * ev - obs_costs[:, 0]
    repl_value = disc_fac * ev[0] - obs_costs[0, 1] - obs_costs[0, 0]

    ev_max = np.max(np.array(maint_value, repl_value))

    exp_centered_maint_value = np.exp(maint_value - ev_max)
    exp_centered_repl_value = np.exp(repl_value - ev_max)
    log_sum_denom = 1 / (exp_centered_maint_value + exp_centered_repl_value)

    jacobian = np.zeros((num_states, num_states + num_params))

    # Calculate derivative to EV(0)
    jacobian[:, 0] = np.dot(
        disc_fac * exp_centered_repl_value * trans_mat, log_sum_denom
    )
    jacobian[0, 0] = (
        jacobian[0, 0]
        + (1 - log_sum_denom[0] * exp_centered_repl_value) * disc_fac * trans_mat[0, 0]
    )
    # Calculate derivative to EV(1) until EV(num_states)
    jacobian[:, 1:num_states] = (
        trans_mat[:, 1:] * log_sum_denom[1:] * disc_fac * exp_centered_maint_value[1:]
    )
    # Calculate derivative to RC
    jacobian[:, num_states] = np.dot(
        trans_mat, -exp_centered_repl_value * log_sum_denom
    )
    # Calculate derivative to maintenance cost parameters
    log_sum_denom_temp = np.reshape(log_sum_denom, (num_states, 1))
    maint_cost_difference_dev = np.reshape(
        (-exp_centered_maint_value * maint_func_dev(num_states, scale).T).T
        - exp_centered_repl_value * maint_func_dev(num_states, scale)[0],
        (num_states, num_params - 1),
    )

    jacobian[:, num_states + 1 :] = np.reshape(
        np.dot(trans_mat, log_sum_denom_temp * maint_cost_difference_dev),
        (num_states, num_params - 1),
    )
    # Calculate the Jacobian of EV
    ev_jacobian = np.hstack((np.eye(num_states), np.zeros((num_states, num_params))))

    jacobian = jacobian - ev_jacobian

    return jacobian


def mpec_loglike_cost_params_derivative_model(
    num_states, num_params, disc_fac, scale, maint_func_dev, p_choice
):
    """
    generates the derivative of the log likelihood function of mpec depending
    on the model characteristics.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    num_params : int
        Length of cost parameter vector.
    disc_fac : numpy.float
        see :ref:`disc_fac`
    scale : numpy.float
        see :ref:`scale`
    maint_func_dev : func
        see :ref:`maint_func`
    p_choice numpy.ndarray
        num_states x 2 matrix that contains the calculated conditional choice
        probabilities.

    Returns
    -------
    derivative_both numpy.ndarray
        gives out the derivative of the log likelihood function depending
        on the model characteristics.

    """
    # Create matrix that represents d[V(0)-V(x)]/ d[theta] (depending on x)
    payoff_difference_derivative = np.zeros((num_states + num_params, num_states))
    payoff_difference_derivative[0, 1:] = disc_fac
    payoff_difference_derivative[1:num_states, 1:] = -disc_fac * np.eye(num_states - 1)
    payoff_difference_derivative[num_states, :] = -1
    payoff_difference_derivative[num_states + 1 :, :] = (
        -maint_func_dev(num_states, scale)[0] + maint_func_dev(num_states, scale)
    ).T

    # Calculate derivative depending on whether d is 0 or 1
    derivative_d0 = -payoff_difference_derivative * p_choice[:, 1]
    derivative_d1 = payoff_difference_derivative * p_choice[:, 0]
    derivative_both = np.vstack((derivative_d0, derivative_d1))

    return derivative_both
