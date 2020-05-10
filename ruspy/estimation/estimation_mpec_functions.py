"""
This module contains all the key functions used to estimate MPEC.
"""
import numpy as np

from functools import partial
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from scipy.optimize import approx_fprime
from scipy.optimize._numdiff import approx_derivative


def function_wrapper_like(function, args):
    def call(mpec_params, grad):
        result = function(mpec_params, *args, grad)
        return result
    return call

def function_wrapper_constr(function, args):
    def call(mpec_params):
        result = function(result=np.array([]), mpec_params=mpec_params, *args, grad=np.array([]))
        return result
    return call

def mpec_loglike_cost_params(
    maint_func,
    maint_func_dev,
    num_states,
    num_params,
    state_mat,
    decision_mat,
    disc_fac,
    scale,
    mpec_params,
    grad=np.array([]),
):
    """
    Calculate the negative partial log likelihood depending on cost parameters 
    as well as the discretized expected values.

    Parameters
    ----------
    maint_func : func
        see :ref:`maint_func`
    num_states : int
        The size of the state space.
    state_mat : numpy.array
        see :ref:`state_mat`
    decision_mat : numpy.array
        see :ref:`decision_mat` 
    disc_fac : numpy.float
        see :ref:`disc_fac`
    scale : numpy.float
        see :ref:`scale`
    mpec_params : numpy.array
        Contains the expected values as well as the cost parameters.
    grad : numpy.array, optional
        The gradient of the function. The default is np.array([]).

    Returns
    -------
    log_like: float
        Contains the negative partial log likelihood for the given parameters.

    """
    if grad.size>0:
        # numerical gradient (comment out for analytical)
        # partial_loglike_mpec = partial(mpec_loglike_cost_params, maint_func,
        #                        maint_func_dev, num_states, num_params,
        #                        state_mat, decision_mat, 
        #                        disc_fac, scale)
        # grad[:] = approx_fprime(mpec_params, partial_loglike_mpec, 10e-6)
       
        # analytical gradient (comment out for numerical)
        grad[:] = mpec_loglike_cost_params_dev(mpec_params, maint_func, maint_func_dev, 
                                                       num_states, num_params, 
                                                       disc_fac, scale, 
                                                       decision_mat, state_mat)
        
    costs = calc_obs_costs(num_states, maint_func, mpec_params[num_states:], scale)   
    p_choice = choice_prob_gumbel(mpec_params[0:num_states], costs, disc_fac)
    log_like = like_hood_data(np.log(p_choice), decision_mat, state_mat)
    return float(log_like)


def like_hood_data(l_values, decision_mat, state_mat):
    return -np.sum(decision_mat * np.dot(l_values.T, state_mat))


def mpec_constraint(maint_func,
                    maint_func_dev,
                    num_states,
                    num_params,
                    trans_mat, 
                    disc_fac,
                    scale,
                    result, 
                    mpec_params,
                    grad=np.array([])):
    """
    Calulate the constraint of MPEC.
    
    Parameters
    ----------
    maint_func : func
        see :ref:`maint_func`
    num_states : int
        The size of the state space.
    decision_mat : numpy.array
        see :ref:`decision_mat` 
    disc_fac : numpy.float
        see :ref:`disc_fac`
    scale : numpy.float
        see :ref:`scale`
    result : numpy.array
        Contains the left hand side of the constraint minus the right hand side
        for the nlopt solver. This should be zero for the constraint to hold.
    mpec_params : numpy.array
        Contains the expected values as well as the cost parameters.
    grad : numpy.array, optional
        The gradient of the function. The default is np.array([]).

    Returns
    -------
    None.

    """
    if grad.size > 0:
        # numerical jacobian (comment out for analytical)
        # partial_constr_mpec_deriv = function_wrapper_constr(mpec_constraint, 
        #                                             args=(maint_func,
        #                                                   maint_func_dev,
        #                                                   num_states,
        #                                                   num_params,
        #                                                   trans_mat, 
        #                                                   disc_fac,
        #                                                   scale))
        # grad[:, :] = approx_derivative(partial_constr_mpec_deriv, mpec_params)
        
        # analytical jacobian (comment out for numerical)
        grad[:, :] = mpec_constraint_dev(mpec_params,
                                        maint_func,
                                        maint_func_dev,
                                        num_states,
                                        num_params,
                                        disc_fac,
                                        scale,
                                        trans_mat)      
        
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
    if result.size > 0:
        result[:] = ev_new - ev
    return ev_new - ev


def mpec_loglike_cost_params_dev(mpec_params,
                                 maint_func,
                                 maint_func_dev,
                                 num_states,
                                 num_params,
                                 disc_fac,
                                 scale,
                                 decision_mat,
                                 state_mat,
                                 ):
    """
    Computing the gradient of the objective function for MPEC.

    Parameters
    ----------
    mpec_params : numpy.array
        Contains the expected values as well as the cost parameters.
    maint_func: func
        see :ref: `maint_func`
    maint_func_dev : func
        see :ref: `maint_func_dev`
    num_states : int
        The size of the state space.
    num_params : int
        Length of cost parameter vector.
    disc_fac : numpy.float
        see :ref:`disc_fac`
    scale : numpy.float
        see :ref:`scale`
    decision_mat : numpy.array
        see :ref:`decision_mat` 
    state_mat : numpy.array
        see :ref:`state_mat`

    Returns
    -------
    gradient : numpy.array
        Vector that holds the derivative of the negative log likelihood function
        to the parameters.

    """
    # Calculate choice probabilities
    costs = calc_obs_costs(num_states, maint_func, mpec_params[num_states:], scale)   
    p_choice = choice_prob_gumbel(mpec_params[0:num_states], costs, disc_fac)

    # Create matrix that represents d[V(0)-V(x)]/ d[theta] (depending on x)
    payoff_difference_derivative = np.zeros((num_states + num_params, num_states))
    payoff_difference_derivative[0, 1:] = disc_fac 
    payoff_difference_derivative[1:num_states, 1:] = -disc_fac*np.eye(num_states-1) 
    payoff_difference_derivative[num_states, :] = -1
    payoff_difference_derivative[num_states+1:, :] = (
        -maint_func_dev(num_states, scale)[0] + maint_func_dev(num_states, scale)).T
    
    # Calculate derivative depending on whether d is 0 or 1
    derivative_d0 = -payoff_difference_derivative * p_choice[:,1]
    derivative_d1 = payoff_difference_derivative * p_choice[:,0]
    derivative_both = np.vstack((derivative_d0, derivative_d1))
    
    # Calculate actual gradient depending on the given data
    decision_mat_temp = np.vstack((np.tile(decision_mat[0], (num_states+num_params, 1)),
                                   np.tile(decision_mat[1], (num_states+num_params, 1))))
    
    gradient_temp = -np.sum(decision_mat_temp * np.dot(derivative_both, state_mat), axis = 1)
    gradient = np.reshape(gradient_temp, (num_states+num_params, 2), order = "F").sum(axis = 1)
    
    return gradient

def mpec_constraint_dev(mpec_params,
                        maint_func,
                        maint_func_dev,
                        num_states,
                        num_params,
                        disc_fac,
                        scale,
                        trans_mat):
    """
    Calculating the Jacobian of the MPEC constraint.

    Parameters
    ----------
    mpec_params : numpy.array
        Contains the expected values as well as the cost parameters.
    maint_func: func
        see :ref: `maint_func`
    maint_func_dev : func
        see :ref: `maint_func_dev`
    num_states : int
        The size of the state space.
    num_params : int
        Length of cost parameter vector.
    disc_fac : numpy.float
        see :ref:`disc_fac`
    scale : numpy.float
        see :ref:`scale`
    trans_mat : numpy.array
        see :ref:`trans_mat`

    Returns
    -------
    jacobian : numpy.array
        Jacobian of the MPEC constraint.

    """
    # Calculate a vector representing 1 divided by the right hand side of the MPEC constraint
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
    jacobian[:, 0] = np.dot(disc_fac * exp_centered_repl_value * trans_mat, log_sum_denom)
    jacobian[0, 0] = jacobian[0, 0] + (1 - log_sum_denom[0] * exp_centered_repl_value) * disc_fac * trans_mat[0, 0]
    # Calculate derivative to EV(1) until EV(num_states)
    jacobian[:, 1:num_states] = trans_mat[:,1:] * log_sum_denom[1:] * disc_fac * exp_centered_maint_value[1:]
    # Calculate derivative to RC
    jacobian[:, num_states] = np.dot(trans_mat, -exp_centered_repl_value * log_sum_denom)     
    # Calculate derivative to maintenance cost parameters
    jacobian[:, num_states+1:] = np.reshape(
        np.dot(trans_mat, log_sum_denom * (
        (-exp_centered_maint_value * maint_func_dev(num_states, scale).T).T - 
        exp_centered_repl_value * maint_func_dev(num_states, scale)[0])
            ), (num_states, num_params-1)
        )
    # Calculate the Jacobian of EV 
    ev_jacobian = np.hstack((np.eye(num_states), np.zeros((num_states, num_params))))
    
    jacobian = jacobian - ev_jacobian
    
    return jacobian
