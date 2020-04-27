import nlopt
import numpy as np

from functools import partial
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from scipy.optimize import approx_fprime
from scipy.optimize._numdiff import approx_derivative


def function_wrapper_like(function, args):
    def call(params, grad):
        result = function(params, *args, grad)
        return result
    return call

def function_wrapper_constr(function, args):
    def call(params):
        result = function(result=np.array([]), params=params, *args, grad=np.array([]))
        return result
    return call

def loglike_cost_params_mpec(
    maint_func,
    num_states,
    state_mat,
    decision_mat,
    disc_fac,
    scale,
    params,
    grad=np.array([]),
):
    if grad.size>0:
        grad[:] = approx_fprime(params, partial_loglike_mpec, 10e-6)
    
    costs = calc_obs_costs(num_states, maint_func, params[num_states:], scale)   
    p_choice = choice_prob_gumbel(params[0:num_states], costs, disc_fac)
    log_like = like_hood_data(np.log(p_choice), decision_mat, state_mat)
    return float(log_like)

def constraint_mpec(maint_func,
                    num_states,
                    trans_mat, 
                    disc_fac,
                    scale,
                    result, 
                    params,
                    grad=np.array([])):

    if grad.size > 0:
        grad[:, :] = approx_derivative(partial_constr_mpec_deriv, params)
    
    ev = params[0:num_states]
    obs_costs = calc_obs_costs(num_states, maint_func, params[num_states:], scale)
    
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
        result[:] = ev - ev_new
    return ev - ev_new

def like_hood_data(l_values, decision_mat, state_mat):
    return -np.sum(decision_mat * np.dot(l_values.T, state_mat))

partial_loglike_mpec = partial(loglike_cost_params_mpec, maint_func, 
                               num_states, state_mat, decision_mat, 
                               disc_fac, scale)

partial_constr_mpec = partial(constraint_mpec, maint_func, num_states, 
                              trans_mat, disc_fac, scale)

partial_constr_mpec_deriv = function_wrapper_constr(constraint_mpec, 
                                                    args=(maint_func, num_states,trans_mat, disc_fac, scale))

# set up nlopt
opt=nlopt.opt(nlopt.LD_SLSQP, num_states + num_params)
opt = nlopt.opt(nlopt.AUGLAG_EQ, num_states + num_params)
opt.set_local_optimizer(nlopt.opt(nlopt.LD_SLSQP, num_states + num_params))
opt.set_min_objective(partial_loglike_mpec)
lb = np.concatenate((np.full(num_states, -np.inf), np.full(num_params, 0.0)))
ub = np.concatenate((np.full(num_states, 50.0), np.full(num_params, np.inf)))
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)
opt.add_equality_mconstraint(partial_constr_mpec, 
                             np.full(num_states, 1e-6),
                             )
opt.set_ftol_abs(1e-15)
# opt.set_ftol_rel(1e-6)
opt.set_xtol_rel(1e-15)
opt.set_xtol_abs(1e-3)
opt.set_maxeval(500)
start = np.concatenate((np.full(num_states, 0.0), np.array([10.0,2.0])))
x_LD_SLSQP_500 = opt.optimize(start)




