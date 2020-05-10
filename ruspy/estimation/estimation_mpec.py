"""
This module contains the main function for the estimation process of the 
Mathematical Programming with Equilibrium Constraints (MPEC).
"""
import nlopt
import numpy as np

from functools import partial
from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.estimation_interface import select_model_parameters
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.estimation.estimation_mpec_functions import (mpec_loglike_cost_params, 
                                                        mpec_constraint)
                                                        
                                                        

def estimate_mpec(init_dict, df):
    """
    Estimation function of Mathematical Programming with Equilibrium Constraints
    (MPEC) in ruspy.


    Parameters
    ----------
    init_dict : dictionary
        see ref:`_est_init_dict`

    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    mpec_transition_results : dictionary
        see :ref:`mpec_transition_results`
    mpec_cost_parameters : dictionary
        see :ref:`mpec_cost_parameters`



    """

    mpec_transition_results = estimate_transitions(df)

    endog = df.loc[(slice(None), slice(1, None)), "decision"].to_numpy(int)
    states = df.loc[(slice(None), slice(1, None)), "state"].to_numpy(int)

    (
        disc_fac,
        num_states,
        maint_func,
        maint_func_dev,
        num_params,
        scale,
    ) = select_model_parameters(init_dict)

    decision_mat = np.vstack(((1 - endog), endog))
    trans_mat = create_transition_matrix(num_states, np.array(mpec_transition_results["x"]))
    state_mat = create_state_matrix(states, num_states)

    
    # Calculate partial functions needed for nlopt
    partial_loglike_mpec = partial(mpec_loglike_cost_params, maint_func,
                                   maint_func_dev, num_states, num_params,
                                   state_mat, decision_mat, 
                                   disc_fac, scale)
    
    partial_constr_mpec = partial(mpec_constraint, maint_func, maint_func_dev,
                                  num_states, num_params,
                                  trans_mat, disc_fac, scale)
    
    
    # set up nlopt
    opt=nlopt.opt(nlopt.LD_SLSQP, num_states + num_params)
    # opt = nlopt.opt(nlopt.AUGLAG_EQ, num_states + num_params)
    # opt.set_local_optimizer(nlopt.opt(nlopt.LD_SLSQP, num_states + num_params))
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
    # opt.set_maxeval(1000)
    start = np.concatenate((np.full(num_states, 0.0), np.array([4.0,1.0])))
    # Solving nlopt
    mpec_cost_parameters = opt.optimize(start)

    return mpec_transition_results, mpec_cost_parameters

