"""
This module contains the main function for the estimation process of the
Mathematical Programming with Equilibrium Constraints (MPEC).
"""
from functools import partial

import nlopt
import numpy as np

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.estimation_interface import select_model_parameters
from ruspy.estimation.estimation_interface import select_optimizer_options
from ruspy.estimation.estimation_mpec_functions import mpec_constraint
from ruspy.estimation.estimation_mpec_functions import mpec_loglike_cost_params
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions


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
    trans_mat = create_transition_matrix(
        num_states, np.array(mpec_transition_results["x"])
    )
    state_mat = create_state_matrix(states, num_states)

    optimizer_options = select_optimizer_options(init_dict, num_params, num_states)
    gradient = optimizer_options.pop("gradient")

    # Calculate partial functions needed for nlopt
    partial_loglike_mpec = partial(
        mpec_loglike_cost_params,
        maint_func,
        maint_func_dev,
        num_states,
        num_params,
        state_mat,
        decision_mat,
        disc_fac,
        scale,
        gradient,
    )

    partial_constr_mpec = partial(
        mpec_constraint,
        maint_func,
        maint_func_dev,
        num_states,
        num_params,
        trans_mat,
        disc_fac,
        scale,
        gradient,
    )

    # set up nlopt
    opt = nlopt.opt(
        eval("nlopt." + optimizer_options.pop("algorithm")), num_states + num_params
    )
    opt.set_min_objective(partial_loglike_mpec)
    opt.add_equality_mconstraint(
        partial_constr_mpec, np.full(num_states, 1e-6),
    )

    # supply user choices
    params = optimizer_options.pop("params")

    if "set_local_optimizer" in optimizer_options:
        exec(
            "opt.set_local_optimizer(nlopt.opt(nlopt."
            + optimizer_options.pop("set_local_optimizer")
            + ", num_states+num_params))"
        )

    for key, _value in optimizer_options.items():
        exec("opt." + key + "(_value)")

    # Solving nlopt
    mpec_cost_parameters = {}
    mpec_cost_parameters["x"] = opt.optimize(params)
    mpec_cost_parameters["fun"] = opt.last_optimum_value()
    if opt.last_optimize_result() > 0:
        mpec_cost_parameters["status"] = True
    else:
        mpec_cost_parameters["status"] = False

    return mpec_transition_results, mpec_cost_parameters
