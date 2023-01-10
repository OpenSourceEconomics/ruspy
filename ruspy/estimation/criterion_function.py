"""
This module specifies the criterion function and its derivative.
"""
from functools import partial

import numpy as np

from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.estimation.mpec import mpec_constraint
from ruspy.estimation.mpec import mpec_constraint_derivative
from ruspy.estimation.mpec import mpec_loglike_cost_params
from ruspy.estimation.mpec import mpec_loglike_cost_params_derivative
from ruspy.estimation.nfxp import create_state_matrix
from ruspy.estimation.nfxp import derivative_loglike_cost_params
from ruspy.estimation.nfxp import derivative_loglike_cost_params_individual
from ruspy.estimation.nfxp import loglike_cost_params
from ruspy.estimation.nfxp import loglike_cost_params_individual
from ruspy.estimation.pre_processing import select_model_parameters


def get_criterion_function(
    init_dict,
    df,
):
    """
    This function specifies the criterion function with its derivative,
    transition probabilites (for NXFP and MPEC) as well as the contraint
    function with its derivative (for MPEC).

    Parameters
    ----------
    init_dict : dictionary
        see :ref:`init_dict`
    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    func_dict :
        see :ref:`func_dict`
    transition_results : dictionary
        see :ref:`result_trans`
    """

    transition_results = estimate_transitions(df)
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

    trans_mat = create_transition_matrix(num_states, transition_results["x"])
    state_mat = create_state_matrix(states, num_states)

    alg_details = {} if "alg_details" not in init_dict else init_dict["alg_details"]

    basic_kwargs = {
        "maint_func": maint_func,
        "maint_func_dev": maint_func_dev,
        "num_states": num_states,
        "disc_fac": disc_fac,
        "scale": scale,
    }

    if "method" in init_dict:
        method = init_dict["method"]
    else:
        raise ValueError("The key 'method' must be in init_dict")

    func_dict = {}
    nfxp_kwargs = {
        **basic_kwargs,
        "trans_mat": trans_mat,
        "state_mat": state_mat,
        "decision_mat": decision_mat,
        "alg_details": alg_details,
    }

    if method == "NFXP":
        func_dict["criterion_function"] = partial(loglike_cost_params, **nfxp_kwargs)
        func_dict["criterion_derivative"] = partial(
            derivative_loglike_cost_params, **nfxp_kwargs
        )

    elif method == "NFXP_BHHH":
        func_dict["criterion_function"] = partial(
            loglike_cost_params_individual, **nfxp_kwargs
        )
        func_dict["criterion_derivative"] = partial(
            derivative_loglike_cost_params_individual, **nfxp_kwargs
        )

    elif method == "MPEC":
        mpec_crit_kwargs = {
            **basic_kwargs,
            "state_mat": state_mat,
            "decision_mat": decision_mat,
        }

        func_dict["criterion_function"] = partial(
            mpec_loglike_cost_params, **mpec_crit_kwargs
        )
        func_dict["criterion_derivative"] = partial(
            mpec_loglike_cost_params_derivative, **mpec_crit_kwargs
        )
        mpec_constr_kwargs = {
            **basic_kwargs,
            "trans_mat": trans_mat,
        }
        func_dict["constraint"] = partial(mpec_constraint, **mpec_constr_kwargs)
        func_dict["constraint_dev"] = partial(
            mpec_constraint_derivative, **mpec_constr_kwargs
        )
    else:
        raise ValueError(
            f"{method} is not implemented. Only MPEC or NFXP are valid choices"
        )

    return func_dict, transition_results
