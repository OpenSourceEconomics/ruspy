"""
This module specifies the criterion function and its derivative.
"""
from functools import partial

import numpy as np

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import derivative_loglike_cost_params
from ruspy.estimation.est_cost_params import loglike_cost_params
from ruspy.estimation.estimation_interface_new import select_model_parameters
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.estimation.mpec import mpec_loglike_cost_params
from ruspy.estimation.mpec import mpec_loglike_cost_params_derivative


def get_criterion_function(
    init_dict,
    df,
):
    """
    This function specifies the criterion function with its derivative
    and arguments for NFXP and MPEC.

    Parameters
    ----------
    init_dict : dictionary
        see :ref:`init_dict`
    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    criterion_func :
        see :ref:`criterion_func`
    criterion_dev :
        see :ref:`criterion_dev`
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

    if "method" in init_dict:
        method = init_dict["method"]
    else:
        raise ValueError("The key 'method' must be in init_dict")

    if method == "NFXP":

        (criterion_function, criterion_derivative) = get_criterion_function_nfxp(
            disc_fac,
            num_states,
            maint_func,
            maint_func_dev,
            scale,
            decision_mat,
            trans_mat,
            state_mat,
            alg_details,
        )

    elif method == "MPEC":

        (criterion_function, criterion_derivative) = get_criterion_function_mpec(
            disc_fac,
            num_states,
            maint_func,
            maint_func_dev,
            num_params,
            scale,
            decision_mat,
            state_mat,
        )

    else:
        raise ValueError(
            f"{method} is not implemented. Only MPEC or NFXP are valid choices"
        )

    return criterion_function, criterion_derivative, transition_results


def get_criterion_function_nfxp(
    disc_fac,
    num_states,
    maint_func,
    maint_func_dev,
    scale,
    decision_mat,
    trans_mat,
    state_mat,
    alg_details,
):
    nfxp_kwargs = {
        "maint_func": maint_func,
        "maint_func_dev": maint_func_dev,
        "num_states": num_states,
        "trans_mat": trans_mat,
        "state_mat": state_mat,
        "decision_mat": decision_mat,
        "disc_fac": disc_fac,
        "scale": scale,
        "alg_details": alg_details,
    }

    criterion_function = partial(loglike_cost_params, **nfxp_kwargs)
    criterion_derivative = partial(derivative_loglike_cost_params, **nfxp_kwargs)
    return criterion_function, criterion_derivative


def get_criterion_function_mpec(
    disc_fac,
    num_states,
    maint_func,
    maint_func_dev,
    num_params,
    scale,
    decision_mat,
    state_mat,
):
    mpec_criterion_kwargs = {
        "maint_func": maint_func,
        "num_states": num_states,
        "state_mat": state_mat,
        "decision_mat": decision_mat,
        "disc_fac": disc_fac,
        "scale": scale,
    }

    mpec_criterion_dev_kwargs = {
        "maint_func": maint_func,
        "maint_func_dev": maint_func_dev,
        "num_states": num_states,
        "num_params": num_params,
        "state_mat": state_mat,
        "decision_mat": decision_mat,
        "disc_fac": disc_fac,
        "scale": scale,
    }

    criterion_function = partial(mpec_loglike_cost_params, **mpec_criterion_kwargs)
    criterion_derivative = partial(
        mpec_loglike_cost_params_derivative, **mpec_criterion_dev_kwargs
    )
    return criterion_function, criterion_derivative
