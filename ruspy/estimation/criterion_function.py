"""
This module specifies the criterion function, its derivative and
arguments needed for the estimation process.
"""
# from functools import partial
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
    criterion_kwargs :
        see :ref:`criterion_kwargs`
    criterion_dev :
        see :ref:`criterion_dev`
    criterion_dev_kwargs :
        see :ref:`criterion_dev_kwargs`
    """
    # not sure if we need that
    transition_results = estimate_transitions(df)
    endog = df.loc[(slice(None), slice(1, None)), "decision"].to_numpy(int)
    states = df.loc[(slice(None), slice(1, None)), "state"].to_numpy(int)

    # this is done in estimation_interface_new.py:
    (
        disc_fac,
        num_states,
        maint_func,
        maint_func_dev,
        num_params,
        scale,
    ) = select_model_parameters(init_dict)

    # not sure if we need that
    decision_mat = np.vstack(((1 - endog), endog))
    trans_mat = create_transition_matrix(num_states, np.array(transition_results["x"]))
    state_mat = create_state_matrix(states, num_states)

    if "method" in init_dict:
        method = init_dict["method"]
    else:
        raise ValueError("The key 'method' must be in init_dict")

    if method == "NFXP":

        alg_details = {} if "alg_details" not in init_dict else init_dict["alg_details"]

        criterion_func = loglike_cost_params
        criterion_dev = derivative_loglike_cost_params

        # inputs loglike_cost_params beside params
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

        criterion_kwargs = nfxp_kwargs
        criterion_dev_kwargs = nfxp_kwargs

    elif method == "MPEC":

        criterion_func = mpec_loglike_cost_params
        criterion_dev = mpec_loglike_cost_params_derivative
        gradient = "Yes"

        mpec_criterion_kwargs = {
            "maint_func": maint_func,
            "maint_func_dev": maint_func_dev,
            "num_states": num_states,
            "num_params": num_params,
            "state_mat": state_mat,
            "decision_mat": decision_mat,
            "disc_fac": disc_fac,
            "scale": scale,
            "gradient": gradient,
        }
        criterion_kwargs = mpec_criterion_kwargs

        # arguments criterion_dev
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

        criterion_dev_kwargs = mpec_criterion_dev_kwargs

    else:
        raise ValueError(
            f"{method} is not implemented. Only MPEC or NFXP are valid choices"
        )

    return criterion_func, criterion_kwargs, criterion_dev, criterion_dev_kwargs
