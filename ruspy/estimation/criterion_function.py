"""
This module specifies the criterion function, its derivative and arguments needed for the estimation process.
"""
import time
from functools import partial

import nlopt
import numpy as np
from estimagic.optimization.optimize import minimize

# we shouldn't need this
try:
    import ipopt  # noqa:F401

    optional_package_is_available = True
except ImportError:
    optional_package_is_available = False

from scipy.optimize._numdiff import approx_derivative
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.estimation import config
from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import (
    loglike_cost_params,
    derivative_loglike_cost_params,
)
from ruspy.estimation.estimation_interface import select_model_parameters
from ruspy.estimation.estimation_interface import select_optimizer_options
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.estimation.mpec import mpec_constraint
from ruspy.estimation.mpec import mpec_constraint_derivative
from ruspy.estimation.mpec import mpec_loglike_cost_params
from ruspy.estimation.mpec import mpec_loglike_cost_params_derivative
from ruspy.estimation.mpec import wrap_ipopt_constraint
from ruspy.estimation.mpec import wrap_ipopt_likelihood
from ruspy.estimation.mpec import wrap_mpec_loglike


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
    args = (
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

    # this is done in estimation_interface_new.py:
    # optimizer_options = select_optimizer_options(init_dict, num_params, num_states)

    if "method" in init_dict:
        method = init_dict["method"]
    else:
        raise ValueError("The key 'method' must be in init_dict")

    if method == "NFXP":

        alg_details = {} if "alg_details" not in init_dict else init_dict["alg_details"]

        # optimizer_options["criterion"] = loglike_cost_params
        # optimizer_options["criterion_dev"] = derivative_loglike_cost_params
        #
        # criterion_temp = optimizer_options.pop("criterion")  # removes and returns an element from a dictionary having the given key.
        # n_evaluations, criterion = wrap_nfxp_criterion(criterion_temp)
        #
        # criterion_dev_temp = optimizer_options.pop("criterion_dev")
        # n_evaluations, criterion_dev = wrap_nfxp_criterion(criterion_dev_temp)

        # kann man auch einfacher schreiben:
        # criterion_temp = loglike_cost_params  # removes and returns an element from a dictionary having the given key.
        # n_evaluations, criterion_func = wrap_nfxp_criterion(criterion_temp)
        # criterion_dev_temp = derivative_loglike_cost_params
        # n_evaluations, criterion_dev = wrap_nfxp_criterion(criterion_dev_temp)

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

        # gradient = optimizer_options.pop("derivative")
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
            "gradient": gradient
            # "grad": grad # optional
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


# wofür brauchen wir diese Funktion?
def wrap_nfxp_criterion(function):
    ncalls = [0]

    def function_wrapper(*wrapper_args, **wrapper_kwargs):
        ncalls[0] += 1
        return function(*wrapper_args, **wrapper_kwargs)

    return ncalls, function_wrapper
