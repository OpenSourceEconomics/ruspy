"""
This module contains the main function for the estimation process.
"""
import numpy as np
import scipy.optimize as opt

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import loglike_cost_params
from ruspy.estimation.estimation_interface import select_model_parameters
from ruspy.estimation.estimation_interface import select_optimizer_options
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.estimation.standard_errors import calc_asymp_stds


def estimate(init_dict, df):
    """
    This function coordinates the estimation process of the ruspy package.

    Parameters
    ----------
    init_dict : dictionary
        see ref:`init_dict`

    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    transition_results : dictionary
        see :ref:`result_trans`
    result_cost_params : dictionary
        see :ref:`result_costs



    """

    transition_results = estimate_transitions(df)

    endog = df.loc[(slice(None), slice(1, None)), "decision"].to_numpy()
    states = df.loc[(slice(None), slice(1, None)), "state"].to_numpy()

    (
        disc_fac,
        num_states,
        maint_func,
        maint_func_dev,
        num_params,
        scale,
    ) = select_model_parameters(init_dict)

    decision_mat = np.vstack(((1 - endog), endog))
    trans_mat = create_transition_matrix(num_states, np.array(transition_results["x"]))
    state_mat = create_state_matrix(states, num_states)

    optimizer_options = select_optimizer_options(init_dict, num_params)

    result_cost_params = opt.minimize(
        loglike_cost_params,
        args=(
            maint_func,
            maint_func_dev,
            num_states,
            trans_mat,
            state_mat,
            decision_mat,
            disc_fac,
            scale,
        ),
        **optimizer_options
    )
    if "hess_inv" in result_cost_params:
        result_cost_params["std"] = calc_asymp_stds(
            result_cost_params["x"], result_cost_params["hess_inv"]
        )
    return transition_results, result_cost_params
