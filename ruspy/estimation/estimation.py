"""
This module contains the main function for the estimation process.
"""
import numpy as np
from estimagic.optimization.optimize import minimize

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import loglike_cost_params
from ruspy.estimation.estimation_interface import select_model_parameters
from ruspy.estimation.estimation_interface import select_optimizer_options
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions


def estimate(init_dict, df):
    """
    Estimation function of ruspy.

    This function coordinates the estimation process of the ruspy package.

    Parameters
    ----------
    init_dict : dictionary
        see ref:`_est_init_dict`

    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    transition_results : dictionary
        see :ref:`result_trans`
    result_cost_params : dictionary
        see :ref:`result_costs`



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
    trans_mat = create_transition_matrix(num_states, np.array(transition_results["x"]))
    state_mat = create_state_matrix(states, num_states)

    optimizer_options = select_optimizer_options(init_dict, num_params)

    alg_details = {} if "alg_details" not in init_dict else init_dict["alg_details"]

    kwargs = {
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

    result_cost_params = {}

    min_result = minimize(
        loglike_cost_params,
        criterion_kwargs=kwargs,
        gradient_kwargs=kwargs,
        **optimizer_options,
    )
    result_cost_params["x"] = min_result[1]["value"].to_numpy()
    result_cost_params["fun"] = min_result[0]["fitness"]
    result_cost_params["status"] = min_result[0]["status"]
    result_cost_params["message"] = min_result[0]["message"]
    result_cost_params["jac"] = min_result[0]["jacobian"]

    return transition_results, result_cost_params
