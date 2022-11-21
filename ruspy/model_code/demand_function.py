"""
Calculates the implied demand function as suggested in Rust (1987).
"""
import numpy as np
import pandas as pd

from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.pre_processing import select_model_parameters
from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.fix_point_alg import calc_fixp


def get_demand(init_dict, demand_dict, demand_params, max_num_iterations=2000):
    """
    Calculates the implied demand for a range of replacement costs
    for a certain number of buses over a certain time period.

    Parameters
    ----------
    init_dict : dict
        see :ref:`init_dict`.
    demand_dict : dict
        see :ref:`demand_dict`.
    demand_params : numpy.ndarray
        see :ref:`demand_params`

    Returns
    -------
    demand_results : pd.DataFrame
        see :ref:`demand_results`

    """
    params = demand_params.copy()
    (
        disc_fac,
        num_states,
        maint_func,
        maint_func_dev,
        num_params,
        scale,
    ) = select_model_parameters(init_dict)

    # Initialize the loop over the replacement costs
    rc_range = np.linspace(
        demand_dict["RC_lower_bound"],
        demand_dict["RC_upper_bound"],
        demand_dict["demand_evaluations"],
    )
    demand_results = pd.DataFrame(index=rc_range, columns=["demand", "success"])
    demand_results.index.name = "RC"

    for rc in rc_range:
        params[-num_params] = rc
        demand_results.loc[(rc), "success"] = "No"

        # solve the model for the given paramaters
        trans_mat = create_transition_matrix(num_states, params[:-num_params])

        obs_costs = calc_obs_costs(num_states, maint_func, params[-num_params:], scale)
        ev = calc_fixp(trans_mat, obs_costs, disc_fac)[0]
        p_choice = choice_prob_gumbel(ev, obs_costs, disc_fac)

        # calculate initial guess for pi and run contraction iterations
        pi_new = np.full((num_states, 2), 1 / (2 * num_states))
        tol = 1
        iteration = 1
        while tol >= demand_dict["tolerance"]:
            pi = pi_new
            pi_new = p_choice * (
                np.dot(trans_mat.T, pi[:, 0])
                + np.dot(np.tile(trans_mat[0, :], (num_states, 1)).T, pi[:, 1])
            ).reshape((num_states, 1))
            tol = np.max(np.abs(pi_new - pi))
            iteration += 1
            if iteration > max_num_iterations:
                break
            if tol < demand_dict["tolerance"]:
                demand_results.loc[(rc), "success"] = "Yes"

        demand_results.loc[(rc), "demand"] = (
            demand_dict["num_buses"] * demand_dict["num_periods"] * pi_new[:, 1].sum()
        )

    return demand_results
