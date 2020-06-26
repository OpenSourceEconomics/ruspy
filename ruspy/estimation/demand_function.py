"""
We calculate and plot the implied demand function as suggested in Rust (1987).
Further we add the option to plot uncertainty around it.
"""
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ruspy.estimation.est_cost_params import get_ev
from ruspy.estimation.estimation_interface import select_model_parameters
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from ruspy.model_code.cost_functions import calc_obs_costs


def get_demand(init_dict, demand_dict, params):
    """
    Parameters
    ----------
    init_dict : dict
        DESCRIPTION.
    params : np.array
        holds all model parameters for which the implied demand shall be calulated.
        The transition probabilities are the first elements, followed by the
        replacement cost parameter and then by the rest of the cost parameters.

    Returns
    -------
    None.

    """

    (
        disc_fac,
        num_states,
        maint_func,
        maint_func_dev,
        num_params,
        scale,
    ) = select_model_parameters(init_dict)

    alg_details = {} if "alg_details" not in init_dict else init_dict["alg_details"]

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

        trans_mat = create_transition_matrix(num_states, params[:-num_params])

        obs_costs = calc_obs_costs(num_states, maint_func, params[-num_params:], scale)
        ev = get_ev(params[-num_params], trans_mat, obs_costs, disc_fac, alg_details)[0]
        p_choice = choice_prob_gumbel(ev, obs_costs, disc_fac)

        # calculate initial guess for pi
        z = 0.99999999
        choice_trans_prob = p_choice[:, 0] * trans_mat
        choice_trans_prob[:, 0] = 1 - choice_trans_prob[:, 1:].sum(axis=1)
        pi = np.linalg.solve(
            (np.eye(num_states) - z * choice_trans_prob).T,
            np.full(num_states, 1 - z) / num_states,
        )
        pi = pi / pi.sum()

        # check if initial guess is good enough
        tol = np.max(np.abs(pi - np.dot(pi.T, choice_trans_prob)))
        if tol >= demand_dict["tolerance"]:
            # refine guess by contraction iterations
            iterations = 1
            while iterations < demand_dict["max_iterations"]:
                pi = np.matmul(pi, choice_trans_prob)
                iterations += 1
                tol = np.max(np.abs(pi - np.dot(pi.T, choice_trans_prob)))
                if tol < demand_dict["tolerance"]:
                    demand_results.loc[(rc), "success"] = "Yes"
                    break
        else:
            demand_results.loc[(rc), "success"] = "Yes"

        pi_temp = pi.copy()
        pi_temp[0] = trans_mat[0, 0] * p_choice[0, 0] * pi[0]
        pi = np.vstack((pi, pi_temp * (1 - p_choice[:, 0]) / p_choice[:, 0])).T

        demand_results.loc[(rc), "demand"] = (
            demand_dict["num_buses"] * demand_dict["num_periods"] * pi[:, 1].sum()
        )

    return demand_results


# demand_results.reset_index(inplace=True)
# fig, axis = plt.subplots()
# axis.plot(
#     "RC",
#     "demand",
#     data=demand_results,
#     color="red",
#     )

# delete after
demand_dict = {
    "RC_lower_bound": 1,
    "RC_upper_bound": 15,
    "demand_evaluations": 200,
    "max_iterations": 20,
    "tolerance": 0.00000001,
    "num_periods": 12,
    "num_buses": 1,
}

params = np.array([0.3919, 0.5953, 1 - 0.3919 - 0.5953, 10.075, 2.293])
