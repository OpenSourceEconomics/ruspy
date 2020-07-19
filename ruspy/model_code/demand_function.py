"""
We calculate and plot the implied demand function as suggested in Rust (1987).
Further we add the option to plot uncertainty around it.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ruspy.estimation.est_cost_params import get_ev
from ruspy.estimation.estimation_interface import select_model_parameters
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.model_code.choice_probabilities import choice_prob_gumbel
from ruspy.model_code.cost_functions import calc_obs_costs


def get_demand(init_dict, demand_dict, demand_params):
    """
    Calculates the implied demand for a range of replacement costs
    for a certain number of buses over a certain time period.

    Parameters
    ----------
    init_dict : dict
        see :ref:`init_dict`.
    demand_dict : dict
        see :ref:`demand_dict`.
    demand_params : np.array
        holds all model parameters for which the implied demand shall be calulated.
        The transition probabilities are the first elements, followed by the
        replacement cost parameter and then by the rest of the cost parameters.

    Returns
    -------
    demand_results : pd.DataFrame
        Index is the replacement cost. For each of those parameters there is the
        demand calculated and whether a dummy saying whether the fixed point
        algorithm exited successfully.

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

        # slove the model for the given paramaters
        trans_mat = create_transition_matrix(num_states, params[:-num_params])

        obs_costs = calc_obs_costs(num_states, maint_func, params[-num_params:], scale)
        ev = get_ev(params[-num_params], trans_mat, obs_costs, disc_fac, alg_details)[0]
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
            iteration = +1
            if iteration > 200:
                break
            if tol < demand_dict["tolerance"]:
                demand_results.loc[(rc), "success"] = "Yes"

        demand_results.loc[(rc), "demand"] = (
            demand_dict["num_buses"] * demand_dict["num_periods"] * pi_new[:, 1].sum()
        )

        conditional_prob = pi_new / pi_new.sum(axis=0)

    return demand_results, conditional_prob


# demand_results.reset_index(inplace=True)
# fig, axis = plt.subplots()
# axis.plot(
#     "RC", "demand", data=demand_results, color="red",
# )
# axis.plot(
#     "RC", "demand_gauss", data=demand_results, color="blue",
# )


# uncertainty
results = np.load("solution_MPEC.npy")
# results = pd.read_pickle("results_ipopt").loc[
#     (0.9999, slice(None), slice(None), "MPEC"), :]
# results = results[["theta_30", "theta_31", "theta_32", "theta_33",
#                   "RC", "theta_11"]].astype(float).to_numpy()

params = np.array([0.3919, 0.5953, 1 - 0.3919 - 0.5953, 11.7257, 2.4569])
init_dict = {
    "model_specifications": {
        "discount_factor": 0.9999,
        "number_states": 175,
        "maint_cost_func": "linear",
        "cost_scale": 1e-3,
    },
    "optimizer": {
        "approach": "NFXP",
        "algorithm": "scipy_L-BFGS-B",
        "gradient": "Yes",
        "params": params,
    },
    "alg_details": {"threshold": 1e-13, "switch_tol": 1e-2},
}
demand_dict = {
    "RC_lower_bound": 2,
    "RC_upper_bound": 15,
    "demand_evaluations": 1000,
    "tolerance": 1e-10,
    "num_periods": 12,
    "num_buses": 1,
}

true_demand = (
    get_demand(init_dict, demand_dict, params)["demand"].astype(float).to_numpy()
)

rc_range = np.linspace(
    demand_dict["RC_lower_bound"],
    demand_dict["RC_upper_bound"],
    demand_dict["demand_evaluations"],
)
demand = pd.DataFrame(index=rc_range)
demand.index.name = "RC"
for j in range(len(results)):
    params[-2:] = results[j, :]
    demand[str(j)] = get_demand(init_dict, demand_dict, params)["demand"]

data = demand.astype(float).to_numpy()
mean = data.mean(axis=1)
std = data.std(axis=1)
lower_percentile = np.percentile(data, 2.5, axis=1)
upper_percentile = np.percentile(data, 97.5, axis=1)
lower_bound = 2 * mean - upper_percentile
upper_bound = 2 * mean - lower_percentile

fig, axis = plt.subplots()
axis.plot(rc_range, true_demand, color="black")
axis.plot(rc_range, mean, color="blue")
axis.plot(rc_range, upper_bound, rc_range, lower_bound, color="red")
axis.fill_between(rc_range, upper_bound, lower_bound, color="0.5")


# delete after
demand_dict = {
    "RC_lower_bound": 1,
    "RC_upper_bound": 15,
    "demand_evaluations": 200,
    "tolerance": 1e-10,
    "num_periods": 1,
    "num_buses": 1,
}

params = np.array([0.3919, 0.5953, 1 - 0.3919 - 0.5953, 10.075, 2.293])
params = np.array(
    [0.1972222222222222, 0.7888888888888889, 0.0138888888888889, 11.0, 3.0]
)
params = np.array([0.1191, 0.5762, 0.2868, 0.0158, 0.00209999999999997, 10.896, 1.1732])
