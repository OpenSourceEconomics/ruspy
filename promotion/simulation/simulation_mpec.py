"""
This module contains the loop to calculate the cost parameters for the simulated
from the original matlab code by Iskhakov et al. for beta equal to 0.975 and
the cost parameters plus transition paramters according to page 2225 of Judd & Su.
Only the starting values of 1 and 4 for the cost parameters and 0.2 for the
transition probabilities are used.
"""
import numpy as np
import pandas as pd
import scipy.io

from ruspy.estimation.estimation import estimate
from ruspy.estimation.estimation_mpec import estimate_mpec

# Load the simulated data for the true parameters as descriebd on page 2225 in Judd & Su
# This data is taken from their original matlab code
# The data is simulated with beta equal to 0.975
mat = scipy.io.loadmat("promotion/simulation/RustBusTableXSimDataMC250_beta97500")
# Load the solutions from the matlab for the simulated data using starting values (1, 4)
solution_NFXP_iskhakov = scipy.io.loadmat(
    "promotion/simulation/solution_NFXP_iskhakov"
)["result_jr87_97500"]

# Initiate the loop
init_dict = {
    "model_specifications": {
        "discount_factor": 0.975,
        "number_states": 175,
        "maint_cost_func": "linear",
        "cost_scale": 1e-3,
    },
    "optimizer": {
        "optimizer_name": "BFGS",
        "use_gradient": "yes",
        "use_search_bounds": "no",
        "start_values": np.array([4, 1]),
    },
}

num_runs = 5
num_params = 2  # number of cost parameters

solution_NFXP = np.zeros((num_runs, num_params))
solution_MPEC = np.zeros((num_runs, num_params))
for j in range(0, num_runs):
    # Prepare the raw matlab data
    number_buses = 50
    number_periods = 120
    state = mat["MC_xt"][:, :, j] - 1
    decision = mat["MC_dt"][:, :, j]
    usage = mat["MC_dx"][:-1, :, j] - 1
    first_usage = np.full((1, usage.shape[1]), np.nan)
    usage = np.vstack((first_usage, usage))

    data = pd.DataFrame()
    state_new = state[:, 0]
    decision_new = decision[:, 0]
    usage_new = usage[:, 0]

    for i in range(0, len(state[0, :]) - 1):
        state_new = np.hstack((state_new, state[:, i + 1]))
        decision_new = np.hstack((decision_new, decision[:, i + 1]))
        usage_new = np.hstack((usage_new, usage[:, i + 1]))

    data["state"] = state_new
    data["decision"] = decision_new
    data["usage"] = usage_new

    iterables = [range(number_buses), range(number_periods)]
    index = pd.MultiIndex.from_product(iterables, names=["Bus_ID", "period"])
    data.set_index(index, inplace=True)

    df = data

    # Calculate ruspy NFXP and MPEC
    result_transitions, result_fixp = estimate(init_dict, df)
    mpec_result_transitions, mpec_cost_parameters = estimate_mpec(init_dict, df)
    solution_NFXP[j, :] = np.array(result_fixp["x"])
    solution_MPEC[j, :] = np.array(mpec_cost_parameters[-num_params:])

# Save the results
np.save("./promotion/simulation/solution_NFXP", solution_NFXP)
np.save("./promotion/simulation/solution_MPEC", solution_MPEC)
