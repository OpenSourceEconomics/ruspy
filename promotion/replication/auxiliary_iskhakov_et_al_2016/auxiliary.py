import numpy as np
import pandas as pd


def process_data(df, run, number_buses, number_periods):
    state = df["MC_xt"][:, :, run] - 1
    decision = df["MC_dt"][:, :, run]
    usage = df["MC_dx"][:-1, :, run] - 1
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

    return data


def process_result(approach, transition_result, cost_result, time, number_states):
    if approach == "NFXP":
        result = np.concatenate((cost_result["x"], transition_result["x"][:4], [time]))

        for name in [
            "status",
            "n_iterations",
            "n_evaluations",
            "n_contraction_steps",
            "n_newt_kant_steps",
        ]:
            result = np.concatenate((result, np.array([cost_result[name]])))

    else:
        result = np.concatenate(
            (cost_result["x"][number_states:], transition_result["x"][:4], [time])
        )

        for name in ["status", "n_iterations", "n_evaluations"]:
            result = np.concatenate((result, np.array([cost_result[name]])))

    return result
