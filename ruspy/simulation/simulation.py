"""
This module contains the main function to manage the simulation process. To simulate
a decision process in the model of John Rust's 1987 paper, it is sufficient to import
the function from this module and feed it with a init dictionary containing the
relevant variables.
"""
import numpy as np
import pandas as pd
from ruspy.simulation.simulation_auxiliary import simulate_strategy, get_unobs
from ruspy.estimation.estimation_cost_parameters import lin_cost, cost_func


def simulate(init_dict, ev_known, real_trans_mat, shock=None):
    """
    The main function to simulate a decision process in the theoretical framework of
    John Rust's 1987 paper. It reads the inputs from the initiation dictionary and
    draws the random variables. It then calls the main subfunction with all the
    relevant parameters. So far, the feature of a agent's misbelief on the underlying
    transition probabilities is not implemented.

    :param init_dict: A dictionary containing the following variables as keys:

        :seed: (Digits)      : The seed determines random draws.
        :buses: (int)        : Number of buses to be simulated.
        :beta: (float)       : Discount factor.
        :periods: (int)      : Number of periods to be simulated.
        :probs:              : A list or array of the true underlying transition
                               probabilities.
        :params:             : A list or array of the cost parameters shaping the cost
                               function.
        :maint_func: (string): The type of cost function, as string. Only linear
                               implemented so far.

    :return: The function returns the following objects:

        :df:         : A pandas dataframe containing for each observation the period,
                       state, decision and a Bus ID.
        :unobs:      : A three dimensional numpy array containing for each bus,
                       for each period random drawn utility for the decision to
                       maintain or replace the bus engine.
        :utilities:  : A two dimensional numpy array containing for each bus in each
                       period the utility as a float.
        :num_states: : A integer documenting the size of the state space.
    """
    if "seed" in init_dict.keys():
        np.random.seed(init_dict["seed"])
    num_buses = init_dict["buses"]
    beta = init_dict["beta"]
    num_periods = init_dict["periods"]
    params = np.array(init_dict["params"])
    if init_dict["maint_func"] == "linear":
        maint_func = lin_cost
    else:
        maint_func = lin_cost
    if ev_known.shape[0] != real_trans_mat.shape[0]:
        raise ValueError(
            "The transition matrix and the expected value of the agent "
            "need to have the same size."
        )
    num_states = ev_known.shape[0]
    unobs = get_unobs(shock, num_buses, num_periods)
    increments = get_increments(real_trans_mat, num_periods, num_buses)
    costs = cost_func(num_states, maint_func, params)
    states = np.zeros((num_buses, num_periods), dtype=int)
    decisions = np.zeros((num_buses, num_periods), dtype=int)
    utilities = np.zeros((num_buses, num_periods), dtype=float)
    states, decisions, utilities = simulate_strategy(
        num_buses,
        states,
        decisions,
        utilities,
        costs,
        ev_known,
        increments,
        num_periods,
        beta,
        unobs,
    )

    df = pd.DataFrame({"state": states.flatten(), "decision": decisions.flatten()})
    bus_id = np.arange(1, num_buses + 1).repeat(num_periods).astype(int)
    df["Bus_ID"] = bus_id
    period = np.array([])
    for _ in range(num_buses):
        period = np.append(period, np.arange(num_periods))
    df["period"] = period.astype(int)
    return df, unobs, utilities


def get_increments(real_trans_mat, num_periods, num_buses):
    num_states = real_trans_mat.shape[0]
    increments = np.zeros(shape=(num_states, num_buses, num_periods))
    for s in range(num_states):
        max_state = np.max(real_trans_mat[s, :].nonzero())
        p = real_trans_mat[s, s : (max_state + 1)]  # noqa: E203
        increments[s, :, :] = np.random.choice(
            len(p), size=(num_buses, num_periods), p=p
        )
    return increments
