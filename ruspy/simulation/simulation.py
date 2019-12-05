"""
This module contains the main function to manage the simulation process. To simulate
a decision process in the model of John Rust's 1987 paper, it is sufficient to import
the function from this module and feed it with a init dictionary containing the
relevant variables.
"""
import numpy as np
import pandas as pd

from ruspy.simulation.simulation_auxiliary import get_unobs_data
from ruspy.simulation.simulation_auxiliary import simulate_strategy


def simulate(init_dict, ev_known, costs, trans_mat, shock=None):
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
        :params:             : A list or array of the cost parameters shaping the cost

    :param ev_known         : A 1d array containing the agent's expectation of the
                              value function in each state of dimension (num_states)
    :param trans_mat        : The transition matrix governing the discrete space Markov
                             decision process of dimension (num_states, num_states).
    :param shock            : A tuple of pandas.Series, where each Series name is
                             the scipy distribution function and the data is the loc
                             and scale specification.

    :return: The function returns the following objects:

        :df:         : A pandas dataframe containing for each observation the period,
                       state, decision and a Bus ID.
        :unobs:      : A three dimensional numpy array containing for each bus,
                       for each period random drawn utility for the decision to
                       maintain or replace the bus engine.
        :utilities:  : A two dimensional numpy array containing for each bus in each
                       period the utility as a float.
    """
    if "seed" in init_dict.keys():
        seed = init_dict["seed"]
    else:
        seed = np.random.randint(1, 100000)
    num_buses = init_dict["buses"]
    beta = init_dict["beta"]
    num_periods = init_dict["periods"]
    if ev_known.shape[0] != trans_mat.shape[0]:
        raise ValueError(
            "The transition matrix and the expected value of the agent "
            "need to have the same size."
        )
    maint_shock_dist_name, repl_shock_dist_name, loc_scale = get_unobs_data(shock)
    states, decisions, utilities = simulate_strategy(
        num_periods,
        num_buses,
        costs,
        ev_known,
        trans_mat,
        beta,
        maint_shock_dist_name,
        repl_shock_dist_name,
        loc_scale,
        seed,
    )
    bus_ids = np.arange(num_buses) + 1
    periods = np.arange(num_periods)
    idx = pd.MultiIndex.from_product([bus_ids, periods], names=["Bus_ID", "period"])
    df = pd.DataFrame(
        index=idx,
        data={
            "state": states.flatten(),
            "decision": decisions.astype(np.uint8).flatten(),
            "utilities": utilities.flatten(),
        },
    )

    return df
