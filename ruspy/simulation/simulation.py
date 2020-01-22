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
    """Simulating the decision process of Harold Zurcher.

    The main function to simulate a decision process in the theoretical framework of
    John Rust's 1987 paper. It reads the inputs from the initiation dictionary and
    draws the random variables. It then calls the main subfunction with all the
    relevant parameters. So far, the feature of a agent's misbelief on the underlying
    transition probabilities is not implemented.

    Parameters
    ----------
    init_dict : dictionary
        See :ref:`sim_init_dict`
    ev_known : numpy.array
        See :ref:`ev`
    costs : numpy.array
        See ref:`costs`
    trans_mat : numpy.array
        See ref:`trans_mat`
    shock : tuple
        See ref:`shock`

    Returns
    -------
    df : pandas.DataFrame
        See ref:`sim_results`

    """
    if "seed" in init_dict.keys():
        seed = init_dict["seed"]
    else:
        seed = np.random.randint(1, 100000)
    num_buses = init_dict["buses"]
    disc_fac = init_dict["disc_fac"]
    num_periods = init_dict["periods"]
    if ev_known.shape[0] != trans_mat.shape[0]:
        raise ValueError(
            "The transition matrix and the expected value of the agent "
            "need to have the same size."
        )
    maint_shock_dist_name, repl_shock_dist_name, loc_scale = get_unobs_data(shock)
    states, decisions, utilities, usage = simulate_strategy(
        num_periods,
        num_buses,
        costs,
        ev_known,
        trans_mat,
        disc_fac,
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
            "usage": usage.flatten(),
        },
    )

    return df
