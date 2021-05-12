"""
This module contains the main function to manage the simulation process. To simulate
a decision process in the model of John Rust's 1987 paper, it is sufficient to import
the function from this module and feed it with a init dictionary containing the
relevant variables.
"""
import warnings

import numpy as np
import pandas as pd

from ruspy.simulation.simulation_funtions import simulate_strategy
from ruspy.simulation.simulation_funtions import (
    simulate_strategy_reduced_data_utilities,simulate_strategy_reduced_data_disc_utility
)


def simulate(init_dict, ev_known, costs, trans_mat, reduced_data=None):
    """Simulating the decision process of Harold Zurcher.

    The main function to simulate a decision process in the theoretical framework of
    John Rust's 1987 paper. It reads the inputs from the initiation dictionary and
    then calls the main subfunction with all the relevant parameters.

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
    reduced_data : string
        Keyword for simulation with reduced data usage.

    Returns
    -------
    df : pandas.DataFrame
        See :ref:`sim_results`
    """
    if "seed" in init_dict.keys():
        seed = init_dict["seed"]
    else:
        seed = np.random.randint(1, 100000)

    num_buses, disc_fac, num_periods = read_init_dict(init_dict)

    if ev_known.shape[0] != trans_mat.shape[0]:
        raise ValueError(
            "The transition matrix and the expected value of the agent "
            "need to have the same size."
        )

    out = create_output_by_keyword(
        num_periods, num_buses, costs, ev_known, trans_mat, disc_fac, seed, reduced_data
    )
    return out


def read_init_dict(init_dict):
    return init_dict["buses"], init_dict["discount_factor"], init_dict["periods"]


def create_output_by_keyword(
    num_periods, num_buses, costs, ev_known, trans_mat, disc_fac, seed, reduced_data
):

    if not reduced_data:
        out, absorbing_state = create_standard_output(
            num_periods, num_buses, costs, ev_known, trans_mat, disc_fac, seed
        )
    elif reduced_data == "utility":
        out, absorbing_state = simulate_strategy_reduced_data_utilities(
            num_periods,
            num_buses,
            costs,
            ev_known,
            trans_mat,
            disc_fac,
            seed,
        )
    elif reduced_data == "discounted utility":
        out, absorbing_state = simulate_strategy_reduced_data_disc_utility(
            num_periods,
            num_buses,
            costs,
            ev_known,
            trans_mat,
            disc_fac,
            seed,
        )

    else:
        raise ValueError(
            f"\"utility\" or \"discounted utility\" are the only valid keyword for "
            f"reduced_data. You "
            f"provided {reduced_data} "
        )

    warn_if_absorbing_state_reached(absorbing_state)
    return out


def create_standard_output(
    num_periods, num_buses, costs, ev_known, trans_mat, disc_fac, seed
):
    states, decisions, utilities, usage, absorbing_state = simulate_strategy(
        num_periods,
        num_buses,
        costs,
        ev_known,
        trans_mat,
        disc_fac,
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
    return df, absorbing_state


def warn_if_absorbing_state_reached(absorbing_state):
    if absorbing_state == 1:
        warnings.warn(
            """
                      For at least one bus in at least one time period the state
                      reached the highest possible grid point. This might confound
                      your results. Please consider increasing the grid size
                      until this messsage does not appear anymore.
                      """
        )
