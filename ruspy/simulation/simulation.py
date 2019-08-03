"""
This module contains the main function to manage the simulation process. To simulate
a decision process in the model of John Rust's 1987 paper, it is sufficient to import
the function from this module and feed it with a init dictionary containing the
relevant variables.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from ruspy.simulation.simulation_auxiliary import simulate_strategy
from ruspy.estimation.estimation_cost_parameters import lin_cost
from ruspy.estimation.estimation_cost_parameters import cost_func
from ruspy.simulation.simulation_auxiliary import simulate_strategy_loop_known


def simulate(init_dict, ev_known=None, shock=None):
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
    if "real_trans" in init_dict.keys():
        real_trans = np.array(init_dict["real_trans"])
    else:
        real_trans = np.array(init_dict["known_trans"])
    if init_dict["maint_func"] == "linear":
        maint_func = lin_cost
    else:
        maint_func = lin_cost
    unobs = get_unobs(shock, num_buses, num_periods)
    increments = np.random.choice(
        len(real_trans), size=(num_buses, num_periods), p=real_trans
    )
    if ev_known is not None:
        # If there is already ev given, the auxiliary function is skipped and the
        # simulation is executed with no further increases of the state space. This
        # option is perfect if only one parameter in the setting is varied and
        # therefore the highest achievable state can be guessed.
        num_states = int(len(ev_known))
        costs = cost_func(num_states, maint_func, params)
        states = np.zeros((num_buses, num_periods), dtype=int)
        decisions = np.zeros((num_buses, num_periods), dtype=int)
        utilities = np.zeros((num_buses, num_periods), dtype=float)
        states, decisions, utilities = simulate_strategy_loop_known(
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
    else:
        known_trans = np.array(init_dict["known_trans"])
        states, decisions, utilities, num_states = simulate_strategy(
            known_trans,
            increments,
            num_buses,
            num_periods,
            params,
            beta,
            unobs,
            maint_func,
        )

    df = pd.DataFrame({"state": states.flatten(), "decision": decisions.flatten()})
    bus_id = np.arange(1, num_buses + 1).repeat(num_periods).astype(int)
    df["Bus_ID"] = bus_id
    period = np.array([])
    for _ in range(num_buses):
        period = np.append(period, np.arange(num_periods))
    df["period"] = period.astype(int)
    return df, unobs, utilities, num_states


def get_unobs(shock, num_buses, num_periods):
    unobs = np.empty(shape=(num_buses, num_periods, 2), dtype=float)
    shock = (
        (
            pd.Series(index=["loc"], data=[-np.euler_gamma], name="gumbel_r"),
            pd.Series(index=["loc"], data=[-np.euler_gamma], name="gumbel_r"),
        )
        if shock is None
        else shock
    )
    dist_func_shocks_maint = getattr(stats, shock[0].name)
    dist_func_shocks_repl = getattr(stats, shock[1].name)
    unobs[:, :, 0] = dist_func_shocks_maint.rvs(
        **shock[0], size=[num_buses, num_periods]
    )
    unobs[:, :, 1] = dist_func_shocks_repl.rvs(
        **shock[1], size=[num_buses, num_periods]
    )
    return unobs
