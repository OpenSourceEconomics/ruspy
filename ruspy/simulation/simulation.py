"""
This module contains the main function to manage the simulation process. To simulate
a decision process in the model of John Rust's 1987 paper, it is sufficient to import
the function from this module and feed it with a init dictionary containing the
relevant variables.
"""
import numpy as np
import mpmath as mp
import pandas as pd
from ruspy.simulation.simulation_auxiliary import simulate_strategy
from ruspy.estimation.estimation_cost_parameters import lin_cost


def simulate(init_dict):
    """
    The main function to simulate a decision process in the theoretical framework of
    John Rust's 1987 paper. It reads the inputs from the initiation dictionary and
    draws the random variables. It then calls the main subfunction with all the
    relevant parameters. So far, the feature of a agent's misbelief on the underlying
    transition probabilities is not implemented.

    :param init_dict: A dictionary containing the following relevant variables as keys:

        :seed:       :  The seed for random draws.
        :buses:      :  Number of buses to be simulated.
        :beta:       :  Discount factor.
        :periods:    :  Number of periods to be simulated.
        :probs:      :  True underlying trans probabilities.
        :params:     :  Cost parameters shaping the cost function.
        :maint_func: :  The type of cost function, as string. Only linear implemented so
                        far.

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
    np.random.seed(init_dict['seed'])  #
    num_buses = init_dict['buses']  #
    beta = init_dict['beta']  #
    num_periods = init_dict['periods'] #
    real_trans = np.array(init_dict['probs']) #
    known_trans = np.array(init_dict['probs'])  # right know no misbeliefs
    params = np.array(init_dict['params'])  #
    if init_dict['maint_func'] == 'linear':
        maint_func = lin_cost
    else:
        maint_func = lin_cost
    unobs = np.random.gumbel(loc=-mp.euler, size=[num_buses, num_periods, 2])
    increments = np.random.choice(len(real_trans), size=(num_buses, num_periods), p=real_trans)
    states, decisions, utilities, num_states = \
        simulate_strategy(known_trans, increments, num_buses, num_periods, params,
                          beta, unobs, maint_func)
    df = pd.DataFrame({'state': states.flatten(), 'decision': decisions.flatten()})
    df['period'] = np.arange(num_periods).repeat(num_buses).astype(int)
    bus_id = np.array([])
    for i in range(1, num_buses + 1):
        bus_id = np.append(bus_id, np.full(num_periods, i, dtype=int))
    df['Bus_ID'] = bus_id
    return df, unobs, utilities, num_states
