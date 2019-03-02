import numpy as np
import mpmath as mp
import pandas as pd
from ruspy.simulation.simulation_auxiliary import simulate_strategy
from ruspy.estimation.estimation_auxiliary import lin_cost


def simulate(init_dict):
    np.random.seed(init_dict['seed'])
    num_buses = init_dict['buses']
    beta = init_dict['beta']
    num_periods = init_dict['periods']
    if init_dict['maint_func'] == 'linear':
        maint_func = lin_cost
    else:
        maint_func = lin_cost
    unobs = np.random.gumbel(loc=-mp.euler, size=[num_buses, num_periods, 2])
    known_trans = np.array(init_dict['probs']) # right know no misbeliefs
    real_trans = np.array(init_dict['probs'])
    increments = np.random.choice(len(real_trans), size=(num_periods, num_buses), p=real_trans)  # need to switch sizes
    params = np.array(init_dict['params'])
    states, decisions, utilities, num_states = \
        simulate_strategy(known_trans, increments, num_buses, num_periods, params, beta, unobs, maint_func)
    df = pd.DataFrame({'state': states.flatten(), 'decision': decisions.flatten()})
    df['period'] = np.arange(num_periods).repeat(num_buses).astype(int)
    bus_id = np.array([])
    for i in range(1, num_buses + 1):
        bus_id = np.append(bus_id, np.full(num_periods, i, dtype=int))
    df['Bus_ID'] = bus_id
    return df, unobs, utilities, num_states
