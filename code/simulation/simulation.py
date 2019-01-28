import numpy as np
import mpmath as mp
import pandas as pd
from simulation.simulation_auxiliary import simulate_strategy
from estimation.estimation_auxiliary import lin_cost


def simulate(init_dict):
    np.random.seed(init_dict['simulation']['seed'])
    num_buses = init_dict['simulation']['buses']
    beta = init_dict['simulation']['beta']
    num_periods = init_dict['simulation']['periods']
    num_states = init_dict['simulation']['states']
    unobs = np.random.gumbel(loc=-mp.euler, scale=1, size=[num_periods, 2])
    zurcher_trans = np.array(init_dict['simulation']['probs'])
    params = np.array(init_dict['simulation']['params'])
    states, decisions, utilities = simulate_strategy(zurcher_trans, zurcher_trans, num_buses, num_periods,
                                                     num_states, params, beta, unobs, lin_cost)
    df = pd.DataFrame({'state' : states.flatten(), 'decision' : decisions.flatten()})
    df['period'] = np.arange(num_periods).repeat(num_buses).astype(int)
    bus_id = np.array([])
    for i in range(1, num_buses + 1):
        bus_id = np.append(bus_id, np.full(num_periods, i, dtype=int))
    df['Bus_ID'] = bus_id
    return df, utilities
