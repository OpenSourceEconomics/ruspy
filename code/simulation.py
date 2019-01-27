import numpy as np
import json
import mpmath as mp
from simulation_auxiliary import simulate_strategy
from estimation_auxiliary import lin_cost
from estimation_auxiliary import myopic_costs
from estimation_auxiliary import calc_fixp
from estimation_auxiliary import create_transition_matrix

np.random.seed(234)
num_buses = 1
beta = 0.9999
num_periods = 50000
num_states = 300
unobs = np.random.gumbel(loc = -mp.euler, scale=1, size=(num_periods, 2))
zurcher_trans, estimation_results = json.load(open('result_5000.json', 'r'))
zurcher_trans = np.array(zurcher_trans)
params = np.array(estimation_results)
states, decisions, utilities = simulate_strategy(zurcher_trans, zurcher_trans, num_buses, num_periods, num_states, params, beta, unobs, lin_cost)
num_states_used = np.amax(states)
repl_costs = myopic_costs(num_states_used, lin_cost, params)[:, 1]
m_costs = myopic_costs(num_states_used, lin_cost, params)[:, 0]
trans_mat = create_transition_matrix(num_states, zurcher_trans)
ev = calc_fixp(num_states, trans_mat, lin_cost, params, beta, threshold=1e-6)
json.dump([states.tolist(), decisions.tolist(), utilities.tolist(), m_costs.tolist(),
           repl_costs.tolist(), ev.tolist(), unobs.tolist()],
          open('simulation.json', 'w'))


"""
v_ges = list()
for i in range(1, 100000):
    print(i)
    v_0 = 0
    num_periods = i
    if i < 25:
        num_states = i * 20
    else:
        num_states = 500
    costs = myopic_costs(num_states, lin_cost, params)
    df = 
    for j in range(num_periods):
        if df.loc[j, 'decision'] == 1:
            u = -costs[0, 0] - costs[0, 1]
        else:
            u = -costs[df.loc[j, 'state'], 0]

        v_0 = v_0 + (beta ** j) * u
    v_ges += [v_0]
num_states = 500
trans_mat = create_transition_matrix(num_states, zurcher_trans)
ev = calc_fixp(num_states, trans_mat, lin_cost, params, beta)
json.dump([v_ges, ev[0]],
              open('result_sim.json', 'w'))
print(v_ges[98], ev[0])
"""


