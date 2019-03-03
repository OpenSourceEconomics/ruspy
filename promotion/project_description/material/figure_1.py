import matplotlib.pyplot as plt
import numpy as np
import os
from ruspy.simulation.simulation import simulate
from ruspy.simulation.simulation_auxiliary import discount_utility
from ruspy.estimation.estimation_auxiliary import calc_fixp
from ruspy.estimation.estimation_auxiliary import lin_cost
from ruspy.estimation.estimation_auxiliary import myopic_costs
from ruspy.estimation.estimation_auxiliary import create_transition_matrix


def plot_convergence(init_dict):
    beta = init_dict['simulation']['beta']

    df, unobs, utilities, num_states = simulate(init_dict['simulation'])

    costs = myopic_costs(num_states, lin_cost, init_dict['simulation']['params'])
    trans_probs = np.array(init_dict['simulation']['probs'])
    trans_mat = create_transition_matrix(num_states, trans_probs)
    ev = calc_fixp(num_states, trans_mat, costs, beta)
    num_buses = init_dict['simulation']['buses']
    num_periods = init_dict['simulation']['periods']
    gridsize = init_dict['plot']['gridsize']
    num_points = int(num_periods/gridsize)

    v_calc = 0
    for i in range(num_buses):
        v_calc = v_calc + unobs[i, 0, 0] + ev[0]
    v_calc = v_calc / num_buses
    v_exp = list(np.full(num_points, v_calc))

    v_start = np.zeros(num_points)
    v_disc = list(discount_utility(v_start, num_buses, gridsize, num_points, utilities, beta))

    periods = list(np.arange(0, num_periods, gridsize))

    ax = plt.figure(figsize=(14, 6))

    ax1 = ax.add_subplot(111)

    ax1.set_ylim([0, -1500])

    ax1.set_ylabel(r"Value at time 0")
    ax1.set_xlabel(r"Periods")

    l1 = ax1.plot(periods, v_disc, color='blue')
    l2 = ax1.plot(periods, v_exp, color='orange')

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/figure_1.png', dpi=300)
