import numpy as np
import numba
from ruspy.estimation.estimation_auxiliary import calc_fixp
from ruspy.estimation.estimation_auxiliary import create_transition_matrix
from ruspy.estimation.estimation_auxiliary import myopic_costs


def simulate_strategy(known_trans, increments, num_buses, num_periods, params, beta, unobs, maint_func):
    """

    :param known_trans:
    :param increments:
    :param num_buses:
    :param num_periods:
    :param params:
    :param beta:
    :param unobs:
    :param maint_func:
    :return:
    """
    num_states = 500
    known_trans_mat = create_transition_matrix(num_states, known_trans)
    costs = myopic_costs(num_states, maint_func, params)
    ev = calc_fixp(num_states, known_trans_mat, costs, beta)
    states_start = np.zeros((num_buses, num_periods), dtype=int)
    decisions_start = np.zeros((num_buses, num_periods), dtype=int)
    utilities_start = np.zeros((num_buses, num_periods), dtype=float)
    return simulate_strategy_loop(num_buses, states_start, decisions_start, utilities_start, costs,
                                   ev, increments, num_periods, beta, unobs)


@numba.jit(nopython=True)
def simulate_strategy_loop(num_buses, states, decisions, utilities, costs, ev, increments, num_periods, beta, unobs):
    """

    :param num_buses:
    :param states:
    :param decisions:
    :param utilities:
    :param costs:
    :param ev:
    :param increments:
    :param num_periods:
    :param beta:
    :param unobs:
    :return:
    """
    for bus in range(num_buses):
        for i in range(0, num_periods):
            old_state = states[bus, i]
            if (- costs[old_state, 0] + unobs[bus, i, 0] + beta * ev[old_state]) >\
                    (- costs[0, 0] - costs[0, 1] + unobs[bus, i, 1] + beta * ev[0]):
                decision = 0
                utility = - costs[old_state, 0] + unobs[bus, i, 0]
                new_state = old_state + increments[i, bus]
            else:
                decision = 1
                utility = - costs[0, 0] - costs[0, 1] + unobs[bus, i, 1]
                new_state = increments[i, bus]

            decisions[bus, i] = decision
            utilities[bus, i] = utility
            if i < num_periods - 1:
                states[bus, i + 1] = new_state
    return states, decisions, utilities


@numba.jit(nopython=True)
def discount_utility(v_disc, num_buses, steps, num_points, utilities, beta):
    """
    
    :param v_disc:
    :param num_buses:
    :param steps:
    :param num_points:
    :param utilities:
    :param beta:
    :return:
    """
    for point in range(num_points):
        v = 0
        for i in range(point * steps):
            v += (beta ** i) * np.sum(utilities[:, i])
        v_disc[point] = v / num_buses
    return v_disc


