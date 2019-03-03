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
    num_states = 100
    start_period = 0
    states = np.zeros((num_buses, num_periods), dtype=int)
    decisions = np.zeros((num_buses, num_periods), dtype=int)
    utilities = np.zeros((num_buses, num_periods), dtype=float)
    while start_period < num_periods - 1:
        num_states = 2 * num_states
        known_trans_mat = create_transition_matrix(num_states, known_trans)
        costs = myopic_costs(num_states, maint_func, params)
        ev = calc_fixp(num_states, known_trans_mat, costs, beta)
        states, decisions, utilities, start_period = \
            simulate_strategy_loop(num_buses, states, decisions, utilities, costs, ev, increments,
                                   num_states, start_period, num_periods, beta, unobs)
    return states, decisions, utilities, num_states



@numba.jit(nopython=True)
def simulate_strategy_loop(num_buses, states, decisions, utilities, costs,
                           ev, increments, num_states, start_period, num_periods, beta, unobs):
    """

    :param num_buses:
    :type num_buses: int
    :param states:
    :param decisions:
    :param utilities:
    :param costs:
    :param ev:
    :param increments:
    :param num_states:
    :param start_period:
    :param num_periods:
    :param beta:
    :param unobs:
    :return:
    """
    need_size = False
    for period in range(start_period, num_periods):
        for bus in range(num_buses):
            old_state = states[bus, period]
            if (- costs[old_state, 0] + unobs[bus, period, 0] + beta * ev[old_state]) >\
                    (- costs[0, 0] - costs[0, 1] + unobs[bus, period, 1] + beta * ev[0]):
                decision = 0
                utility = - costs[old_state, 0] + unobs[bus, period, 0]
                new_state = old_state + increments[period, bus]
            else:
                decision = 1
                utility = - costs[0, 0] - costs[0, 1] + unobs[bus, period, 1]
                new_state = increments[period, bus]

            decisions[bus, period] = decision
            utilities[bus, period] = utility
            if period < num_periods - 1:
                if new_state > (num_states / 2):
                    need_size = True
                states[bus, period + 1] = new_state
        if need_size:
            return states, decisions, utilities, period
    return states, decisions, utilities, period


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


