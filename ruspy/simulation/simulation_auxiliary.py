import numpy as np
import numba
from ruspy.estimation.estimation_cost_parameters import calc_fixp
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix
from ruspy.estimation.estimation_cost_parameters import cost_func


def simulate_strategy(
    known_trans, increments, num_buses, num_periods, params, beta, unobs, maint_func
):
    """
    This function manages the simulation process. It initializes the auxiliary
    variables and calls therefore the subfuctions from estimation auxiliary. It then
    calls the decision loop, written for numba. As the state size of the fixed point
    needs to be a lot larger than the actual state, the size is doubled, if the loop
    hasn't run yet through all the periods.

    :param known_trans: A numpy array containing the transition probabilities the agent
                        assumes.
    :param increments:  A two dimensional numpy array containing for each bus in each
                        period a random drawn state increase as integer.
    :param num_buses:   The number of buses to be simulated.
    :type num_buses:    int
    :param num_periods: The number of periods to be simulated.
    :type num_periods:  int
    :param params:      A numpy array containing the parameters shaping the cost
                        function.
    :param beta:        The discount factor.
    :type beta:         float
    :param unobs:       A three dimensional numpy array containing for each bus,
                        for each period for the decision to maintain or replace the
                        bus engine a random drawn utility as float.
    :param maint_func:  The maintenance cost function. Only linear implemented so
                        far.

    :return: The function returns the following objects:

        :states:           : A two dimensional numpy array containing for each bus in
                             each period the state as an integer.
        :decisions:        : A two dimensional numpy array containing for each bus in
                             each period the decision as an integer.
        :utilities:        : A two dimensional numpy array containing for each bus in
                             each period the utility as a float.
        :num_states: (int) : The size of the state space.
    """
    num_states = int(200)
    start_period = int(0)
    states = np.zeros((num_buses, num_periods), dtype=int)
    decisions = np.zeros((num_buses, num_periods), dtype=int)
    utilities = np.zeros((num_buses, num_periods), dtype=float)
    while start_period < num_periods - 1:
        num_states = 2 * num_states
        known_trans_mat = create_transition_matrix(num_states, known_trans)
        costs = cost_func(num_states, maint_func, params)
        ev = calc_fixp(num_states, known_trans_mat, costs, beta)
        states, decisions, utilities, start_period = simulate_strategy_loop(
            num_buses,
            states,
            decisions,
            utilities,
            costs,
            ev,
            increments,
            num_states,
            start_period,
            num_periods,
            beta,
            unobs,
        )
    return states, decisions, utilities, num_states


@numba.jit(nopython=True)
def simulate_strategy_loop(
    num_buses,
    states,
    decisions,
    utilities,
    costs,
    ev,
    increments,
    num_states,
    start_period,
    num_periods,
    beta,
    unobs,
):
    """
    This function simulates the decision strategy, as long as the current period is
    below the number of periods and the current highest state of a bus is in the
    first half of the state space.

    :param num_buses:    The number of buses to be simulated.
    :type num_buses:     int
    :param states:       A two dimensional numpy array containing for each bus in each
                         period the state as integer. Default value for each bus
                         in each period not yet simulated is zero.
    :param decisions:    A two dimensional numpy array containing for each bus in each
                         period the decision as integer. Default value for each bus
                         in each period not yet simulated is zero.
    :param utilities:    A two dimensional numpy array containing for each bus in each
                         period the utility as float. Default value for each bus
                         in each period not yet simulated is zero.
    :param costs:        A two dimensional float numpy array containing for each
                         state the cost to maintain in the first and to replace the bus
                         engine in the second column.
    :param ev:           A numpy array containing for each state the expected value
                         fixed point.total
    :param increments:   A two dimensional numpy array containing for each bus in each
                         period a random drawn state increase as integer.
    :param num_states:   The size of the state space.
    :type num_states:    int
    :param start_period: The start period for simulation the decisions.
    :type start_period:  int
    :param num_periods:  The number of periods to be simulated.
    :type num_periods:   int
    :param beta:         The discount factor.
    :type beta:          float
    :param unobs:        A three dimensional numpy array containing for each bus,
                         for each period for the decision to maintain or replace the
                         bus engine a random drawn utility as float.

    :return: The function returns the following objects:

        :states:           : A two dimensional numpy array containing for each bus in
                             each period the state as an integer.
        :decisions:        : A two dimensional numpy array containing for each bus in
                             each period the decision as an integer.
        :utilities:        : A two dimensional numpy array containing for each bus in
                             each period the utility as a float.
        :num_states: (int) : The size of the state space.

    """
    need_size = bool(False)
    period = int(0)
    for period in range(start_period, num_periods):
        for bus in range(num_buses):
            old_state = states[bus, period]
            if (-costs[old_state, 0] + unobs[bus, period, 0] + beta * ev[old_state]) > (
                -costs[0, 0] - costs[0, 1] + unobs[bus, period, 1] + beta * ev[0]
            ):
                decision = 0
                utility = -costs[old_state, 0] + unobs[bus, period, 0]
                new_state = old_state + increments[bus, period]
            else:
                decision = 1
                utility = -costs[0, 0] - costs[0, 1] + unobs[bus, period, 1]
                new_state = increments[bus, period]

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
def simulate_strategy_loop_known(
    num_buses,
    states,
    decisions,
    utilities,
    costs,
    ev,
    increments,
    num_periods,
    beta,
    unobs,
):
    """
    This function simulates the decision strategy, as long as the current period is
    below the number of periods and the current highest state of a bus is in the
    first half of the state space.

    :param num_buses:    The number of buses to be simulated.
    :type num_buses:     int
    :param states:       A two dimensional numpy array containing for each bus in each
                         period the state as integer. Default value for each bus
                         in each period not yet simulated is zero.
    :param decisions:    A two dimensional numpy array containing for each bus in each
                         period the decision as integer. Default value for each bus
                         in each period not yet simulated is zero.
    :param utilities:    A two dimensional numpy array containing for each bus in each
                         period the utility as float. Default value for each bus
                         in each period not yet simulated is zero.
    :param costs:        A two dimensional float numpy array containing for each
                         state the cost to maintain in the first and to replace the bus
                         engine in the second column.
    :param ev:           A numpy array containing for each state the expected value
                         fixed point.total
    :param increments:   A two dimensional numpy array containing for each bus in each
                         period a random drawn state increase as integer.
    :param num_periods:  The number of periods to be simulated.
    :type num_periods:   int
    :param beta:         The discount factor.
    :type beta:          float
    :param unobs:        A three dimensional numpy array containing for each bus,
                         for each period for the decision to maintain or replace the
                         bus engine a random drawn utility as float.

    :return: The function returns the following objects:

        :states:           : A two dimensional numpy array containing for each bus in
                             each period the state as an integer.
        :decisions:        : A two dimensional numpy array containing for each bus in
                             each period the decision as an integer.
        :utilities:        : A two dimensional numpy array containing for each bus in
                             each period the utility as a float.
        :num_states: (int) : The size of the state space.

    """
    for period in range(num_periods):
        for bus in range(num_buses):

            old_state = states[bus, period]
            if (-costs[old_state, 0] + unobs[bus, period, 0] + beta * ev[old_state]) > (
                -costs[0, 0] - costs[0, 1] + unobs[bus, period, 1] + beta * ev[0]
            ):
                decision = 0
                utility = -costs[old_state, 0] + unobs[bus, period, 0]
                new_state = old_state + increments[bus, period]
            else:
                decision = 1
                utility = -costs[0, 0] - costs[0, 1] + unobs[bus, period, 1]
                new_state = increments[bus, period]

            decisions[bus, period] = decision
            utilities[bus, period] = utility
            states[bus, period + 1] = new_state
    return states, decisions, utilities
