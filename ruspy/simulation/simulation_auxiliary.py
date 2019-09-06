import pandas as pd
import numpy as np
import numba
import scipy.stats as stats


@numba.jit(nopython=True)
def simulate_strategy(
    bus, states, decisions, utilities, costs, ev, increments, beta, unobs
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
    num_states = ev.shape[0]
    num_periods = decisions.shape[1]
    for period in range(num_periods):
        old_state = states[bus, period]
        value_replace = (
            -costs[0, 0] - costs[0, 1] + unobs[bus, period, 1] + beta * ev[0]
        )
        value_maintain = (
            -costs[old_state, 0] + unobs[bus, period, 0] + beta * ev[old_state]
        )
        if value_maintain > value_replace:
            decision = 0
            utility = -costs[old_state, 0] + unobs[bus, period, 0]
            new_state = old_state + increments[old_state, period]
        else:
            decision = 1
            utility = -costs[0, 0] - costs[0, 1] + unobs[bus, period, 1]
            new_state = increments[0, period]

        decisions[bus, period] = decision
        utilities[bus, period] = utility
        if period < num_periods - 1:
            states[bus, period + 1] = new_state
        if new_state > num_states - 10:
            raise ValueError("State space is too small.")
    return states, decisions, utilities


def get_unobs(shock, num_buses, num_periods):
    """
    :param shock            : A tuple of pandas.Series, where each Series name is
                             the scipy distribution function and the data is the loc
                             and scale specification.
    :param num_buses        : Number of buses to be simulated.
    :param num_periods      : Number of periods to be simulated.

    :return: A 3d numpy array containing for each bus in each period a random shock
    for each decision.
    """
    unobs = np.empty(shape=(num_buses, num_periods, 2), dtype=float)
    # If no specification on the shocks is given. A right skewed gumbel distribution
    # with mean 0 and scale pi^2/6 is assumed for each shock component.
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


def get_increments(trans_mat, num_periods):
    num_states = trans_mat.shape[0]
    increments = np.zeros(shape=(num_states, num_periods), dtype=int)
    for s in range(num_states):
        max_state = np.max(trans_mat[s, :].nonzero())
        p = trans_mat[s, s: (max_state + 1)]  # noqa: E203
        increments[s, :] = np.random.choice(len(p), size=num_periods, p=p)
    return increments
