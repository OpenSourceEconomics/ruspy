import pandas as pd
import numpy as np
import numba


@numba.jit(nopython=True)
def simulate_strategy(
    bus,
    states,
    decisions,
    utilities,
    costs,
    ev,
    increments,
    beta,
    maint_func,
    repl_func,
    loc_scale,
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
        unobs = (
            draw_unob(maint_func, loc_scale[0, 0], loc_scale[0, 1]),
            draw_unob(repl_func, loc_scale[1, 0], loc_scale[1, 1]),
        )

        value_replace = -costs[0, 0] - costs[0, 1] + unobs[1] + beta * ev[0]
        value_maintain = -costs[old_state, 0] + unobs[0] + beta * ev[old_state]
        if value_maintain > value_replace:
            decision = 0
            utility = -costs[old_state, 0] + unobs[0]
            new_state = old_state + increments[old_state, period]
        else:
            decision = 1
            utility = -costs[0, 0] - costs[0, 1] + unobs[1]
            new_state = increments[0, period]

        decisions[bus, period] = decision
        utilities[bus, period] = utility
        if period < num_periods - 1:
            states[bus, period + 1] = new_state
        if new_state > num_states - 10:
            raise ValueError("State space is too small.")
    return states, decisions, utilities


def get_unobs_data(shock):
    """
    :param shock            : A tuple of pandas.Series, where each Series name is
                             the scipy distribution function and the data is the loc
                             and scale specification.

    :return: A list of distribution names and an array of loc and scalle parameters.
    """
    # If no specification on the shocks is given. A right skewed gumbel distribution
    # with mean 0 and scale pi^2/6 is assumed for each shock component.
    shock = (
        (
            pd.Series(index=["loc"], data=[-np.euler_gamma], name="gumbel"),
            pd.Series(index=["loc"], data=[-np.euler_gamma], name="gumbel"),
        )
        if shock is None
        else shock
    )

    loc_scale = np.zeros((2, 2), dtype=float)
    for i, params in enumerate(shock):
        if "loc" in params.index:
            loc_scale[i, 0] = params["loc"]
        else:
            loc_scale[i, 0] = 0
        if "scale" in params.index:
            loc_scale[i, 1] = params["scale"]
        else:
            loc_scale[i, 1] = 1

    return shock[0].name, shock[1].name, loc_scale


@numba.jit(nopython=True)
def draw_unob(dist_name, loc, scale):
    if dist_name == "gumbel":
        return np.random.gumbel(loc, scale)
    elif dist_name == "normal":
        return np.random.normal(loc, scale)
    else:
        raise ValueError


def get_increments(trans_mat, num_periods):
    num_states = trans_mat.shape[0]
    increments = np.zeros(shape=(num_states, num_periods), dtype=int)
    for s in range(num_states):
        max_state = np.max(trans_mat[s, :].nonzero())
        p = trans_mat[s, s : (max_state + 1)]  # noqa: E203
        increments[s, :] = np.random.choice(len(p), size=num_periods, p=p)
    return increments
