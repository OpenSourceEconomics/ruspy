import pandas as pd
import numpy as np
import numba


@numba.jit(nopython=True)
def simulate_strategy(
    num_periods,
    costs,
    ev,
    trans_mat,
    beta,
    maint_shock_dist_name,
    repl_shock_dist_name,
    loc_scale,
    seed,
):
    """
    This function simulates the decision strategy, as long as the current period is
    below the number of periods and the current highest state of a bus is in the
    first half of the state space.

    :param costs:        A two dimensional float numpy array containing for each
                         state the cost to maintain in the first and to replace the bus
                         engine in the second column.
    :param ev:           A numpy array containing for each state the expected value
                         fixed point.total
    :param num_periods:  The number of periods to be simulated.
    :type num_periods:   int
    :param beta:         The discount factor.
    :type beta:          float

    :return: The function returns the following objects:

        :states:           : A two dimensional numpy array containing for each bus in
                             each period the state as an integer.
        :decisions:        : A two dimensional numpy array containing for each bus in
                             each period the decision as an integer.
        :utilities:        : A two dimensional numpy array containing for each bus in
                             each period the utility as a float.
        :num_states: (int) : The size of the state space.

    """
    np.random.seed(seed)
    num_states = ev.shape[0]
    states = np.zeros(num_periods, dtype=numba.int32)
    decisions = np.zeros(num_periods, dtype=numba.int32)
    utilities = np.zeros(num_periods, dtype=numba.float32)
    for period in range(num_periods):
        old_state = states[period]
        unobs = (
            draw_unob(maint_shock_dist_name, loc_scale[0, 0], loc_scale[0, 1]),
            draw_unob(repl_shock_dist_name, loc_scale[1, 0], loc_scale[1, 1]),
        )

        value_replace = -costs[0, 0] - costs[0, 1] + unobs[1] + beta * ev[0]
        value_maintain = -costs[old_state, 0] + unobs[0] + beta * ev[old_state]
        if value_maintain > value_replace:
            decision = 0
            utility = -costs[old_state, 0] + unobs[0]
            intermediate_state = old_state
        else:
            decision = 1
            utility = -costs[0, 0] - costs[0, 1] + unobs[1]
            intermediate_state = 0

        decisions[period] = decision
        utilities[period] = utility
        new_state = intermediate_state + draw_increment(intermediate_state, trans_mat)
        if period < num_periods - 1:
            states[period + 1] = new_state
        if new_state > num_states - 10:
            raise ValueError("State space is too small.")
    return states, decisions, utilities


@numba.jit(nopython=True)
def draw_increment(state, trans_mat):
    max_state = np.max(np.nonzero(trans_mat[state, :])[0])
    p = trans_mat[state, state : (max_state + 1)]  # noqa: E203
    return np.argmax(np.random.multinomial(1, p))


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
