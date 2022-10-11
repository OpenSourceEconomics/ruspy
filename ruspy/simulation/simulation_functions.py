import numba
import numpy as np

from ruspy.simulation.simulation_model import decide
from ruspy.simulation.simulation_model import draw_increment


@numba.jit(nopython=True)
def simulate_strategy(
    num_periods,
    num_buses,
    costs,
    ev,
    trans_mat,
    disc_fac,
    seed,
):
    """
    Simulating the decision process.

    This function simulates the decision strategy, as long as the current period is
    below the number of periods and the current highest state of a bus is in the
    first half of the state space.

    Parameters
    ----------
    num_periods : int
         The number of periods to be simulated.
    num_buses : int
        The number of buses to be simulated.
    costs : numpy.array
        see :ref:`costs`
    ev : numpy.array
        see :ref:`ev`
    trans_mat : numpy.array
        see :ref:`trans_mat`
    disc_fac : float
        see :ref:`disc_fac`
    seed : int
        A positive integer setting the random seed for drawing random numbers.

    Returns
    -------
    states : numpy.array
        A two dimensional numpy array containing for each bus in each period the
        state as an integer.

    decisions : numpy.array
        A two dimensional numpy array containing for each bus in each period the
        decision as an integer.

    utilities : numpy.array
        A two dimensional numpy array containing for each bus in each period the
        utility as a float.

    usage : numpy.array
        A two dimensional numpy array containing for each bus in each period the
        mileage usage of last period as integer.
    """
    np.random.seed(seed)
    num_states = ev.shape[0]
    states = np.zeros((num_buses, num_periods), dtype=numba.u2)
    decisions = np.zeros((num_buses, num_periods), dtype=numba.b1)
    utilities = np.zeros((num_buses, num_periods), dtype=numba.float32)
    usage = np.zeros((num_buses, num_periods), dtype=numba.u1)
    absorbing_state = 0
    for bus in range(num_buses):
        for period in range(num_periods):
            old_state = states[bus, period]

            intermediate_state, decision, utility = decide(
                old_state,
                costs,
                disc_fac,
                ev,
            )

            state_increase = draw_increment(intermediate_state, trans_mat)
            decisions[bus, period] = decision
            utilities[bus, period] = utility
            new_state = intermediate_state + state_increase
            if new_state > num_states:
                new_state = num_states
                state_increase = num_states - intermediate_state
            usage[bus, period] = state_increase
            if period < num_periods - 1:
                states[bus, period + 1] = new_state
            if new_state == num_states:
                absorbing_state = 1

    return states, decisions, utilities, usage, absorbing_state


@numba.jit(nopython=True)
def simulate_strategy_reduced_data_utilities(
    num_periods,
    num_buses,
    costs,
    ev,
    trans_mat,
    disc_fac,
    seed,
):
    """
    Simulating the decision process with reduced data usage.

    This function simulates the decision strategy, as long as the current period is
    below the number of periods and the current highest state of a bus is in the
    first half of the state space.

    Parameters
    ----------
    num_periods : int
         The number of periods to be simulated.
    num_buses : int
        The number of buses to be simulated.
    costs : numpy.array
        see :ref:`costs`
    ev : numpy.array
        see :ref:`ev`
    trans_mat : numpy.array
        see :ref:`trans_mat`
    disc_fac : float
        see :ref:`disc_fac`
    seed : int
        A positive integer setting the random seed for drawing random numbers.

    Returns
    -------
    utilities : numpy.array
        A two dimensional numpy array containing for each bus in each period the
        utility as a float.
    """
    np.random.seed(seed)
    num_states = ev.shape[0]
    utilities = np.zeros((num_buses, num_periods), dtype=numba.float32)
    absorbing_state = 0
    for bus in range(num_buses):
        new_state = 0
        for period in range(num_periods):
            old_state = new_state

            intermediate_state, decision, utility = decide(
                old_state,
                costs,
                disc_fac,
                ev,
            )

            state_increase = draw_increment(intermediate_state, trans_mat)
            utilities[bus, period] = utility

            new_state = intermediate_state + state_increase
            if new_state > num_states:
                new_state = num_states
            if new_state == num_states:
                absorbing_state = 1

    return utilities, absorbing_state


@numba.jit(nopython=True)
def simulate_strategy_reduced_data_disc_utility(
    num_periods,
    num_buses,
    costs,
    ev,
    trans_mat,
    disc_fac,
    seed,
):
    """
    Simulating the decision process with reduced data usage.

    This function simulates the decision strategy, as long as the current period is
    below the number of periods and the current highest state of a bus is in the
    first half of the state space.

    Parameters
    ----------
    num_periods : int
         The number of periods to be simulated.
    num_buses : int
        The number of buses to be simulated.
    costs : numpy.array
        see :ref:`costs`
    ev : numpy.array
        see :ref:`ev`
    trans_mat : numpy.array
        see :ref:`trans_mat`
    disc_fac : float
        see :ref:`disc_fac`
    seed : int
        A positive integer setting the random seed for drawing random numbers.

    Returns
    -------
    utilities : numpy.array
        A two dimensional numpy array containing for each bus in each period the
        utility as a float.
    """
    np.random.seed(seed)
    num_states = ev.shape[0]
    disc_utility = 0.0
    absorbing_state = 0
    for _ in range(num_buses):
        new_state = 0
        for period in range(num_periods):
            old_state = new_state

            intermediate_state, decision, utility = decide(
                old_state,
                costs,
                disc_fac,
                ev,
            )

            state_increase = draw_increment(intermediate_state, trans_mat)
            disc_utility += disc_fac**period * utility

            new_state = intermediate_state + state_increase
            if new_state > num_states:
                new_state = num_states
            if new_state == num_states:
                absorbing_state = 1
    disc_utility /= num_buses
    return disc_utility, absorbing_state


# This was an old attempt to implement more shocks than the standard gumbel. Would do
# this much different now!!!! Just keep it for further work!
# def get_unobs_data(shock):
#     # If no specification on the shocks is given. A right skewed gumbel distribution
#     # with mean 0 and scale pi^2/6 is assumed for each shock component.
#     shock = (
#         (
#             pd.Series(index=["loc"], data=[], name="gumbel"),
#             pd.Series(index=["loc"], data=[-np.euler_gamma], name="gumbel"),
#         )
#         if shock is None
#         else shock
#     )
#
#     loc_scale = np.zeros((2, 2), dtype=float)
#     for i, params in enumerate(shock):
#         if "loc" in params.index:
#             loc_scale[i, 0] = params["loc"]
#         else:
#             loc_scale[i, 0] = 0
#         if "scale" in params.index:
#             loc_scale[i, 1] = params["scale"]
#         else:
#             loc_scale[i, 1] = 1
#
#     return shock[0].name, shock[1].name, loc_scale
#
#
# @numba.jit(nopython=True)
# def draw_unob(dist_name, loc, scale):
#     if dist_name == "gumbel":
#         return np.random.gumbel(loc, scale)
#     elif dist_name == "normal":
#         return np.random.normal(loc, scale)
#     else:
#         raise ValueError
