import numba
import numpy as np


@numba.jit(nopython=True)
def decide(
    old_state,
    costs,
    disc_fac,
    ev,
):
    """
    Choosing action in current state.

    Parameters
    ----------
    old_state: int
        Current state.
    costs : numpy.array
        see :ref:`costs`
    disc_fac : float
        see :ref:`disc_fac`
    ev : numpy.array
        see :ref:`ev`

    Returns
    -------
    intermediate_state : int
        State before transition.
    decision : int
        Decision of this period.
    utility : float
        Utility of this period.
    """
    unobs = np.random.gumbel(-np.euler_gamma, 1, size=2)

    value_replace = -costs[0, 0] - costs[0, 1] + unobs[1] + disc_fac * ev[0]
    value_maintain = -costs[old_state, 0] + unobs[0] + disc_fac * ev[old_state]
    if value_maintain > value_replace:
        decision = False
        utility = -costs[old_state, 0] + unobs[0]
        intermediate_state = old_state
    else:
        decision = True
        utility = -costs[0, 0] - costs[0, 1] + unobs[1]
        intermediate_state = 0
    return intermediate_state, decision, utility


@numba.jit(nopython=True)
def draw_increment(state, trans_mat):
    """
    Drawing a random increase.

    Parameters
    ----------
    state : int
        Current state.
    trans_mat : numpy.array
        see :ref:`trans_mat`

    Returns
    -------
    increase : int
        Number of state increase.
    """
    max_state = np.max(np.nonzero(trans_mat[state, :])[0])
    p = trans_mat[state, state : (max_state + 1)]  # noqa: E203
    increase = np.argmax(np.random.multinomial(1, p))
    return increase