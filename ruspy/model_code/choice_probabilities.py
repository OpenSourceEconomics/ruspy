import numpy as np


def choice_prob_gumbel(ev, obs_costs, disc_fac):
    """
    This function calculates the choice probabilities to maintain or replace the
    bus engine for each state.

    Parameters
    ----------
    ev : numpy.array
        see :ref:`ev`
    costs : numpy.array
        see :ref:`costs`
    disc_fac : numpy.float
        see :ref:`disc_fac`

    Returns
    -------
    pchoice : numpy.array
        see :ref:`pchoice`


    """
    s = ev.shape[0]
    util_main = disc_fac * ev - obs_costs[:, 0]  # Utility to maintain the bus
    util_repl = np.full(
        util_main.shape, disc_fac * ev[0] - obs_costs[0, 0] - obs_costs[0, 1]
    )
    util = np.vstack((util_main, util_repl)).T

    util = util - np.max(util)

    pchoice = np.exp(util) / (np.sum(np.exp(util), axis=1).reshape(s, -1))
    return pchoice
