import numpy as np


def choice_prob_gumbel(ev, costs, beta):
    """
    This function calculates the choice probabilities to maintain or replace the
    bus engine for each state.

    :param ev:      A numpy array containing for each state the expected value
                    fixed point.
    :param costs:  A numpy array containing the parameters shaping the cost
                    function.

    :param beta:    The discount factor.
    :type beta:     float

    :return: A two dimensional numpy array containing in each row the choice
             probability for maintenance in the first and for replacement in second
             column.
    """
    s = ev.shape[0]
    util_main = beta * ev - costs[:, 0]  # Utility to maintain the bus
    util_repl = np.full(util_main.shape, beta * ev[0] - costs[0, 0] - costs[0, 1])
    util = np.vstack((util_main, util_repl)).T
    util = util - np.amin(util)
    pchoice = np.exp(util) / (np.sum(np.exp(util), axis=1).reshape(s, -1))
    return pchoice
