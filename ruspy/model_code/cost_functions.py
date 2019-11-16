import numpy as np


def calc_obs_costs(num_states, maint_func, params):
    """
    This function calculates a vector containing the costs for maintenance and
    replacement, without recognizing the future.

    :param num_states:  The size of the state space.
    :type num_states:   int
    :param maint_func:  The maintenance cost function. Only linear implemented so
                        far.
    :param params:      A numpy array containing the parameters shaping the cost
                        function.

    :return: A two dimensional numpy array containing in each row the cost for
             maintenance in the first and for replacement in the second column.
    """
    rc = params[0]
    maint_cost = maint_func(num_states, params[1:])
    repl_cost = np.full(maint_cost.shape, rc + maint_cost[0])
    return np.vstack((maint_cost, repl_cost)).T


def lin_cost(num_states, params, scale=0.001):
    """
    This function describes a linear cost function, which Rust concludes is the most
    realistic maintenance function.

    :param num_states:  The size of the state space.
    :type num_states:   int
    :param params:      A numpy array containing the parameters shaping the cost
                        function.
    :param scale:       A factor for scaling the maintenance costs.

    :return: A numpy array containing the maintenance cost for each state.
    """
    states = np.arange(num_states)
    return states * scale * params[0]


def lin_cost_dev(num_states, scale=0.001):
    return np.arange(num_states) * scale
