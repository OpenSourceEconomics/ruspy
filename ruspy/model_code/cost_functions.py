import numpy as np


def calc_obs_costs(num_states, maint_func, params, scale=0.001):
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
    maint_cost = maint_func(num_states, params[1:], scale=scale)
    repl_cost = np.full(maint_cost.shape, rc + maint_cost[0])
    return np.vstack((maint_cost, repl_cost)).T


def lin_cost(num_states, params, scale):
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
    costs = states * params[0] * scale
    return costs


def lin_cost_dev(num_states, scale=0.001):
    dev = np.arange(num_states) * scale
    return dev


def cubic_costs(num_states, params, scale):
    states = np.arange(num_states)
    costs = (
        params[0] * scale * states
        + params[1] * scale * (states ** 2)
        + params[2] * scale * (states ** 3)
    )
    return costs


def cubic_costs_dev(num_states, scale=0.001):
    states = np.arange(num_states)
    dev = np.array([states * scale, scale * (states ** 2), scale * (states ** 3)]).T
    return dev


def quadratic_costs(num_states, params, scale=0.001):
    states = np.arange(num_states)
    costs = params[0] * scale * states + params[1] * scale * (states ** 2)
    return costs


def quadratic_costs_dev(num_states, scale=0.001):
    states = np.arange(num_states)
    dev = scale * states + scale * (states ** 2)
    return dev


def sqrt_costs(num_states, params, scale=0.001):
    states = np.arange(num_states)
    costs = params[0] * scale * np.sqrt(states)
    return costs


def sqrt_costs_dev(num_states, scale=0.001):
    states = np.arange(num_states)
    dev = scale * np.sqrt(states)
    return dev


def hyperbolic_costs(num_states, params, scale=0.001):
    states = np.arange(num_states)
    costs = params[0] * scale / ((num_states + 1) - states)
    return costs


def hyperbolic_costs_dev(num_states, scale=0.001):
    states = np.arange(num_states)
    dev = scale / ((num_states + 1) - states)
    return dev
