import numpy as np


def calc_obs_costs(num_states, maint_func, params, scale):
    """
    Calculating the observed costs of maintenance and replacement for each state.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    maint_func : callable
        see :ref:`maint_func`
    params : numpy.array
        see :ref:`params`
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    obs_costs : numpy.array
        see :ref:`costs`


    """
    rc = params[0]
    maint_cost = maint_func(num_states, params[1:], scale=scale)
    repl_cost = np.full(maint_cost.shape, rc + maint_cost[0])
    obs_costs = np.vstack((maint_cost, repl_cost)).T
    return obs_costs


def lin_cost(num_states, params, scale):
    """
    Calculating for each state the observed costs of maintenance in the case of a
    linear cost function.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    params : numpy.array
        see :ref:`params`
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    costs : numpy.array
        A num_states sized one dimensional numpy array containing the maintenance
        costs for each state.

    """
    states = np.arange(num_states)
    costs = states * params[0] * scale
    return costs


def lin_cost_dev(num_states, scale):
    """
    Calculating for each state the derivative of the linear maintenance cost function.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    dev : numpy.array
        A num_states sized one dimensional numpy array containing the derivative of the
        linear maintenance cost function for each state.

    """
    dev = np.arange(num_states) * scale
    return dev


def cubic_costs(num_states, params, scale):
    """
    Calculating for each state the observed costs of maintenance in the case of a
    cubic cost function.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    params : numpy.array
        see :ref:`params`
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    costs : numpy.array
        A num_states sized one dimensional numpy array containing the maintenance
        costs for each state.

    """
    states = np.arange(num_states)
    costs = (
        params[0] * scale * states
        + params[1] * scale * (states**2)
        + params[2] * scale * (states**3)
    )
    return costs


def cubic_costs_dev(num_states, scale):
    """
    Calculating for each state the derivative of the cubic maintenance cost function.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    dev : numpy.array
        A num_states x 3 dimensional numpy array containing the derivative of
        the cubic maintenance cost function for each state.

    """
    states = np.arange(num_states)
    dev = np.array([states * scale, scale * (states**2), scale * (states**3)]).T
    return dev


def quadratic_costs(num_states, params, scale):
    """
    Calculating for each state the observed costs of maintenance in the case of a
    quadratic cost function.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    params : numpy.array
        see :ref:`params`
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    costs : numpy.array
        A num_states sized one dimensional numpy array containing the maintenance
        costs for each state.

    """
    states = np.arange(num_states)
    costs = params[0] * scale * states + params[1] * scale * (states**2)
    return costs


def quadratic_costs_dev(num_states, scale):
    """
    Calculating for each state the derivative of the quadratic maintenance cost
    function.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    dev : numpy.array
        A num_states x 2 dimensional numpy array containing the derivative of
        the quadratic maintenance cost function for each state.

    """
    states = np.arange(num_states)
    dev = np.array([scale * states, scale * (states**2)]).T
    return dev


def sqrt_costs(num_states, params, scale):
    """
    Calculating for each state the observed costs of maintenance in the case of a
    square root cost function.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    params : numpy.array
        see :ref:`params`
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    costs : numpy.array
        A num_states sized one dimensional numpy array containing the maintenance
        costs for each state.

    """
    states = np.arange(num_states)
    costs = params[0] * scale * np.sqrt(states)
    return costs


def sqrt_costs_dev(num_states, scale):
    """
    Calculating for each state the derivative of the square root maintenance cost
    function.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    dev : numpy.array
        A num_states sized one dimensional numpy array containing the derivative of the
        square root maintenance cost function for each state.

    """
    states = np.arange(num_states)
    dev = scale * np.sqrt(states)
    return dev


def hyperbolic_costs(num_states, params, scale):
    """
    Calculating for each state the observed costs of maintenance in the case of a
    hyperbolic cost function.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    params : numpy.array
        see :ref:`params`
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    costs : numpy.array
        A num_states sized one dimensional numpy array containing the maintenance
        costs for each state.

    """
    states = np.arange(num_states)
    costs = params[0] * scale / ((num_states + 1) - states)
    return costs


def hyperbolic_costs_dev(num_states, scale):
    """
    Calculating for each state the derivative of the hyperbolic maintenance cost
    function.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    scale : numpy.float
        see :ref:`scale`

    Returns
    -------
    dev : numpy.array
        A num_states sized one dimensional numpy array containing the derivative of the
        hyperbolic maintenance cost function for each state.

    """
    states = np.arange(num_states)
    dev = scale / ((num_states + 1) - states)
    return dev
