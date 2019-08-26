"""
This module contains a function, which can be used to create plots with discounted
utility.
"""
import numba
import numpy as np


@numba.jit(nopython=True)
def discount_utility(gridsize, utilities, beta):
    """
    This function can be called to average the discounted utility of all buses in a
    sample. It is designed for a plot, where the averaged discounted utility in
    period 0 is plotted against the number of periods driven by the buses. Therefore
    the last point, at which the function is evaluated is always the averaged
    discounted utility of all periods in time 0.

    :param gridsize:    The gridsize for the periods to be evaluated.
    :type gridsize:     int
    :param utilities:   A two dimensional numpy array containing for each bus in
                        each period the utility as a float.A
    :param beta:        The discount factor.
    :type beta:         float

    :return: The function returns a list or numpy array with the averaged discounted
    utility for each point to be evaluated.
    """
    num_buses = utilities.shape[0]
    num_periods = utilities.shape[1]
    num_points = int(num_periods / gridsize) + 1
    v_disc = np.zeros(num_points, dtype=numba.float64)
    for point in range(num_points):
        v = 0.0
        for i in range(point * gridsize):
            v += (beta ** i) * np.sum(utilities[:, i])
        v_disc[point] = v / num_buses
    return v_disc


@numba.jit(nopython=True)
def calc_ev_0(ev, unobs, num_buses):
    v_calc = 0
    for i in range(num_buses):
        v_calc = v_calc + unobs[i, 0, 0] + ev[0]
    v_calc = v_calc / num_buses
    return v_calc
