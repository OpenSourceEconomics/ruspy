"""
This module contains a function, which can be used to create plots with discounted
utility.
"""
import numba
import numpy as np


def discount_utility(df, gridsize, beta):
    """
    This function can be called to average the discounted utility of all buses in a
    sample. It is designed for a plot, where the averaged discounted utility in
    period 0 is plotted against the number of periods driven by the buses. Therefore
    the last point, at which the function is evaluated is always the averaged
    discounted utility of all periods in time 0.
    """
    num_buses = df["Bus_ID"].nunique()
    num_periods = int(df["period"].max()) + 1
    num_points = int(num_periods / gridsize) + 1
    utilities = df["utilities"].to_numpy(np.float64).reshape(num_buses, num_periods)
    return disc_ut_loop(gridsize, num_buses, num_points, utilities, beta)


@numba.jit(nopython=True)
def disc_ut_loop(gridsize, num_buses, num_points, utilities, beta):
    v_disc = np.zeros(num_points, dtype=numba.float64)
    for point in range(num_points):
        v = 0.0
        for i in range(point * gridsize):
            v += (beta ** i) * np.sum(utilities[:, i])
        v_disc[point] = v / num_buses
    return v_disc


def calc_ev_0(df, ev):
    num_buses = df["Bus_ID"].nunique()
    num_periods = int(df["period"].max()) + 1
    unobs = np.zeros((num_buses, num_periods, 2), dtype=np.float64)
    unobs[:, :, 0] = df["unobs_maint"].to_numpy().reshape(num_buses, num_periods)
    unobs[:, :, 1] = df["unobs_repl"].to_numpy().reshape(num_buses, num_periods)
    return calc_ev_0_loop(ev, unobs, num_buses)


@numba.jit(nopython=True)
def calc_ev_0_loop(ev, unobs, num_buses):
    v_calc = 0.
    for i in range(num_buses):
        v_calc += unobs[i, 0, 0] + ev[0]
    v_calc = v_calc / num_buses
    return v_calc
