"""
This module contains a function, which can be used to create plots with discounted
utility.
"""
import numba
import numpy as np


@numba.jit(nopython=True)
def discount_utility(utilities, num_periods, beta):
    v_tot = np.sum(
        np.multiply(beta ** np.arange(num_periods), np.sum(utilities, axis=0))
    )
    return v_tot / utilities.shape[0]
