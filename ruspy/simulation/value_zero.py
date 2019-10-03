"""
This module contains a function, which can be used to create plots with discounted
utility.
"""
import numba
import numpy as np


@numba.jit(nopython=True)
def discount_utility(utilities, beta):
    v = 0.0
    for i in np.arange(utilities.shpae[1]):
        v += (beta ** i) * np.sum(utilities[:, i])
    return v
