import numba
import numpy as np


def discount_utility(df, disc_fac):
    v = 0.0
    for i in df.index.levels[0]:
        v += np.sum(np.multiply(disc_fac ** df.xs([i]).index, df.xs([i])["utilities"]))
    return v / len(df.index.levels[0])


@numba.jit(nopython=True)
def disc_ut_loop(utilities, disc_fac):
    num_buses, num_periods = utilities.shape
    v = 0.0
    for i in range(num_periods):
        v += (disc_fac**i) * np.sum(utilities[:, i])
    return v / num_buses
