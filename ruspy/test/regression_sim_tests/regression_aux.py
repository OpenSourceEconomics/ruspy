import numba
import numpy as np


def discount_utility(df, disc_fac):
    v = 0.0
    for i in df.index.levels[0]:
        v += np.sum(np.multiply(disc_fac ** df.xs([i]).index, df.xs([i])["utilities"]))
    return v / len(df.index.levels[0])


@numba.jit(nopython=True)
def disc_ut_loop(gridsize, num_buses, num_points, utilities, disc_fac):
    v_disc = np.zeros(num_points, dtype=numba.float64)
    for point in range(num_points):
        v = 0.0
        for i in range(point * gridsize):
            v += (disc_fac ** i) * np.sum(utilities[:, i])
        v_disc[point] = v / num_buses
    return v_disc
