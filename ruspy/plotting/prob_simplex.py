import numpy as np


def get_x(p, l_simp):
    """
    This function calculates the x_coordinate of a 3 dimensional probability vector
    in a probability simplex with length l_simp.
    :param p: probability vector
    :param l_simp: length of simplex
    :return: x coordinate
    """
    return l_simp - p[0] * np.sqrt(1 + l_simp ** 2 / 4) - 0.5 * l_simp * p[1]


def create_set(
    p_ml,
    roh,
    l_simp,
    step,
    min_grid_p_0,
    min_grid_p_1,
    max_grid_p_0,
    max_grid_p_1,
    set_method="Kullback",
):
    """
    This function evaluates points in a given grid and returns their coordinate if
    they are in the given set.
    :param p_ml: Middle point of set
    :param roh: set size
    :param l_simp: lenght of simplex
    :param step: stepsize for grid
    :param min_grid_p_0: minimal value of p_0 in the grid
    :param min_grid_p_1: minimal value of p_1 in the grid
    :param max_grid_p_0: maximal value of p_0 in the grid
    :param max_grid_p_1: maximal value of p_1 in the grid
    :param set_method: Method to calculate distance between probability vectors
    :return:
    """
    if set_method == "Kullback":
        set_cond = Kullback_Leibler
    else:
        raise ValueError("Set measure not supported.")
    set_coordinates_x = []
    set_coordinates_y = []
    set_coordinates = np.empty(shape=(1, 2))
    set_probs = np.empty(shape=(1, len(p_ml)))
    for p_0 in np.arange(min_grid_p_0, max_grid_p_0, step):
        for p_1 in np.arange(min_grid_p_1, max_grid_p_1, step):
            p = np.array([p_0, p_1, 1 - p_0 - p_1])
            if p[(p > 0) & (p < 1)].size == p.size:
                if roh - set_cond(p, p_ml) >= 0:
                    x = get_x(p, l_simp)
                    set_coordinates_x += [x]
                    set_coordinates_y += [p_1]
                    set_coordinates = np.append(set_coordinates, [[x, p_1]], axis=0)
                    set_probs = np.append(set_probs, p.reshape(1, len(p_ml)), axis=0)
    return set_coordinates_x, set_coordinates_y, set_coordinates[1:], set_probs[1:]


def Kullback_Leibler(p, p_ml):
    """
    This function calculate the Kullback Leibler Divergence of p and p_ml.
    :param p:
    :param p_ml:
    :return:
    """
    val = 0
    for i, q in enumerate(p):
        val += q * np.log(q / p_ml[i])
    return val
