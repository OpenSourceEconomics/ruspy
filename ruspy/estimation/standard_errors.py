import numpy as np


def calc_asymp_stds(params_raw, hesse_inv_raw, reparam=None, runs=1000):

    reparam = no_reparam if reparam is None else reparam
    params = reparam(params_raw)

    draws = draw_from_raw(reparam, params_raw, hesse_inv_raw, runs)
    std_err = np.zeros((2, len(params_raw)), dtype=float)
    for i in range(len(params_raw)):
        std_err[0, i] = params[i] - np.percentile(draws[:, i], 2.5)
        std_err[1, i] = np.percentile(draws[:, i], 97.5) - params[i]
    return std_err


def draw_from_raw(reparam, params_raw, hesse_inv_raw, runs):
    draws = np.zeros((runs, len(params_raw)), dtype=np.float)
    for i in range(runs):
        draw = np.random.multivariate_normal(params_raw, hesse_inv_raw)
        draws[i, :] = reparam(draw)
    return draws


def no_reparam(params_raw):
    return params_raw
