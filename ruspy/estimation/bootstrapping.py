import numpy as np


def calc_95_conf(params_raw, hesse_inv_raw, reparam=None, runs=1000):
    """
    Bootstrapping the 95% interval.

    Parameters
    ----------
    params_raw : numpy.array
        The raw values before reparametrization as output from the scipy minimizer.

    hesse_inv_raw : numpy.array
        The inverse of hessian matrix, provided by the scipy minimizer. For more
        information see : https://docs.scipy.org/ .

    reparam : function
        A reparametrization function. If None is given, the dummy function no_reparam,
        will be selected.

    runs : int
        The number of runs, allows to specify the precision of the bootstrapping.

    Returns
    -------
    conf_bounds : numpy.array
        Returns the 95% confidence bounds.

    """

    reparam = no_reparam if reparam is None else reparam
    params = reparam(params_raw)

    draws = draw_from_raw(reparam, params_raw, hesse_inv_raw, runs)
    conf_bounds = np.zeros((2, len(params_raw)), dtype=float)
    for i in range(len(params_raw)):
        conf_bounds[0, i] = params[i] - np.percentile(draws[:, i], 2.5)
        conf_bounds[1, i] = np.percentile(draws[:, i], 97.5) - params[i]
    return conf_bounds


def draw_from_raw(reparam, params_raw, hesse_inv_raw, runs):
    draws = np.zeros((runs, len(params_raw)), dtype=np.float)
    for i in range(runs):
        draw = np.random.multivariate_normal(params_raw, hesse_inv_raw)
        draws[i, :] = reparam(draw)
    return draws


def no_reparam(params_raw):
    return params_raw
