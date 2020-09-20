"""
This module contains the functions necessary for the estimation process of transition
probabilities.
"""
import numba
import numpy as np
import pandas as pd
from estimagic.optimization.optimize import minimize


def estimate_transitions(df):
    """Estimating the transition proabilities.

    The sub function for managing the estimation of the transition probabilities.

    Parameters
    ----------
    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    result_transitions : dictionary
        see :ref:`result_trans`

    """
    result_transitions = {}
    usage = df["usage"].to_numpy(dtype=float)
    usage = usage[~np.isnan(usage)].astype(int)
    result_transitions["trans_count"] = transition_count = np.bincount(usage)

    # Prepare DataFrame for estimagic
    name = ["trans_prob"]
    number = np.arange(1, len(transition_count) + 1)
    index = pd.MultiIndex.from_product([name, number], names=["name", "number"])
    params = pd.DataFrame(
        transition_count / sum(transition_count), columns=["value"], index=index,
    )
    params.loc[params["value"] == 0] = 1e-20
    constr = [{"loc": "trans_prob", "type": "probability"}]

    raw_result_trans = minimize(
        criterion=loglike_trans,
        params=params,
        algorithm="scipy_L-BFGS-B",
        constraints=constr,
        criterion_kwargs={"transition_count": transition_count},
        logging=False,
    )

    result_transitions["x"] = raw_result_trans[1]["value"].to_numpy()
    result_transitions["fun"] = raw_result_trans[0]["fitness"]

    return result_transitions


def loglike_trans_individual(params, transition_count):
    """
    Individual negative Log-likelihood function of transition probability estimation.

    Parameters
    ----------
    p_raw : pandas.DataFrame
        The untransformed transition probability guess.
    transition_count : numpy.array
        The pooled count of state increases per period in the data.

    Returns
    -------
    log_like_individual : numpy.array
        The individual negative log-likelihood contributions of the transition probabilities

    """
    p_raw = params.loc["trans_prob", "value"].to_numpy()
    log_like_individual = -np.multiply(transition_count, np.log(p_raw))
    return log_like_individual


def loglike_trans(params, transition_count):
    """
    Sum the individual negative log-likelihood.

    Parameters
    ----------
    params : pd.DataFrame
        parameter guess of the transition probabilities.
    transition_count : numpy.array
        The pooled count of state increases per period in the data.

    Returns
    -------
    log_like : float
        the negative log likelihood given some transition probability guess.

    """
    log_like = loglike_trans_individual(params, transition_count).sum()
    return log_like


def loglike_trans_individual_derivative(params, transition_count):
    """
    generates the jacobian of the individual log likelihood function of the
    transition probabilities. This function is currently not used but is kept
    for further development of the package when estimagic can handle constrains
    with analytical derivatives.

    Parameters
    ----------
    params : pd.DataFrame
        parameter guess of the transition probabilities.
    transition_count : numpy.array
        The pooled count of state increases per period in the data.

    Returns
    -------
    jacobian : np.array
        a dim(params) x dim(params) matrix containing the Jacobian.

    """
    p_raw = params.loc["trans_prob", "value"].to_numpy()
    diagonal = -np.multiply(transition_count, 1 / p_raw)
    jacobian = diagonal * np.eye(len(p_raw))

    return jacobian


def loglike_trans_derivative(params, transition_count):
    gradient = loglike_trans_individual_derivative(params, transition_count).sum(axis=1)
    return gradient


@numba.jit(nopython=True)
def create_transition_matrix(num_states, trans_prob):
    """
    Creating the transition matrix with the assumption, that in every row the state
    increases have the same probability.

    Parameters
    ----------
    num_states : int
        The size of the state space.
    trans_prob : numpy.array
        The probabilities of an state increase.

    Returns
    -------
    trans_mat : numpy.array
        see :ref:`trans_mat`

    """
    trans_mat = np.zeros((num_states, num_states))
    for i in range(num_states):  # Loop over all states.
        for j, p in enumerate(trans_prob):  # Loop over the possible increases.
            if i + j < num_states - 1:
                trans_mat[i, i + j] = p
            elif i + j == num_states - 1:
                trans_mat[i, num_states - 1] = trans_prob[j:].sum()
            else:
                pass
    return trans_mat
