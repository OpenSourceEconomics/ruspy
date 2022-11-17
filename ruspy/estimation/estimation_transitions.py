"""
This module contains the functions necessary for the estimation process of transition
probabilities.
"""
import numba
import numpy as np


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
    result_transitions["x"] = tans_probs = transition_count / np.sum(transition_count)
    result_transitions["fun"] = loglike_trans(tans_probs, transition_count)

    return result_transitions


def loglike_trans_individual(trans_dist, transition_count):
    """
    Individual negative Log-likelihood function of transition probability estimation.

    Parameters
    ----------
    trans_dist : numpy.ndarray
        The transition probabilities.
    transition_count : numpy.ndarray
        The pooled count of state increases per period in the data.

    Returns
    -------
    log_like_individual : numpy.ndarray
        The individual negative log-likelihood contributions of the transition probabilities

    """
    log_like_individual = -np.multiply(transition_count, np.log(trans_dist))
    return log_like_individual


def loglike_trans(trans_dist, transition_count):
    """
    Sum the individual negative log-likelihood.

    Parameters
    ----------
    trans_dist : np.ndarray
        parameters of the transition probabilities.
    transition_count : numpy.ndarray
        The pooled count of state increases per period in the data.

    Returns
    -------
    log_like : float
        the negative log likelihood given some transition probability guess.

    """
    log_like = loglike_trans_individual(trans_dist, transition_count).sum()
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
    transition_count : numpy.ndarray
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
    trans_prob : numpy.ndarray
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
