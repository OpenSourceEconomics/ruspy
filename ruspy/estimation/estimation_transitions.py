"""
This module contains the functions necessary for the estimation process of transition
probabilities.
"""
import numba
import numpy as np
import pandas as pd

from estimagic.optimization.optimize import minimize
from ruspy.estimation.bootstrapping import bootstrapp


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
    params = pd.DataFrame(np.full(len(transition_count), 1/len(transition_count)), 
                            columns=["value"],
                            index=index)
    constr = [{"loc": "trans_prob", "type": "probability"}]

    raw_result_trans = minimize(
        criterion=loglike_trans,
        params=params,
        algorithm='scipy_L-BFGS-B',
        constraints=constr,
        criterion_kwargs={"transition_count": transition_count},
        logging="logging_transition.db",
        )
    
    result_transitions["x"] = raw_result_trans[1]["value"].to_numpy()
    result_transitions["fun"] = raw_result_trans[0]["fitness"]

    # bootstrapping does not work right now as estimagic gives out correct x but 
    # but the reparam version of the Hessian
    # if isinstance(raw_result_trans[0]["hessian"], np.ndarray):
    #     result_transitions["95_conf_interv"], result_transitions["std_errors"] = bootstrapp(
    #     p_raw, raw_result_trans["hess_inv"], reparam=reparam_trans
    #     )

    return result_transitions


def loglike_trans(params, transition_count):
    """
    Log-likelihood function of transition probability estimation.

    Parameters
    ----------
    p_raw : numpy.array
        The raw values before reparametrization, on which there are no constraints
        or bounds.
    transition_count : numpy
        The pooled count of state increases per period in the data.

    Returns
    -------

    log_like : numpy.float
        The negative log-likelihood value of the transition probabilities
    """
    p_raw = params.loc["trans_prob", "value"].to_numpy()
    log_like = -np.sum(np.multiply(transition_count, np.log(p_raw)))
    return log_like


def reparam_trans(p_raw):
    """
    The reparametrization function for transition probabilities.

    Parameters
    ----------
    p_raw : numpy.array
        The raw values before reparametrization, on which there are no constraints
        or bounds.

    Returns
    -------
    trans_prob : numpy.array
        The probabilities of an state increase.

    """
    p = np.exp(p_raw) / np.sum(np.exp(p_raw))
    return p


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
