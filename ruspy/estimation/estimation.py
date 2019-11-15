"""
This module contains the main function for the estimation process.
"""
import numba
import numpy as np
import scipy.optimize as opt

from ruspy.estimation.estimation_cost_parameters import create_transition_matrix
from ruspy.estimation.estimation_cost_parameters import loglike_opt_rule
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.model_code.cost_functions import lin_cost


def estimate(init_dict, df, repl_4=False):
    """
    This function calls the auxiliary functions to estimate the decision parameters.
    Therefore it manages the estimation process. As mentioned in the model theory
    chapter of the paper, the estimation of the transition probabilities and the
    estimation of the parameters shaping the cost function
    are completely separate.

    :param init_dict: A dictionary containing the following variables as keys:

        :beta: (float)       : Discount factor.
        :states: (int)       : The size of the statespace.
        :maint_func: (func)  : The maintenance cost function. Default is the linear
                               from the paper.

    :param df:        A pandas dataframe, which contains for each observation the Bus
                      ID, the current state of the bus, the current period and the
                      decision made in this period.

    :param repl_4: Auxiliary variable indicating the complete setting of the
                   replication of the paper with group 4.

    :return: The function returns the optimization result of the transition
             probabilities and of the cost parameters as separate dictionaries.

    """
    beta = init_dict["beta"]
    transition_results = estimate_transitions(df, repl_4=repl_4)
    endog = df.loc[:, "decision"].to_numpy()
    states = df.loc[:, "state"].to_numpy()
    num_obs = df.shape[0]
    num_states = init_dict["states"]
    maint_func = lin_cost  # For now just set this to a linear cost function
    decision_mat = np.vstack(((1 - endog), endog))
    trans_mat = create_transition_matrix(num_states, np.array(transition_results["x"]))
    state_mat = create_state_matrix(states, num_states, num_obs)
    result = opt.minimize(
        loglike_opt_rule,
        args=(maint_func, num_states, trans_mat, state_mat, decision_mat, beta),
        x0=np.array([5, 5]),
        bounds=[(1e-6, None), (1e-6, None)],
        method="L-BFGS-B",
    )
    return transition_results, result


@numba.jit(nopython=True)
def create_state_matrix(states, num_states, num_obs):
    """
    This function constructs a auxiliary matrix for the likelihood.

    :param states:      A numpy array containing the observed states.
    :param num_states:  The size of the state space s.
    :type num_states:   int
    :param num_obs:     The total number of observations n.
    :type num_obs:      int

    :return:            A two dimensional numpy array containing n x s matrix
                        with TRUE in each row at the column in which the bus was in
                        that observation.
    """
    state_mat = np.full((num_states, num_obs), 0.0)
    for i, value in enumerate(states):
        state_mat[value, i] = 1.0
    return state_mat
