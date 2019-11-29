"""
This module contains the main function for the estimation process.
"""
import numpy as np
import scipy.optimize as opt

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import loglike_cost_params
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.model_code.cost_functions import lin_cost


def estimate(init_dict, df):
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

    :return: The function returns the optimization result of the transition
             probabilities and of the cost parameters as separate dictionaries.

    """
    beta = init_dict["beta"]
    transition_results = estimate_transitions(df)
    endog = df.loc[:, "decision"].to_numpy()
    states = df.loc[:, "state"].to_numpy()
    num_states = init_dict["states"]
    maint_func = lin_cost  # For now just set this to a linear cost function
    decision_mat = np.vstack(((1 - endog), endog))
    trans_mat = create_transition_matrix(num_states, np.array(transition_results["x"]))
    state_mat = create_state_matrix(states, num_states)
    result = opt.minimize(
        loglike_cost_params,
        args=(maint_func, num_states, trans_mat, state_mat, decision_mat, beta),
        x0=np.array([5, 5]),
        # jac=derivative_loglike_cost_params,
        method="BFGS",
    )
    return transition_results, result
