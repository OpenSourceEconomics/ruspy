"""
This module contains the primary function for estimating the parameters describing the decision rule of a single
agent in a dynamic discrete choice model. All functions needed for the estimation are contained in estimation_auxiliary.
"""

import numpy as np
import scipy.optimize as opt
from ruspy.estimation.estimation_auxiliary import estimate_transitions_5000
from ruspy.estimation.estimation_auxiliary import create_transition_matrix
from ruspy.estimation.estimation_auxiliary import create_state_matrix
from ruspy.estimation.estimation_auxiliary import loglike_opt_rule
from ruspy.estimation.estimation_auxiliary import lin_cost


def estimate(init_dict, df):
    """ This function calls the auxiliary functions to estimate the decision parameters.
    Therefore it manages the estimation process. As mentioned in the model theory chapter of the paper,
    the estimation of the transition probabilities and the estimation of the parameters shaping the cost function
    are completely separate.

    Args:
        init_dict (dictionary): A dictionary containing the discount factor, the size of state space and the type of
        the cost function.
        df (PandasDataFrame) : A pandas dataframe, which contains for each observation the Bus ID, the current state of
        the bus, the current period and the decision made in this period.

    Returns: The function returns the optimization result of the transition probabilities and of the cost parameters
    separate.

    """

    beta = init_dict['beta']
    transition_results = estimate_transitions_5000(df)
    endog = df.loc[:, 'decision']
    exog = df.loc[:, 'state']
    num_obs = df.shape[0]
    num_states = init_dict['states']
    if init_dict['maint_func'] == 'linear':
        maint_func = lin_cost
    else:
        maint_func = lin_cost
    decision_mat = np.vstack(((1 - endog), endog))
    trans_mat = create_transition_matrix(num_states, np.array(transition_results['x']))
    state_mat = create_state_matrix(exog, num_states, num_obs)
    result = opt.minimize(loglike_opt_rule, args=(maint_func, num_states, trans_mat, state_mat, decision_mat, beta),
                          x0=np.array([10, 2]), bounds=[(1e-6, None), (1e-6, None)], method='L-BFGS-B')
    return transition_results, result
