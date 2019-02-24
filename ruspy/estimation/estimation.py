""""
This function replicates Rust's 1987 paper on optimal bus engine replacement for the GMC 1975 buses
in the sample.
:return:
result.json         : A json file with the results stored.
transition_results : A list with the estimated transition probabilities.
result              : A dictionary with the estimation results on the cost parameters.
"""

import numpy as np
import scipy.optimize as opt
import json

from ruspy.estimation.estimation_auxiliary import estimate_transitions_5000
from ruspy.estimation.estimation_auxiliary import create_transition_matrix
from ruspy.estimation.estimation_auxiliary import create_state_matrix
from ruspy.estimation.estimation_auxiliary import loglike_opt_rule
from ruspy.estimation.estimation_auxiliary import lin_cost


def estimate(init_dict, df):
    beta = init_dict['beta']
    transition_results = estimate_transitions_5000(df)
    endog = df.loc[:, 'decision']
    exog = df.loc[:, 'state']
    num_obs = df.shape[0]
    num_states = init_dict['states']
    decision_mat = np.vstack(((1 - endog), endog))
    trans_mat = create_transition_matrix(num_states, np.array(transition_results['x']))
    state_mat = create_state_matrix(exog, num_states, num_obs)
    result = opt.minimize(loglike_opt_rule, args=(lin_cost, num_states, trans_mat, state_mat, decision_mat, beta),
                          x0=np.array([10, 2]), bounds=[(1e-6, None), (1e-6, None)], method='L-BFGS-B')
    print(transition_results, result)
    return transition_results, result
