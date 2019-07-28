"""
This module contains the main function for the estimation process.
"""
import numpy as np
import scipy.optimize as opt
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix
from ruspy.estimation.estimation_cost_parameters import create_state_matrix
from ruspy.estimation.estimation_cost_parameters import loglike_opt_rule
from ruspy.estimation.estimation_cost_parameters import lin_cost


def estimate(init_dict, df, maint_func=lin_cost, repl_4=False):
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
    # Our state space is 20% larger, than the maximal observed state. We prevent
    # accumulation effects on the late states. As John Rust defines this variable we
    # provide a hardcoded option for the full replication of the paper with group 4.
    if repl_4:
        num_states = 90
    else:
        num_states = int(1.2 * np.max(states))
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
