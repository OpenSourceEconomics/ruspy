import numpy as np
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.estimation.estimation_cost_parameters import (
    create_transition_matrix,
    create_state_matrix,
    loglike_opt_rule,
)
from estimagic.differentiation.differentiation import hessian


def calc_cov_multinomial(n, p):
    """Calculates the covariance matrix of a multinominal distribution. We use this
    function to calculate the standard errors of the transition probabilities"""
    dim = len(p)
    cov = np.zeros(shape=(dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                cov[i, i] = p[i] * (1 - p[i])
            else:
                cov[i, j] = -p[i] * p[j]
    return cov / n


def params_hess(params, df, maint_func, beta, repl_4=False):
    """Calculates the hessian of the cost parameters."""
    transition_results = estimate_transitions(df, repl_4=repl_4)
    states = df.loc[:, "state"].to_numpy()
    num_obs = df.shape[0]
    if repl_4:
        num_states = 90
    else:
        num_states = int(1.2 * np.max(states))
    trans_mat = create_transition_matrix(num_states, np.array(transition_results["x"]))
    state_mat = create_state_matrix(states, num_states, num_obs)
    endog = df.loc[:, "decision"].to_numpy()
    decision_mat = np.vstack(((1 - endog), endog))
    return hessian(
        loglike_opt_rule,
        params,
        func_args=(maint_func, num_states, trans_mat, state_mat, decision_mat, beta),
    )
