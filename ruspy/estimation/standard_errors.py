import numpy as np
import pandas as pd
from estimagic.differentiation.differentiation import hessian

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import loglike_cost_params
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions


def cov_multinomial(n, p):
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


def params_hess(params, df, beta, maint_func, repl_4=False):
    """Calculates the hessian of the cost parameters."""
    transition_results = estimate_transitions(df, repl_4=repl_4)
    ll_trans = transition_results["fun"]
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
    params_df = pd.DataFrame(index=["RC", "theta_1_1"], columns=["value"], data=params)
    wrap_func = create_wrap_func(
        maint_func, num_states, trans_mat, state_mat, decision_mat, beta, ll_trans
    )
    return hessian(wrap_func, params_df)


def create_wrap_func(
    maint_func, num_states, trans_mat, state_mat, decision_mat, beta, ll_trans
):
    def wrap_func(x):
        x_np = x["value"].to_numpy()
        return ll_trans + loglike_cost_params(
            x_np, maint_func, num_states, trans_mat, state_mat, decision_mat, beta
        )

    return wrap_func
