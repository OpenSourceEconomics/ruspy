import numpy as np
import scipy.optimize as opt
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix
from ruspy.estimation.estimation_cost_parameters import calc_fixp
from ruspy.estimation.estimation_cost_parameters import myopic_costs
from ruspy.estimation.estimation_cost_parameters import lin_cost


def get_worst_trans(init_dict, roh, num_states, max_it=1000, min_state=0):
    beta = init_dict["beta"]
    x_0 = np.array(init_dict["known probs"])
    dim = x_0.shape[0]
    params = np.array(init_dict["params"])
    costs = myopic_costs(num_states, lin_cost, params)
    eq_constr = {"type": "eq", "fun": lambda x: 1 - np.sum(x)}
    ineq_constr = {
        "type": "ineq",
        "fun": lambda x: roh - np.sum(np.multiply(x, np.log(np.divide(x, x_0)))),
    }
    res = opt.minimize(
        select_fixp,
        args=(min_state, num_states, costs, beta, max_it),
        x0=x_0,
        bounds=[(1e-6, 1)] * dim,
        method="SLSQP",
        constraints=[eq_constr, ineq_constr],
    )
    worst_trans = np.array(res["x"])
    return worst_trans


def select_fixp(trans_probs, state, num_states, costs, beta, max_it):
    trans_mat = create_transition_matrix(num_states, trans_probs)
    fixp = calc_fixp(num_states, trans_mat, costs, beta, max_it=max_it)
    return fixp[state]
