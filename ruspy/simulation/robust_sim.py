import numpy as np
import scipy.optimize as opt
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix
from ruspy.estimation.estimation_cost_parameters import calc_fixp
from ruspy.estimation.estimation_cost_parameters import converge_choice
from ruspy.estimation.estimation_cost_parameters import cost_func
from ruspy.estimation.estimation_cost_parameters import lin_cost


def get_worst_trans(
    init_dict, roh, num_states, choice=True, max_it=10000000, min_state=0
):
    beta = init_dict["beta"]
    x_0 = np.array(init_dict["known trans"])
    dim = x_0.shape[0]
    params = np.array(init_dict["params"])
    costs = cost_func(num_states, lin_cost, params)
    eq_constr = {"type": "eq", "fun": lambda x: 1 - np.sum(x)}
    ineq_constr = {
        "type": "ineq",
        "fun": lambda x: roh - np.sum(np.multiply(x, np.log(np.divide(x, x_0)))),
    }
    if choice:
        res = opt.minimize(
            select_choice,
            args=(num_states, costs, beta, max_it),
            x0=x_0,
            bounds=[(1e-6, 1)] * dim,
            method="SLSQP",
            constraints=[eq_constr, ineq_constr],
        )
    else:
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


def select_choice(trans_probs, num_states, costs, beta, max_it):
    trans_mat = create_transition_matrix(num_states, trans_probs)
    choice = converge_choice(num_states, trans_mat, costs, beta, max_it=max_it)
    return -choice[int(num_states / 2), 0]


def draw_trans_probs_mulitvar(n, p, size):
    mean = p * n
    cov = calc_cov_multinomial(n, p) * (n ** 2)
    draw_array = np.random.multivariate_normal(mean, cov, size=size)
    probs = []
    for draw in draw_array:
        probs += [draw / sum(draw)]
    return probs


def calc_cov_multinomial(n, p):
    dim = len(p)
    cov = np.zeros(shape=(dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            if i == j:
                cov[i, i] = p[i] * (1 - p[i])
            else:
                cov[i, j] = -p[i] * p[j]
    return cov / n
