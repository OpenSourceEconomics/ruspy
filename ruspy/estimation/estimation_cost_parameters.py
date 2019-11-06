"""
This module contains functions for estimating the parameters shaping the cost
function. Therefore it also contains the heart of this project. The python
implementation of fix point algorithm developed by John Rust.
"""
import numpy as np

# import numba


# @numba.jit(nopython=True)
def create_transition_matrix(num_states, trans_prob):
    """
    This function creates a markov transition matrix. By the assumptions of the
    underlying model, only the diagonal and elements to the right can have a non-zero
    entry.

    :param num_states:  The size of the state space s.
    :type num_states:   int
    :param trans_prob:  A numpy array containing the transition probabilities for a
                        state increase.

    :return: A two dimensional numpy array containing a s x s markov transition matrix.
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


# @numba.jit(nopython=True)
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


# @numba.jit(nopython=True)
def loglike_opt_rule(
    params,
    maint_func,
    num_states,
    trans_mat,
    state_mat,
    decision_mat,
    beta,
    max_it=10000000,
):
    """
    This is the logliklihood function for the estimation of the cost parameters.

    :param params:       A numpy array containing the parameters shaping the cost
                         function.
    :param maint_func:   The maintenance cost function. Only linear implemented so
                         far.
    :param num_states:   The size of the state space s.
    :type num_states:    int
    :param trans_mat:    A two dimensional numpy array containing a s x s markov
                         transition matrix.
    :param state_mat:    A two dimensional numpy array containing n x s matrix
                         with TRUE in each row at the column in which the bus was in
                         that observation.
    :param decision_mat: A two dimensional numpy array contaning  a n x 2 vector
                         with 1 in the first row for maintaining and 1 in the second
                         for replacement.
    :param beta:         The discount factor.
    :type beta:          float
    :param max_it:       Maximum number of iterations for evaluating the fix point.
    :type max_it:        int

    :return: The negative loglikelihood value for minimizing the objective function.
    """
    costs = cost_func(num_states, maint_func, params)
    ev = calc_fixp(num_states, trans_mat, costs, beta, max_it)
    p_choice = choice_prob(ev, costs, beta)
    ll_prob = np.log(np.dot(p_choice.T, state_mat))
    return -np.sum(decision_mat * ll_prob)


# @numba.jit(nopython=True)
def lin_cost(num_states, params, scale=0.001):
    """
    This function describes a linear cost function, which Rust concludes is the most
    realistic maintenance function.

    :param num_states:  The size of the state space.
    :type num_states:   int
    :param params:      A numpy array containing the parameters shaping the cost
                        function.
    :param scale:       A factor for scaling the maintenance costs.

    :return: A numpy array containing the maintenance cost for each state.
    """
    states = np.arange(num_states)
    return states * scale * params[0]


# @numba.jit(nopython=True)
def cost_func(num_states, maint_func, params):
    """
    This function calculates a vector containing the costs for maintenance and
    replacement, without recognizing the future.

    :param num_states:  The size of the state space.
    :type num_states:   int
    :param maint_func:  The maintenance cost function. Only linear implemented so
                        far.
    :param params:      A numpy array containing the parameters shaping the cost
                        function.

    :return: A two dimensional numpy array containing in each row the cost for
             maintenance in the first and for replacement in the second column.
    """
    rc = params[0]
    maint_cost = maint_func(num_states, params[1:])
    repl_cost = np.full(maint_cost.shape, rc + maint_cost[0])
    return np.vstack((maint_cost, repl_cost)).T


# @numba.jit(nopython=True)
def choice_prob(ev, costs, beta):
    """
    This function calculates the choice probabilities to maintain or replace the
    bus engine for each state.

    :param ev:      A numpy array containing for each state the expected value
                    fixed point.
    :param costs:  A numpy array containing the parameters shaping the cost
                    function.

    :param beta:    The discount factor.
    :type beta:     float

    :return: A two dimensional numpy array containing in each row the choice
             probability for maintenance in the first and for replacement in second
             column.
    """
    s = ev.shape[0]
    util_main = beta * ev - costs[:, 0]  # Utility to maintain the bus
    util_repl = np.full(util_main.shape, beta * ev[0] - costs[0, 0] - costs[0, 1])
    util = np.vstack((util_main, util_repl)).T
    util = util - np.amin(util)
    pchoice = np.exp(util) / (np.sum(np.exp(util), axis=1).reshape(s, -1))
    return pchoice


# @numba.jit(nopython=True)
def calc_fixp(
    num_states, trans_mat, costs, beta, max_it=1000000, threshold=1e-12, switch_tol=1e-3
):
    """
    The function to calculate the expected value fix point.

    :param num_states:  The size of the state space.
    :type num_states:   int
    :param trans_mat:   A two dimensional numpy array containing a s x s markov
                        transition matrix.
    :param costs:       A two dimensional float numpy array containing for each
                        state the cost to maintain in the first and to replace the bus
                        engine in the second column.
    :param beta:        The discount factor.
    :type beta:         float
    :param threshold:   A threshold for the convergence. By default set to 1e-6.
    :type threshold:    float
    :param max_it:      Maximum number of iterations. By default set to 1000000.
    :type max_it:       int

    :return: A numpy array containing for each state the expected value fixed point.
    """
    it_count = 0
    # contract_count = 0
    # kanteorich_count = 0
    ev = np.zeros(num_states)
    ev_new = np.dot(trans_mat, np.log(np.sum(np.exp(-costs), axis=1)))
    converge_crit = np.max(np.abs(ev_new - ev))
    while converge_crit > threshold:
        ev = ev_new
        if converge_crit > switch_tol:
            ev_new = contraction_iteration(ev, trans_mat, costs, beta)
            # contract_count +=1
        else:
            ev_new = kantevorich_step(ev, trans_mat, costs, beta)
            # kanteorich_count += 1

        it_count += 1
        if it_count > max_it:
            break
        converge_crit = np.max(np.abs(ev_new - ev))
    # import pdb
    # pdb.set_trace()
    return ev_new


# @numba.jit(nopython=True)
def contraction_iteration(ev, trans_mat, costs, beta):
    maint_value = beta * ev - costs[:, 0]
    repl_value = beta * ev[0] - costs[0, 1] - costs[0, 0]

    # Select the minimal absolute value to rescale the value vector for the
    # exponential function.
    ev_min = maint_value[0]

    log_sum = ev_min + np.log(
        np.exp(maint_value - ev_min) + np.exp(repl_value - ev_min)
    )
    return np.dot(trans_mat, log_sum)


# @numba.jit(nopython=True)
def kantevorich_step(ev, trans_mat, costs, beta):
    state_size = ev.shape[0]
    choice_probs = choice_prob(ev, costs, beta)
    # Need to stack first column
    t_prime = trans_mat[:, 1:] * choice_probs[1:, 0]
    iteration_step = contraction_iteration(ev, trans_mat, costs, beta)
    eye_minus_t_prime_inv = np.linalg.inv(np.eye(state_size) - t_prime)
    return ev - np.dot(eye_minus_t_prime_inv, (ev - iteration_step))
