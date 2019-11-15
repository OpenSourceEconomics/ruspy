import numpy as np

from ruspy.estimation.estimation_cost_parameters import choice_prob


def calc_fixp(
    trans_mat,
    costs,
    beta,
    threshold=1e-16,
    switch_tol=1e-3,
    max_contr_steps=20,
    max_newt_kant_steps=20,
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
    contr_step_count = 0
    newt_kante_step_count = 0
    ev_new = np.dot(trans_mat, np.log(np.sum(np.exp(-costs), axis=1)))
    converge_crit = threshold + 1  # Make sure that the loop starts
    while converge_crit > threshold:
        while converge_crit > switch_tol:
            ev = ev_new
            ev_new = contraction_iteration(ev, trans_mat, costs, beta)
            contr_step_count += 1
            if contr_step_count > max_contr_steps:
                break
            converge_crit = np.amax(np.abs(ev_new - ev))
        ev = ev_new
        ev_new = kantevorich_step(ev, trans_mat, costs, beta)
        newt_kante_step_count += 1
        if newt_kante_step_count > max_newt_kant_steps:
            break
        converge_crit = calc_convergence_crit(ev_new, trans_mat, costs, beta)
    return ev_new


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


def kantevorich_step(ev, trans_mat, costs, beta):
    state_size = ev.shape[0]
    choice_probs = choice_prob(ev, costs, beta)
    t_prime_pre = trans_mat[:, 1:] * choice_probs[1:, 0]
    t_prime = beta * np.column_stack((1 - np.sum(t_prime_pre, axis=1), t_prime_pre))
    iteration_step = contraction_iteration(ev, trans_mat, costs, beta)
    return ev - np.linalg.lstsq(np.eye(state_size) - t_prime, (ev - iteration_step))[0]


def calc_convergence_crit(ev, trans_mat, costs, beta):
    return np.amax(np.abs(ev - contraction_iteration(ev, trans_mat, costs, beta)))
