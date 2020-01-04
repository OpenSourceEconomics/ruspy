import numpy as np

from ruspy.model_code.choice_probabilities import choice_prob_gumbel


def calc_fixp(
    trans_mat,
    obs_costs,
    disc_fac,
    threshold=1e-12,
    switch_tol=1e-3,
    max_contr_steps=20,
    max_newt_kant_steps=20,
):
    """
    Calculating the expected value of maintenance fix-point with the polyalgorithm
    proposed by Rust (1987) and Rust (2000).

    Parameters
    ----------
    trans_mat : numpy.array
        see :ref:`trans_mat`
    obs_costs : numpy.array
        see :ref:`costs`
    disc_fac : numpy.float
        see :ref:`disc_fac`
    threshold : numpy.float
        see :ref:`alg_details`
    switch_tol : numpy.float
        see :ref:`alg_details`
    max_contr_steps : int
        see :ref:`alg_details`
    max_newt_kant_steps : int
        see :ref:`alg_details`

    Returns
    -------
    ev_new : numpy.array
        see :ref:`ev`
    """
    contr_step_count = 0
    newt_kante_step_count = 0
    ev_new = np.dot(trans_mat, np.log(np.sum(np.exp(-obs_costs), axis=1)))
    converge_crit = threshold + 1  # Make sure that the loop starts
    while converge_crit > threshold:
        while converge_crit > switch_tol:
            ev = ev_new
            ev_new = contraction_iteration(ev, trans_mat, obs_costs, disc_fac)
            contr_step_count += 1
            if contr_step_count > max_contr_steps:
                break
            converge_crit = np.max(np.abs(ev_new - ev))
        ev = ev_new
        ev_new = kantevorich_step(ev, trans_mat, obs_costs, disc_fac)
        newt_kante_step_count += 1
        if newt_kante_step_count > max_newt_kant_steps:
            break
        converge_crit = np.max(
            np.abs(ev - contraction_iteration(ev, trans_mat, obs_costs, disc_fac))
        )
    return ev_new


def contraction_iteration(ev, trans_mat, obs_costs, disc_fac):
    """
    Calculating one iteration of the contraction mapping.

    Parameters
    ----------
    ev : numpy.array
        see :ref:`ev`
    trans_mat : numpy.array
        see :ref:`trans_mat`
    obs_costs : numpy.array
        see :ref:`costs`
    disc_fac : numpy.float
        see :ref:`disc_fac`

    Returns
    -------
    ev_new : numpy.array
        see :ref:`ev`


    """
    maint_value = disc_fac * ev - obs_costs[:, 0]
    repl_value = disc_fac * ev[0] - obs_costs[0, 1] - obs_costs[0, 0]

    # Select the minimal absolute value to rescale the value vector for the
    # exponential function.

    ev_max = np.max(np.array(maint_value, repl_value))

    log_sum = ev_max + np.log(
        np.exp(maint_value - ev_max) + np.exp(repl_value - ev_max)
    )

    ev_new = np.dot(trans_mat, log_sum)
    return ev_new


def kantevorich_step(ev, trans_mat, costs, disc_fac):
    """
    Calculating one Newton-Kantevorich step for approximating the fix-point.

    Parameters
    ----------
    ev : numpy.array
        see :ref:`ev`
    trans_mat : numpy.array
        see :ref:`trans_mat`
    obs_costs : numpy.array
        see :ref:`costs`
    disc_fac : numpy.float
        see :ref:`disc_fac`

    Returns
    -------
    ev_new : numpy.array
        see :ref:`ev`


    """
    iteration_step = contraction_iteration(ev, trans_mat, costs, disc_fac)
    ev_diff = solve_equ_system_fixp(ev - iteration_step, ev, trans_mat, costs, disc_fac)
    ev_new = ev - ev_diff
    return ev_new


def solve_equ_system_fixp(fixp_vector, ev, trans_mat, costs, disc_fac):
    num_states = ev.shape[0]
    t_prime = cont_op_dev_wrt_fixp(ev, trans_mat, costs, disc_fac)
    sol = np.linalg.lstsq(np.eye(num_states) - t_prime, fixp_vector, rcond=None)[0]
    return sol


def cont_op_dev_wrt_fixp(ev, trans_mat, costs, disc_fac):
    choice_probs = choice_prob_gumbel(ev, costs, disc_fac)
    t_prime_pre = trans_mat[:, 1:] * choice_probs[1:, 0]
    t_prime = disc_fac * np.column_stack((1 - np.sum(t_prime_pre, axis=1), t_prime_pre))
    return t_prime


def contr_op_dev_wrt_params(trans_mat, maint_choice_prob, cost_dev):
    dev = np.dot(trans_mat, np.multiply(-cost_dev, maint_choice_prob))
    return dev


def contr_op_dev_wrt_rc(trans_mat, maint_choice_prob):
    return np.dot(trans_mat, maint_choice_prob - 1)
