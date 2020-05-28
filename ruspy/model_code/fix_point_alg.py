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
    newt_kant_step_count = 0
    ev_new = np.dot(trans_mat, np.log(np.sum(np.exp(-obs_costs), axis=1)))
    converge_crit = threshold + 1  # Make sure that the loop starts
    while converge_crit > threshold:
        while converge_crit > switch_tol and contr_step_count < max_contr_steps:
            ev = ev_new
            ev_new = contraction_iteration(ev, trans_mat, obs_costs, disc_fac)
            contr_step_count += 1
            converge_crit = np.max(np.abs(ev_new - ev))
        ev = ev_new
        ev_new = kantorovich_step(ev, trans_mat, obs_costs, disc_fac)
        newt_kant_step_count += 1
        if newt_kant_step_count > max_newt_kant_steps:
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


def kantorovich_step(ev, trans_mat, obs_costs, disc_fac):
    """
    Calculating one Newton-Kantorovich step for approximating the fix-point.

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
    iteration_step = contraction_iteration(ev, trans_mat, obs_costs, disc_fac)
    ev_diff = solve_equ_system_fixp(
        ev - iteration_step, ev, trans_mat, obs_costs, disc_fac
    )
    ev_new = ev - ev_diff
    return ev_new


def solve_equ_system_fixp(fixp_vector, ev, trans_mat, obs_costs, disc_fac):
    """
    Solving the multiple used equation system, deviated from the implicit
    function theorem


    Parameters
    ----------
    fixp_vector: numpy.array
        A state space sized containing the right hand side of the euqation.
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
    sol : numpy.array
        The state space sized solution of the equation.

    """
    num_states = ev.shape[0]
    t_prime = frechnet_dev(ev, trans_mat, obs_costs, disc_fac)
    sol = np.linalg.lstsq(np.eye(num_states) - t_prime, fixp_vector, rcond=None)[0]
    return sol


def frechnet_dev(ev, trans_mat, obs_costs, disc_fac):
    """
    Calculating the Frechnet derivative of the contraction mapping.

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
    t_prime : numpy.array
        A num_states x num_states matrix containing the frechnet derivative of the
        contraction mapping. For details see Rust (2000).

    """
    choice_probs = choice_prob_gumbel(ev, obs_costs, disc_fac)
    t_prime_pre = trans_mat[:, 1:] * choice_probs[1:, 0]
    t_prime = disc_fac * np.column_stack((1 - np.sum(t_prime_pre, axis=1), t_prime_pre))
    return t_prime


def contr_op_dev_wrt_params(trans_mat, maint_choice_prob, cost_dev):
    """
    Calculating the derivative of the contraction mapping with respect to one
    particular maintenance cost parameter.

    Parameters
    ----------
    trans_mat : numpy.array
        see :ref:`trans_mat`
    maint_choice_prob : numpy.array
        A num_states sized one dimensional numpy array containing the derivative of the
        maintenance cost function with respect to one particular parameter.
    cost_dev : numpy.array
        A num_states sized one dimensional numpy array containing the derivative of the
        maintenance cost function with respect to one particular parameter.


    Returns
    -------
    dev : numpy.array
        A num_states sized one dimensional numpy array containing the derivative of the
        contraction mapping with respect to one particular maintenance cost parameter.

    """
    dev = np.dot(trans_mat, np.multiply(-cost_dev, maint_choice_prob))
    return dev


def contr_op_dev_wrt_rc(trans_mat, maint_choice_prob):
    """
    Calculating the derivative of the contraction mapping with respect to the
    replacement costs

    Parameters
    ----------
    trans_mat : numpy.array
        see :ref:`trans_mat`
    maint_choice_prob : numpy.array
        A num_states sized one dimensional numpy array containing the derivative of the
        maintenance cost function for one particular parameter.


    Returns
    -------
    dev : numpy.array
        A num_states sized one dimensional numpy array containing the derivative of the
        contraction mapping with respect to the replacement cost parameter.

    """
    dev = np.dot(trans_mat, maint_choice_prob - 1)
    return dev
