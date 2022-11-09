"""
This module contains the main function for the estimation process.
"""
import time
from functools import partial

import nlopt
import numpy as np
from estimagic.optimization.optimize import minimize

try:
    import ipopt  # noqa:F401

    optional_package_is_available = True
except ImportError:
    optional_package_is_available = False

if optional_package_is_available:
    from ipopt import minimize_ipopt
from scipy.optimize._numdiff import approx_derivative

from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.estimation import config
from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.estimation_interface import select_model_parameters
from ruspy.estimation.estimation_interface import select_optimizer_options
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.estimation.mpec import mpec_constraint
from ruspy.estimation.mpec import mpec_constraint_derivative
from ruspy.estimation.mpec import mpec_loglike_cost_params
from ruspy.estimation.mpec import mpec_loglike_cost_params_derivative
from ruspy.estimation.mpec import wrap_ipopt_constraint
from ruspy.estimation.mpec import wrap_ipopt_likelihood
from ruspy.estimation.mpec import wrap_mpec_loglike


def estimate(
    init_dict,
    df,
):
    """
    Estimation function of ruspy.
    This function coordinates the estimation process of the ruspy package.

    Parameters
    ----------
    init_dict : dictionary
        see :ref:`init_dict`
    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    results_transition_params : dictionary
        see :ref:`result_trans`
    results_cost_params : dictionary
        see :ref:`result_costs`

    """
    transition_results = estimate_transitions(df)

    endog = df.loc[(slice(None), slice(1, None)), "decision"].to_numpy(int)
    states = df.loc[(slice(None), slice(1, None)), "state"].to_numpy(int)

    args = (
        disc_fac,
        num_states,
        maint_func,
        maint_func_dev,
        num_params,
        scale,
    ) = select_model_parameters(init_dict)

    decision_mat = np.vstack(((1 - endog), endog))
    trans_mat = create_transition_matrix(num_states, np.array(transition_results["x"]))
    state_mat = create_state_matrix(states, num_states)

    optimizer_options = select_optimizer_options(init_dict, num_params, num_states)

    if "approach" in init_dict["optimizer"]:
        approach = init_dict["optimizer"]["approach"]
    else:
        raise ValueError("The key 'approach' must be in the optimizer dictionairy")

    if approach == "NFXP":
        alg_details = {} if "alg_details" not in init_dict else init_dict["alg_details"]
        results_transition_params, results_cost_params = estimate_nfxp(
            *args,
            decision_mat,
            trans_mat,
            state_mat,
            optimizer_options,
            transition_results,
            alg_details,
        )
    elif approach == "MPEC" and optimizer_options["algorithm"] == "ipopt":
        results_transition_params, results_cost_params = estimate_mpec_ipopt(
            *args,
            decision_mat,
            trans_mat,
            state_mat,
            optimizer_options,
            transition_results,
        )
    elif approach == "MPEC" and optimizer_options["algorithm"] != "ipopt":
        results_transition_params, results_cost_params = estimate_mpec_nlopt(
            *args,
            decision_mat,
            trans_mat,
            state_mat,
            optimizer_options,
            transition_results,
        )
    else:
        raise ValueError(
            f"{approach} is not implemented. Only MPEC or NFXP are valid choices"
        )

    return results_transition_params, results_cost_params


def estimate_nfxp(
    disc_fac,
    num_states,
    maint_func,
    maint_func_dev,
    num_params,
    scale,
    decision_mat,
    trans_mat,
    state_mat,
    optimizer_options,
    transition_results,
    alg_details,
):
    """
    Estimation function for the nested fixed point algorithm in ruspy.


    Parameters
    ----------
    disc_fac : numpy.float
        see :ref:`disc_fac`
    num_states : int
        The size of the state space.
    maint_func: func
        see :ref: `maint_func`
    maint_func_dev: func
        see :ref: `maint_func_dev`
    num_params : int
        The number of parameters to be estimated.
    scale : numpy.float
        see :ref:`scale`
    decision_mat : numpy.array
        see :ref:`decision_mat`
    trans_mat : numpy.array
        see :ref:`trans_mat`
    state_mat : numpy.array
        see :ref:`state_mat`
    optimizer_options : dict
        The options chosen for the optimization algorithm in the initialization
        dictionairy.
    transition_results : dict
        The results from ``estimate_transitions``.
    alg_details : dict
        see :ref: `alg_details`


    Returns
    -------
    transition_results : dictionary
        see :ref:`result_trans`
    result_cost_params : dictionary
        see :ref:`result_costs`

    """

    criterion_temp = optimizer_options.pop("criterion")
    n_evaluations, criterion = wrap_nfxp_criterion(criterion_temp)

    kwargs = {
        "maint_func": maint_func,
        "maint_func_dev": maint_func_dev,
        "num_states": num_states,
        "trans_mat": trans_mat,
        "state_mat": state_mat,
        "decision_mat": decision_mat,
        "disc_fac": disc_fac,
        "scale": scale,
        "alg_details": alg_details,
    }

    result_cost_params = {}

    tic = time.perf_counter()
    min_result = minimize(
        criterion,
        criterion_kwargs=kwargs,
        derivative_kwargs=kwargs,
        **optimizer_options,
    )
    toc = time.perf_counter()
    timing = toc - tic

    result_cost_params["x"] = min_result["solution_params"]["value"].to_numpy()
    result_cost_params["fun"] = min_result["solution_criterion"]
    if min_result["success"]:
        result_cost_params["status"] = 1
    else:
        result_cost_params["status"] = 0
    result_cost_params["message"] = min_result["message"]
    result_cost_params["jac"] = min_result["solution_derivative"]
    result_cost_params["n_evaluations"] = min_result["n_criterion_evaluations"]
    result_cost_params["n_iterations"] = min_result["n_iterations"]
    result_cost_params["n_contraction_steps"] = config.total_contr_count
    result_cost_params["n_newt_kant_steps"] = config.total_newt_kant_count
    result_cost_params["time"] = timing

    config.total_contr_count = 0
    config.total_newt_kant_count = 0

    return transition_results, result_cost_params


def estimate_mpec_nlopt(
    disc_fac,
    num_states,
    maint_func,
    maint_func_dev,
    num_params,
    scale,
    decision_mat,
    trans_mat,
    state_mat,
    optimizer_options,
    transition_results,
):
    """
    Estimation function of Mathematical Programming with Equilibrium Constraints
    (MPEC) in ruspy.


    Parameters
    ----------
    disc_fac : numpy.float
        see :ref:`disc_fac`
    num_states : int
        The size of the state space.
    maint_func: func
        see :ref: `maint_func`
    maint_func_dev: func
        see :ref: `maint_func_dev`
    num_params : int
        The number of parameters to be estimated.
    scale : numpy.float
        see :ref:`scale`
    decision_mat : numpy.array
        see :ref:`decision_mat`
    trans_mat : numpy.array
        see :ref:`trans_mat`
    state_mat : numpy.array
        see :ref:`state_mat`
    optimizer_options : dict
        The options chosen for the optimization algorithm in the initialization
        dictionairy.
    transition_results : dict
        The results from ``estimate_transitions``.

    Returns
    -------
    transition_results : dictionary
        see :ref:`result_trans`
    mpec_cost_parameters : dictionary
        see :ref:`result_costs`


    """

    gradient = optimizer_options.pop("derivative")

    # Calculate partial functions needed for nlopt
    n_evaluations, partial_loglike_mpec = wrap_mpec_loglike(
        args=(
            maint_func,
            maint_func_dev,
            num_states,
            num_params,
            state_mat,
            decision_mat,
            disc_fac,
            scale,
            gradient,
        )
    )

    partial_constr_mpec = partial(
        mpec_constraint,
        maint_func,
        maint_func_dev,
        num_states,
        num_params,
        trans_mat,
        disc_fac,
        scale,
        gradient,
    )

    # set up nlopt
    opt = nlopt.opt(
        eval("nlopt." + optimizer_options.pop("algorithm")), num_states + num_params
    )
    opt.set_min_objective(partial_loglike_mpec)
    opt.add_equality_mconstraint(
        partial_constr_mpec,
        np.full(num_states, 1e-6),
    )

    # supply user choices
    params = optimizer_options.pop("params")
    if "get_expected_values" in optimizer_options:
        get_expected_values = optimizer_options.pop("get_expected_values")
    else:
        get_expected_values = "No"

    if "set_local_optimizer" in optimizer_options:
        sub = nlopt.opt(  # noqa: F841
            eval("nlopt." + optimizer_options.pop("set_local_optimizer")),
            num_states + num_params,
        )
        exec("opt.set_local_optimizer(sub)")
        for key, _value in optimizer_options.items():
            exec("sub." + key + "(_value)")

    for key, _value in optimizer_options.items():
        exec("opt." + key + "(_value)")

    # Solving nlopt
    tic = time.perf_counter()
    if get_expected_values == "Yes":
        obs_costs = calc_obs_costs(num_states, maint_func, params, scale)
        ev = calc_fixp(trans_mat, obs_costs, disc_fac)[0]
        params = np.concatenate((ev, params))
    result = opt.optimize(params)
    toc = time.perf_counter()
    timing = toc - tic

    mpec_cost_parameters = {}
    mpec_cost_parameters["x"] = result
    mpec_cost_parameters["fun"] = opt.last_optimum_value()
    if opt.last_optimize_result() > 0:
        mpec_cost_parameters["status"] = True
    else:
        mpec_cost_parameters["status"] = False
    mpec_cost_parameters["n_iterations"] = opt.get_numevals()
    mpec_cost_parameters["n_evaluations"] = n_evaluations[0]
    mpec_cost_parameters["reason"] = opt.last_optimize_result()
    mpec_cost_parameters["time"] = timing

    return transition_results, mpec_cost_parameters


def estimate_mpec_ipopt(
    disc_fac,
    num_states,
    maint_func,
    maint_func_dev,
    num_params,
    scale,
    decision_mat,
    trans_mat,
    state_mat,
    optimizer_options,
    transition_results,
):
    """
    Estimation function of Mathematical Programming with Equilibrium Constraints
    (MPEC) in ruspy.


    Parameters
    ----------
    disc_fac : numpy.float
        see :ref:`disc_fac`
    num_states : int
        The size of the state space.
    maint_func: func
        see :ref: `maint_func`
    maint_func_dev: func
        see :ref: `maint_func_dev`
    num_params : int
        The number of parameters to be estimated.
    scale : numpy.float
        see :ref:`scale`
    decision_mat : numpy.array
        see :ref:`decision_mat`
    trans_mat : numpy.array
        see :ref:`trans_mat`
    state_mat : numpy.array
        see :ref:`state_mat`
    optimizer_options : dict
        The options chosen for the optimization algorithm in the initialization
        dictionairy.
    transition_results : dict
        The results from ``estimate_transitions``.

    Returns
    -------
    transition_results : dictionary
        see :ref:`result_trans`
    mpec_cost_parameters : dictionary
        see :ref:`result_costs`


    """

    if not optional_package_is_available:
        raise NotImplementedError(
            """To use this you need to install cyipopt. If you are mac or Linux user
            the command is $ conda install -c conda-forge cyipopt. If you use
            Windows you have to install from source. A description can be found
            here: https://github.com/matthias-k/cyipopt"""
        )

    del optimizer_options["algorithm"]
    gradient = optimizer_options.pop("derivative")
    params = optimizer_options.pop("params")
    lower_bounds = optimizer_options.pop("set_lower_bounds")
    upper_bounds = optimizer_options.pop("set_upper_bounds")
    bounds = np.vstack((lower_bounds, upper_bounds)).T
    bounds = list(map(tuple, bounds))
    if "get_expected_values" in optimizer_options:
        get_expected_values = optimizer_options.pop("get_expected_values")
    else:
        get_expected_values = "No"

    n_evaluations, neg_criterion = wrap_ipopt_likelihood(
        mpec_loglike_cost_params,
        args=(
            maint_func,
            maint_func_dev,
            num_states,
            num_params,
            state_mat,
            decision_mat,
            disc_fac,
            scale,
        ),
    )

    constraint_func = wrap_ipopt_constraint(
        mpec_constraint,
        args=(
            maint_func,
            maint_func_dev,
            num_states,
            num_params,
            trans_mat,
            disc_fac,
            scale,
        ),
    )

    if gradient == "No":

        def approx_gradient(params):
            fun = approx_derivative(neg_criterion, params, method="2-point")
            return fun

        gradient_func = approx_gradient

        def approx_jacobian(params):
            fun = approx_derivative(constraint_func, params, method="2-point")
            return fun

        jacobian_func = approx_jacobian
    else:
        gradient_func = partial(
            mpec_loglike_cost_params_derivative,
            maint_func,
            maint_func_dev,
            num_states,
            num_params,
            disc_fac,
            scale,
            decision_mat,
            state_mat,
        )
        jacobian_func = partial(
            mpec_constraint_derivative,
            maint_func,
            maint_func_dev,
            num_states,
            num_params,
            disc_fac,
            scale,
            trans_mat,
        )

    constraints = {
        "type": "eq",
        "fun": constraint_func,
        "jac": jacobian_func,
    }

    tic = time.perf_counter()
    if get_expected_values == "Yes":
        obs_costs = calc_obs_costs(num_states, maint_func, params, scale)
        ev = calc_fixp(trans_mat, obs_costs, disc_fac)[0]
        params = np.concatenate((ev, params))
    results_ipopt = minimize_ipopt(
        neg_criterion,
        params,
        bounds=bounds,
        jac=gradient_func,
        constraints=constraints,
        **optimizer_options,
    )
    toc = time.perf_counter()
    timing = toc - tic

    mpec_cost_parameters = {}
    mpec_cost_parameters["x"] = results_ipopt["x"]
    mpec_cost_parameters["fun"] = results_ipopt["fun"]
    if results_ipopt["success"] is True:
        mpec_cost_parameters["status"] = True
    else:
        mpec_cost_parameters["status"] = False
    mpec_cost_parameters["n_iterations"] = results_ipopt["nit"]
    mpec_cost_parameters["n_evaluations"] = results_ipopt["nfev"]
    mpec_cost_parameters["time"] = timing
    mpec_cost_parameters["n_evaluations_total"] = n_evaluations[0]

    return transition_results, mpec_cost_parameters


def wrap_nfxp_criterion(function):
    ncalls = [0]

    def function_wrapper(*wrapper_args, **wrapper_kwargs):
        ncalls[0] += 1
        return function(*wrapper_args, **wrapper_kwargs)

    return ncalls, function_wrapper
