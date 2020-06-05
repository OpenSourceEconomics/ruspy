"""
This module contains the main function for the estimation process.
"""
import time
from functools import partial

import ipopt
import nlopt
import numpy as np
from estimagic.optimization.optimize import minimize
from scipy.optimize import approx_fprime
from scipy.optimize._numdiff import approx_derivative

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


def estimate(init_dict, df):

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

    This function coordinates the estimation process of the ruspy package.

    Parameters
    ----------
    init_dict : dictionary
        see ref:`_est_init_dict`

    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    transition_results : dictionary
        see :ref:`result_trans`
    result_cost_params : dictionary
        see :ref:`result_costs`

    """

    criterion = optimizer_options.pop("criterion")

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
        criterion, criterion_kwargs=kwargs, gradient_kwargs=kwargs, **optimizer_options,
    )
    toc = time.perf_counter()
    timing = toc - tic

    result_cost_params["x"] = min_result[1]["value"].to_numpy()
    result_cost_params["fun"] = min_result[0]["fitness"]
    result_cost_params["status"] = min_result[0]["status"]
    result_cost_params["message"] = min_result[0]["message"]
    result_cost_params["jac"] = min_result[0]["jacobian"]
    result_cost_params["n_evaluations"] = min_result[0]["n_evaluations"]
    result_cost_params["n_iterations"] = min_result[0]["n_iterations"]
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
    init_dict : dictionary
        see ref:`_est_init_dict`

    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    mpec_transition_results : dictionary
        see :ref:`mpec_transition_results`
    mpec_cost_parameters : dictionary
        see :ref:`mpec_cost_parameters`



    """

    gradient = optimizer_options.pop("gradient")

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
        partial_constr_mpec, np.full(num_states, 0),
    )

    # supply user choices
    params = optimizer_options.pop("params")

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
    init_dict : dictionary
        see ref:`_est_init_dict`

    df : pandas.DataFrame
        see :ref:`df`

    Returns
    -------
    mpec_transition_results : dictionary
        see :ref:`mpec_transition_results`
    mpec_cost_parameters : dictionary
        see :ref:`mpec_cost_parameters`



    """
    del optimizer_options["algorithm"]
    gradient = optimizer_options.pop("gradient")

    neg_criterion = wrap_ipopt_likelihood(
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

    class mpec_ipopt:
        def __init__(self):
            pass

        def objective(self):
            return neg_criterion(self)

        def gradient(self):
            if gradient == "No":
                gradient_value = approx_fprime(self, neg_criterion, 10e-6)
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
                gradient_value = gradient_func(self)
            return gradient_value

        def constraints(self):
            return constraint_func(self)

        def jacobian(self):
            if gradient == "No":
                jacobian_value = approx_derivative(constraint_func, self)
            else:
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
                jacobian_value = jacobian_func(self)
            return jacobian_value

    nlp = ipopt.problem(
        n=num_params + num_states,
        m=num_states,
        problem_obj=mpec_ipopt,
        lb=optimizer_options.pop("set_lower_bounds"),
        ub=optimizer_options.pop("set_upper_bounds"),
        cl=np.zeros(num_states),
        cu=np.zeros(num_states),
    )

    nlp.addOption("output_file", "results_ipopt.txt")
    nlp.addOption("file_print_level", 3)

    params = optimizer_options.pop("params")

    for key, value in optimizer_options.items():
        nlp.addOption(key, value)

    results_ipopt = nlp.solve(params)[1]

    mpec_cost_parameters = {}
    mpec_cost_parameters["time"] = 0.0
    mpec_cost_parameters["n_evaluations"] = 0

    if results_ipopt["status"] == 0:
        mpec_cost_parameters["status"] = 1 - results_ipopt["status"]
        file = open("results_ipopt.txt")
        lines = file.readlines()
        rows = [(11, "n_iterations"), (21, "n_evaluations"), (28, "time"), (29, "time")]
        for row, name in rows:
            if name != "n_iterations":
                mpec_cost_parameters[name] += float(lines[row].split("= ", 1)[1])
            else:
                mpec_cost_parameters[name] = int(lines[row].split(": ", 1)[1])
    else:
        mpec_cost_parameters["status"] = 0
        names = ["n_iterations", "n_evaluations", "time"]
        for name in names:
            mpec_cost_parameters[name] = np.nan

    mpec_cost_parameters["x"] = results_ipopt["x"]
    mpec_cost_parameters["fun"] = results_ipopt["obj_val"]

    return transition_results, mpec_cost_parameters
