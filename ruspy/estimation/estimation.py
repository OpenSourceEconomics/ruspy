"""
This module contains the main function for the estimation process.
"""
import numpy as np
import scipy.optimize as opt

from ruspy.estimation.est_cost_params import create_state_matrix
from ruspy.estimation.est_cost_params import derivative_loglike_cost_params
from ruspy.estimation.est_cost_params import loglike_cost_params
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation_transitions import estimate_transitions
from ruspy.model_code.cost_functions import cubic_costs
from ruspy.model_code.cost_functions import cubic_costs_dev
from ruspy.model_code.cost_functions import hyperbolic_costs
from ruspy.model_code.cost_functions import hyperbolic_costs_dev
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.cost_functions import lin_cost_dev
from ruspy.model_code.cost_functions import quadratic_costs
from ruspy.model_code.cost_functions import quadratic_costs_dev
from ruspy.model_code.cost_functions import sqrt_costs
from ruspy.model_code.cost_functions import sqrt_costs_dev


def estimate(init_dict, df):
    """
    This function calls the auxiliary functions to estimate the decision parameters.
    Therefore it manages the estimation process. As mentioned in the model theory
    chapter of the paper, the estimation of the transition probabilities and the
    estimation of the parameters shaping the cost function
    are completely separate.

    :param init_dict: A dictionary containing the following variables as keys:

        :beta: (float)       : Discount factor.
        :states: (int)       : The size of the statespace.
        :maint_func: (func)  : The maintenance cost function. Default is the linear
                               from the paper.

    :param df:        A pandas dataframe, which contains for each observation the Bus
                      ID, the current state of the bus, the current period and the
                      decision made in this period.

    :return: The function returns the optimization result of the transition
             probabilities and of the cost parameters as separate dictionaries.

    """
    beta = init_dict["beta"]
    transition_results = estimate_transitions(df)
    endog = df.loc[(slice(None), slice(1, None)), "decision"].to_numpy()
    states = df.loc[(slice(None), slice(1, None)), "state"].to_numpy()
    num_states = init_dict["states"]

    maint_func, maint_func_dev, num_params = select_cost_function(
        maint_cost_func_name=init_dict["maint_cost_func"]
    )
    decision_mat = np.vstack(((1 - endog), endog))
    trans_mat = create_transition_matrix(num_states, np.array(transition_results["x"]))
    state_mat = create_state_matrix(states, num_states)

    optimizer_options = select_optimizer_options(init_dict, num_params)

    result = opt.minimize(
        loglike_cost_params,
        args=(
            maint_func,
            maint_func_dev,
            num_states,
            trans_mat,
            state_mat,
            decision_mat,
            beta,
        ),
        **optimizer_options
    )
    return transition_results, result


def select_cost_function(maint_cost_func_name):
    if maint_cost_func_name == "cubic":
        maint_func = cubic_costs
        maint_func_dev = cubic_costs_dev
        num_params = 4
    elif maint_cost_func_name == "quadratic":
        maint_func = quadratic_costs
        maint_func_dev = quadratic_costs_dev
        num_params = 3
    elif maint_cost_func_name == "square_root":
        maint_func = sqrt_costs
        maint_func_dev = sqrt_costs_dev
        num_params = 2
    elif maint_cost_func_name == "hyperbolic":
        maint_func = hyperbolic_costs
        maint_func_dev = hyperbolic_costs_dev
        num_params = 2
    # Linear is the standard
    else:
        maint_func = lin_cost
        maint_func_dev = lin_cost_dev
        num_params = 2
    return maint_func, maint_func_dev, num_params


def select_optimizer_options(init_dict, num_params_costs):

    optimizer_dict = {} if "optimizer" in init_dict else init_dict["optimizer"]
    optimizer_options = {}

    if "optimizer_name" in optimizer_dict:
        optimizer_options["method"] = optimizer_dict["optimizer_name"]
    else:
        optimizer_options["method"] = "L-BFGS-B"

    if "start_values" in optimizer_dict:
        optimizer_options["x0"] = np.array(optimizer_dict["start_values"])
    else:
        optimizer_options["x0"] = np.power(
            np.full(num_params_costs, 10, dtype=float),
            np.arange(1, -num_params_costs + 1, -1),
        )

    if "search_bounds" in optimizer_dict:
        if "search_bounds" == "yes":
            optimizer_options["bounds"] = [
                (np.finfo(float).eps, None)
            ] * num_params_costs
        else:
            optimizer_options["bounds"] = np.array(optimizer_dict["search_bounds"])

    if "use_gradient" == "yes":
        optimizer_options["jac"] = derivative_loglike_cost_params
    else:
        pass

    if "additional_options" in optimizer_dict:
        optimizer_options["options"] = optimizer_dict["additional_options"]
    else:
        optimizer_options["options"] = {}

    return optimizer_options
