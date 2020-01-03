import numpy as np

from ruspy.estimation.est_cost_params import derivative_loglike_cost_params
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


def select_model_parameters(init_dict):
    if "model_specifications" not in init_dict:
        raise ValueError("Specify model parameters")
    model_specification = init_dict["model_specifications"]

    disc_fac = model_specification["discount_factor"]
    num_states = model_specification["number_states"]
    scale = model_specification["cost_scale"]

    maint_func, maint_func_dev, num_params = select_cost_function(
        model_specification["maint_cost_func"]
    )

    return disc_fac, num_states, maint_func, maint_func_dev, num_params, scale


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

    optimizer_dict = {} if "optimizer" not in init_dict else init_dict["optimizer"]
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

    if "use_search_bounds" in optimizer_dict:
        if optimizer_dict["use_search_bounds"] == "yes":
            optimizer_options["bounds"] = [(1e-6, None)] * num_params_costs
        else:
            pass
    else:
        pass

    if "search_bounds" in optimizer_dict:
        optimizer_options["bounds"] = np.array(optimizer_dict["search_bounds"])
    else:
        pass

    if "use_gradient" in optimizer_dict:
        if optimizer_dict["use_gradient"] == "yes":
            optimizer_options["jac"] = derivative_loglike_cost_params
        else:
            pass
    else:
        pass

    if "additional_options" in optimizer_dict:
        optimizer_options["options"] = optimizer_dict["additional_options"]
    else:
        optimizer_options["options"] = {}

    return optimizer_options
