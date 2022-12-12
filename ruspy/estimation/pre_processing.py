"""
This module specifies the model specifications and the cost
function from the initialisation dictionary init_dict.

change to the previous version:
select_optimizer_options is not needed
"""
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
    """
        Selecting the model specifications.

    Parameters
    ----------
    init_dict : dictionary
        see :ref:`init_dict`
    Returns
    -------
        The model sepcifications.
    """
    if "model_specifications" not in init_dict:
        raise ValueError("Specify model parameters")
    model_specification = init_dict["model_specifications"]

    disc_fac = model_specification["discount_factor"]
    num_states = model_specification["num_states"]
    scale = model_specification["cost_scale"]

    maint_func, maint_func_dev, num_params = select_cost_function(
        model_specification["maint_cost_func"]
    )

    return disc_fac, num_states, maint_func, maint_func_dev, num_params, scale


def select_cost_function(maint_cost_func_name):
    """
        Selecting the maintenance cost function.

    Parameters
    ----------
    maint_cost_func_name : string
        The name of the maintenance cost function.

    Returns
    -------
        The maintenance cost function, its derivative and the number of cost
        parameters in this model.
    """
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
