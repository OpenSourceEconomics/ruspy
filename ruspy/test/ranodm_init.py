"""This function provides an random init file generating process."""
import collections

import numpy as np
import yaml


def random_init(constr=None):
    """
    The module provides a random dictionary generating process for test purposes.
    """
    # Check for pre specified constraints
    if constr is not None:
        pass
    else:
        constr = {}

    keys = constr.keys()
    if "BUSES" in keys:
        agents = constr["BUSES"]
    else:
        agents = np.random.randint(20, 100)

    if "disc_fac" in keys:
        disc_fac = constr["disc_fac"]
    else:
        disc_fac = np.random.uniform(0.9, 0.999)

    if "PERIODS" in keys:
        periods = constr["PERIODS"]
    else:
        periods = np.random.randint(1000, 10000)

    if "SEED" in keys:
        seed = constr["SEED"]
    else:
        seed = np.random.randint(1000, 9999)

    if "MAINT_FUNC" in keys:
        maint_func = constr["MAINT_FUNC"]
    else:
        maint_func = "linear"

    init_dict = {}

    for key_ in ["simulation", "estimation"]:
        init_dict[key_] = {}

    init_dict["simulation"]["periods"] = periods
    init_dict["simulation"]["buses"] = agents
    init_dict["simulation"]["disc_fac"] = disc_fac
    init_dict["simulation"]["seed"] = seed
    init_dict["simulation"]["maint_func"] = maint_func

    init_dict["estimation"]["states"] = np.random.randint(100, 150)
    init_dict["estimation"]["disc_fac"] = disc_fac
    init_dict["estimation"]["maint_func"] = maint_func

    # Generate random parameterization

    # Draw probabilities:
    p1 = np.random.uniform(0.37, 0.42)
    p2 = np.random.uniform(0.55, 0.58)
    p3 = 1 - p1 - p2
    init_dict["simulation"]["known_trans"] = [p1, p2, p3]

    return init_dict


def print_dict(init_dict, file_name="test"):
    """
    This function prints the initialization dict to a *.yml file.
    """
    ordered_dict = collections.OrderedDict()
    order = ["simulation", "estimation"]
    for key_ in order:
        ordered_dict[key_] = init_dict[key_]

    with open(f"{file_name}.ruspy.yml", "w") as outfile:
        yaml.dump(ordered_dict, outfile, explicit_start=True, indent=4)
