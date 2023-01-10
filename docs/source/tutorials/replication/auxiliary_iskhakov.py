import numpy as np
import pandas as pd


def check_simulated_data(
    simulated_data,
    discount_factor,
    number_runs,
):
    """
    generates some key statistics of the simulated data set.

    Parameters
    ----------
    simulated_data : pd.DataFrame
        The previously simulated data.
    discount_factor : list
        The discount used in the simulation.
    number_runs : int
        The number of simlation runs.

    Returns
    -------
    results : pd.DataFrame
        The resulting key statistics of the data sets per discount factor.

    """
    columns = [
        "Average State at Replacement",
        "Average of all States",
        "Average Replacement",
    ]
    results = pd.DataFrame(index=discount_factor, columns=columns)
    results.index.name = "Discount Factor"
    for factor in discount_factor:
        temp = np.ones((number_runs, len(columns)))
        for run in np.arange(number_runs):
            temp[run, 0] = (
                simulated_data[factor][run]
                .loc[simulated_data[factor][run]["decision"] == 1, "state"]
                .mean()
            )
            temp[run, 1] = simulated_data[factor][run]["state"].mean()
            temp[run, 2] = simulated_data[factor][run]["decision"].mean()
        results.loc[factor, :] = temp.mean(axis=0)

    return results
