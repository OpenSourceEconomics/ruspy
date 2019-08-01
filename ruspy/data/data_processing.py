"""
This module creates a pickle file which contains the total number of observations for
each group. Therefore every DataFrame row contains the bus identifier, the state
variable and the according decision.
"""


import pandas as pd
import os
import numpy as np


def data_processing(init_dict):
    """
    This function processes data from pickle files, which have the structure as
    explained in the documentation of Rust, to a pandas DataFrame saved in a pickle
    file, which contains in each row the Bus ID, the current period, the current
    state of the bus and the decision in this period.

    :param init_dict: A dictionary containing the name of the group and the size of
    the milage bins, which is used to discretize the raw data.

    :return: The processed data as pandas dataframe.

    """
    dirname = os.path.dirname(__file__)
    df_end = pd.DataFrame()
    df_pool = pd.DataFrame()
    for group in init_dict["groups"].split(","):
        df = pd.read_pickle(dirname + "/pkl/group_data/" + group + ".pkl")
        repl = pd.Series(index=df.index, data=0, dtype=int)
        for j, i in enumerate(df.columns.values[11:]):
            df2 = df[["Bus_ID", i]]
            df2 = df2.assign(decision=0)
            for l in [1, 2]:
                df2.loc[repl == l, i] -= df.loc[repl == l, "Odo_" + str(l)]
            for m in df2.index:
                if i < df.columns.values[-1]:
                    # Check if the bus has a first replacement if it has occurred
                    if (
                        (df.iloc[m][i + 1] > df.iloc[m]["Odo_1"])
                        & (df.iloc[m]["Odo_1"] != 0)
                        & (repl[m] == 0)
                    ):
                        df2.at[m, "decision"] = 1
                        repl[m] = repl[m] + 1
                    # Now check for the second
                    elif (
                        (df.iloc[m][i + 1] > df.iloc[m]["Odo_2"])
                        & (df.iloc[m]["Odo_2"] != 0)
                        & (repl[m] == 1)
                    ):
                        df2.at[m, "decision"] = 1
                        repl[m] = repl[m] + 1
            df2 = df2.rename(columns={i: "state"})
            df_end = pd.concat([df_end, df2])

        num_bus = len(df_end["Bus_ID"].unique())
        num_periods = df_end.shape[0] / num_bus
        df_end["period"] = np.arange(num_periods).repeat(num_bus).astype(int)
        df_end[["state"]] = (df_end[["state"]] / init_dict["binsize"]).astype(int)
        df_end.sort_values(["Bus_ID", "period"], inplace=True)
        df_end.reset_index(drop=True, inplace=True)
        df_pool = pd.concat([df_pool, df_end], axis=0)
    df_pool.reset_index(drop=True, inplace=True)
    os.makedirs(dirname + "/pkl/replication_data", exist_ok=True)
    df_pool.to_pickle(
        dirname
        + "/pkl/replication_data/rep_"
        + init_dict["groups"]
        + "_"
        + str(init_dict["binsize"])
        + ".pkl"
    )
    return df_pool
