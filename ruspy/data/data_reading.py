"""
This module creates eight pickle files from the raw data files provided by John Rust.
Each file contains a pandas DataFrame indexed the same as in the documentation. The
files are named according to their group in the paper.
"""

import pandas as pd
import os


def data_reading():
    """
    This function reads the raw data files and saves each bus group in a separate
    pickle file. The structure of the raw data is documented in the readme file in
    the subfolder original_data. The relevant information from this readme is stored in
    the two dictionaries initialized in the function.

    :return: Saves eight pickle files in pkl/group_data
    """

    dict_data = {
        "g870": [36, 15, "group_1"],
        "rt50": [60, 4, "group_2"],
        "t8h203": [81, 48, "group_3"],
        "a452372": [137, 18, "group_8"],
        "a452374": [137, 10, "group_6"],
        "a530872": [137, 18, "group_7"],
        "a530874": [137, 12, "group_5"],
        "a530875": [128, 37, "group_4"],
    }
    re_col = {
        1: "Bus_ID",
        2: "Month_pur",
        3: "Year_pur",
        4: "Month_1st",
        5: "Year_1st",
        6: "Odo_1",
        7: "Month_2nd",
        8: "Year_2nd",
        9: "Odo_2",
        10: "Month_begin",
        11: "Year_begin",
    }

    dirname = os.path.dirname(__file__)
    for keys in dict_data:
        r, c = dict_data[keys][0], dict_data[keys][1]
        file_cols = open(dirname + "/original_data/" + keys + ".asc").read().split("\n")
        df = pd.DataFrame()
        for j in range(0, c):
            for k in range(j * r, (j + 1) * r):
                df.loc[(k - j * r) + 1, j + 1] = float(file_cols[k])
        df = df.transpose()
        df.rename(columns=re_col, inplace=True)
        df["Bus_ID"] = df["Bus_ID"].astype(int)
        df.reset_index(inplace=True)
        df.drop(df.columns[[0]], axis=1, inplace=True)
        os.makedirs(dirname + "/pkl/group_data", exist_ok=True)
        df.to_pickle(dirname + "/pkl/group_data/" + dict_data[keys][2] + ".pkl")
