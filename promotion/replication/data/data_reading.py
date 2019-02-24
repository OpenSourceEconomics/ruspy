"""
This module creates eight pickle files from the raw data files provided by John Rust.
Each containing a DataFrame indexed the same as in the documentation. The files are named according to their group
in the paper.

"""

import pandas as pd
import os


def data_reading():
    # The following dictionaries contain details on the raw data described by Rust in his documentation.
    # This Information is used to create 8 dataframes, each containing one of the Busgroups used in the paper.
    # Also the first 11 columns contain bus specific information on purchase,engine replacement etc.
    # For further information, see the documnetation.
    dict_data = {'g870': [36, 15, 'group_1'], 'rt50': [60, 4, 'group_2'],
                 't8h203': [81, 48, 'group_3'], 'a452372': [137, 18, 'group_8'],
                 'a452374': [137, 10, 'group_6'], 'a530872': [137, 18, 'group_7'],
                 'a530874': [137, 12, 'group_5'], 'a530875': [128, 37, 'group_4']}
    re_col = {1: 'Bus_ID', 2: "Month_pur", 3: "Year_pur", 4: "Month_1st", 5: "Year_1st", 6: "Odo_1st",
              7: "Month_2nd", 8: "Year_2nd", 9: "Odo_2nd", 10: "Month_begin", 11: "Year_begin"}


    dict_df = dict()
    for keys in dict_data:
        r = dict_data[keys][0]
        c = dict_data[keys][1]
        f_raw = open('data/original_data/' + keys + '.asc').read()
        f_col = f_raw.split('\n')
        df = pd.DataFrame()
        for j in range(0, c):
            for k in range(j*r, (j+1)*r):
                df.loc[(k-j*r)+1, j+1] = float(f_col[k])
        df = df.transpose()
        df = df.rename(columns=re_col)
        df['Bus_ID'] = df['Bus_ID'].astype(int)
        df = df.reset_index()
        df = df.drop(df.columns[[0]], axis=1)
        dict_df[dict_data[keys][2]] = df
        os.makedirs('data/pkl/group_data', exist_ok=True)
        df.to_pickle('data/pkl/group_data/' + dict_data[keys][2] + '.pkl')
