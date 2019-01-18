# This module creates pickle files from the raw data.

import pandas as pd
import os
import numpy as np

# The following dictionaries contain details on the raw data described by Rust in his documentation.
# This Information is used to create 8 dataframes, each containing one of the Busgroups used in the paper.
# Also the first 11 columns contain bus specific information on purchase,engine replacement etc.
# For further information, see the documnetation.

dict_data = {'g870': [36, 15, 'Group1'], 'rt50': [60, 4, 'Group2'],
             't8h203': [81, 48, 'Group3'], 'a452372': [137, 18, 'Group8'],
             'a452374': [137, 10, 'Group6'], 'a530872': [137, 18, 'Group7'],
             'a530874': [137, 12, 'Group5'], 'a530875': [128, 37, 'Group4']}
re_col = {1: 'Bus_ID', 2: "Month_pur", 3: "Year_pur", 4: "Month_1st", 5: "Year_1st", 6: "Odo_1st",
          7: "Month_2nd", 8: "Year_2nd", 9: "Odo_2nd", 10: "Month_begin", 11: "Year_begin"}

''' 
In the first part of the module, the eight raw data files used by Rust are processed to eight pickle files.
Each containing a DataFrame indexed the same as in the documentation. The files are named according to their group
in the paper. 
'''
dict_df = dict()
for keys in dict_data:
    r = dict_data[keys][0]
    c = dict_data[keys][1]
    f_raw = open('../data/' + keys + '.asc').read()
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
    os.makedirs('../pkl/group_data', exist_ok=True)
    df.to_pickle('../pkl/group_data/' + dict_data[keys][2] + '.pkl')

'''
In the second part eight pickle files are created, which contain the total number of observations for each group.
Therefore every DataFrame row contains the bus identifier, the state variable and the according decision. Also the 
odometer gets set to zero after replacement.
'''
for l in sorted(dict_df):
    repl = dict()
    df = dict_df[l]
    for i in df.index:
        repl[i] = 0
    for j, i in enumerate(df.columns.values[11:]):
        df2 = df[['Bus_ID', i]]
        df2 = df2.assign(decision=0)
        for m in df2.index:
            if repl[m] == 1:
                df2.at[m, i] = df2.iloc[m][i] - df.iloc[m]['Odo_1st']
            if repl[m] == 2:
                df2.at[m, i] = df2.iloc[m][i] - df.iloc[m]['Odo_2nd']
            if i < df.columns.values[-1]:
                if (df.iloc[m][i+1] > df.iloc[m]['Odo_1st']) & (df.iloc[m]['Odo_1st'] != 0) & (repl[m] == 0):
                    df2.at[m, 'decision'] = 1
                    repl[m] = repl[m] + 1
                if (df.iloc[m][i+1] > df.iloc[m]['Odo_2nd']) & (df.iloc[m]['Odo_2nd'] != 0) & (repl[m] == 1):
                    df2.at[m, 'decision'] = 1
                    repl[m] = repl[m] + 1
        df2 = df2.rename(columns={i: 'state'})
        if j == 0:
            df3 = df2
        else:
            df3 = pd.concat([df3, df2])
    df3.reset_index(drop=True, inplace=True)
    num_bus = len(df3['Bus_ID'].unique())
    num_periods = df3.shape[0] / num_bus
    df3['period'] = np.arange(num_periods).repeat(num_bus).astype(int)
    os.makedirs('../pkl/replication_data', exist_ok=True)
    df3.to_pickle('../pkl/replication_data/Rep' + l + '.pkl')
