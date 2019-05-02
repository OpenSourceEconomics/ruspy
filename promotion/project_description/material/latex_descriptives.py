import pandas as pd
import os
from ruspy.data.data_reading import data_reading
from ruspy.data.data_location import get_data_storage


def create_desc():
    data_path = get_data_storage()
    dict_df = {}
    if os.path.isdir(data_path + "/pkl/group_data/"):
        list_pkl = os.listdir(data_path + "/pkl/group_data/")
        if len(list_pkl) != 8:
            data_reading()
    else:
        data_reading()
        list_pkl = os.listdir(data_path + "/pkl/group_data/")
    for filename in list_pkl:
        dict_df[filename[0:7]] = pd.read_pickle(
            data_path + "/pkl/group_data/" + filename
        )

    df = pd.DataFrame()
    df_ges = pd.DataFrame()
    for j, i in enumerate(sorted(dict_df.keys())):
        group_name = "Group " + i[-1]
        df2 = dict_df[i][["Odo_1st"]][dict_df[i]["Odo_1st"] > 0]
        df2 = df2.rename(columns={"Odo_1st": group_name})
        df3 = dict_df[i][["Odo_2nd"]].sub(dict_df[i]["Odo_1st"], axis=0)[
            dict_df[i]["Odo_2nd"] > 0
        ]
        df3 = df3.rename(columns={"Odo_2nd": group_name})
        df3 = df3.set_index(df3.index.astype(str) + "_2")
        df4 = pd.concat([df2, df3])
        if j == 0:
            df = df4.describe()
            df_ges["Full sample"] = df4[group_name]
        else:
            df = pd.concat([df, df4.describe()], axis=1)
            df_ges = pd.concat(
                [df_ges, df4.rename(columns={group_name: "Full sample"})], axis=0
            )
    df = pd.concat([df, df_ges.describe()], axis=1)
    df = df.transpose()
    df = df.drop(df.columns[[4, 5, 6]], axis=1)
    df = df[["max", "min", "mean", "std", "count"]].fillna(0).astype(int)
    df.rename(
        columns={
            "max": "Max",
            "min": "Min",
            "mean": "Mean",
            "std": "Std. Dev.",
            "count": "NumObs",
        },
        inplace=True,
    )
    os.makedirs("figures", exist_ok=True)
    f = open("figures/descr_2a.txt", "w+")
    f.write("Milage at Replacement \\\\" + df.to_latex())
    f.close()

    """ The following code produces partly table 2 b of Rust's paper"""
    df = pd.DataFrame()
    df_ges = pd.DataFrame()
    for i in sorted(dict_df.keys()):
        group_name = "Group " + i[-1]
        df2 = dict_df[i][[dict_df[i].columns.values[-1]]][dict_df[i]["Odo_1st"] == 0]
        df2 = df2.rename(columns={df2.columns.values[0]: group_name})
        if j == 0:
            df = df2.describe()
            df_ges["Full sample"] = df2[group_name]
        else:
            df = pd.concat([df, df2.describe()], axis=1)
            df_ges = pd.concat(
                [df_ges, df2.rename(columns={group_name: "Full sample"})], axis=0
            )
    df = pd.concat([df, df_ges.describe()], axis=1)
    df = df.transpose()
    df = df.drop(df.columns[[4, 5, 6]], axis=1)
    df = df[["max", "min", "mean", "std", "count"]].fillna(0).astype(int)
    df.rename(
        columns={
            "max": "Max",
            "min": "Min",
            "mean": "Mean",
            "std": "Std. Dev.",
            "count": "NumObs",
        },
        inplace=True,
    )
    f = open("figures/descr_2b.txt", "w+")
    f.write("Milage at May 1, 1985 \\\\" + df.to_latex())
    f.close()


if __name__ == "__main__":
    create_desc()
