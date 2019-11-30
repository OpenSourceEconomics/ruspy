import pickle as pkl

from ruspy.estimation.estimation import estimate
from ruspy.ruspy_config import TEST_RESOURCES_DIR

TEST_FOLDER = TEST_RESOURCES_DIR + "replication_test/"
init_dict = {
    "groups": "group_4",
    "binsize": 5000,
    "beta": 0.9999,
    "states": 90,
    "maint_cost_func": "cubic",
}
df = pkl.load(open(TEST_FOLDER + "group_4.pkl", "rb"))
result_trans, result_fixp = estimate(init_dict, df)
