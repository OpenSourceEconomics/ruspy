import yaml
import os
import pandas as pd
from ruspy.data.data_reading import data_reading
from ruspy.data.data_processing import data_processing
from ruspy.estimation.estimation import estimate
from ruspy.data.data_location import get_data_storage

data_path = get_data_storage()
with open('init_replication.yml') as y:
    init_dict = yaml.load(y)

group = init_dict['replication']['groups']
binsize = str(init_dict['replication']['binsize'])

if not os.path.isfile(data_path + '/pkl/group_data/' + group + '_' + binsize + '.pkl'):
    data_reading()

if not os.path.isfile(data_path + '/pkl/replication_data/rep_' + group + '_' +
                      binsize + '.pkl'):
    data = data_processing(init_dict['replication'])
else:
    data = pd.read_pickle(data_path + '/pkl/replication_data/rep_' + group + '_' +
                          binsize + '.pkl')

result_transitions, result_fixp = estimate(init_dict['replication'], data)
print(result_transitions, result_fixp)
