import yaml
import os
import pandas as pd
from data.data_reading import data_reading
from data.data_processing import data_processing
from ruspy.estimation.estimation import estimate


with open('init_replication.yml') as y:
    init_dict = yaml.load(y)

group = init_dict['replication']['group']

if not os.path.isfile('data/pkl/group_data/' + group + '.pkl'):
    data_reading()

if not os.path.isfile('data/pkl/replication_data/rep_' + group + '.pkl'):
    data = data_processing(init_dict['replication'])
else:
    data = pd.read_pickle('data/pkl/replication_data/rep_group_4.pkl')

estimate(init_dict['replication'], data)
