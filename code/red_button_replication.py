import yaml
import os
from data.data_reading import data_reading
from data.data_processing import data_processing
from estimation.estimation import estimate


with open('init.yml') as y:
    init_dict = yaml.load(y)

if not os.path.isfile('data/pkl/group_data/Group1.pkl'):
    data_reading()

data = data_processing(init_dict)
estimate(init_dict, data)
