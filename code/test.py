import yaml

from data_processing import data_processing
from estimation import estimate
with open('init.yml') as y:
    init_dict = yaml.load(y)

estimate(init_dict, data_processing(init_dict))