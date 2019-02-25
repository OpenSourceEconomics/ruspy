import yaml
import pickle
from ruspy.simulation.simulation import simulate


with open('init.yml') as y:
    init_dict = yaml.load(y)
pickle.dump(simulate(init_dict['simulation']), open('sim_file.pkl', 'wb'))