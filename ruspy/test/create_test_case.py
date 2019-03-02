import yaml
import pickle
from ruspy.simulation.simulation import simulate

with open('test.ruspy.yml') as y:
    init_dict = yaml.load(y)

df, unobs, ut = simulate(init_dict['simulation'])

f = open('resources/linear_5_agents.pkl', 'wb')
pickle.dump((init_dict, df, unobs, ut), f)