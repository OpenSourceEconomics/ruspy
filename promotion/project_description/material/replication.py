import yaml
import os
import pandas as pd
from ruspy.data.data_reading import data_reading
from ruspy.data.data_processing import data_processing
from ruspy.estimation.estimation import estimate
from ruspy.data.data_location import get_data_storage


def get_table(init_dict):
    """ This function produces a latex table with replication results for group 4 with different statesizes.

    :param init_dict: A dictionary containing a section with the group name, the binsize, discount factor beta,
    the statesize for the NXFP and the type of cost function.
    :type init_dict: dictionary
    :return: A txt file for the use in the latex.
    """
    result_trans_5000, result_fixp_5000 = get_repl_result(init_dict['replication'])
    init_dict['replication']['binsize'] = 2571
    init_dict['replication']['states'] = 175
    result_trans, result_fixp = get_repl_result(init_dict['replication'])
    table = '\\begin{tabular}{lrrrrr} \\toprule States & 90 & 175 \\\\ Binsize & 5000 & 2571 \\\\ $\\theta_0$ & '
    table += str(round(result_trans_5000['x'][0], 4)) + ' & ' + str(round(result_trans['x'][0], 4)) + '\\\\'
    table += '$\\theta_1$ & ' + str(round(result_trans_5000['x'][1], 4)) + ' & ' + str(round(result_trans['x'][1], 4))
    table += ' \\\\ ' + ' $\\theta_2$ & ' + str(round(result_trans_5000['x'][2], 4)) + ' & '
    table += str(round(result_trans['x'][2], 4)) + ' \\\\ $\\theta_3$ &  & ' + str(round(result_trans['x'][3], 4))
    table += ' \\\\ \\bottomrule \\end{tabular}'
    os.makedirs('figures', exist_ok=True)
    f = open('figures/replication.txt', 'w+')
    f.write(table)
    f.close()


def get_repl_result(init_dict):
    """ A function to evaluate the replication result of John Rust's 1987 paper.

    :param init_dict: A dictionary containing the relevant variables for the replication.
    :return: The optimization result of the transition probabilities and cost parameters.
    """
    data_path = get_data_storage()

    group = init_dict['groups']
    binsize = str(init_dict['binsize'])

    if not os.path.isfile(data_path + '/pkl/group_data/' + group + '_' + binsize + '.pkl'):
        data_reading()

    if not os.path.isfile(data_path + '/pkl/replication_data/rep_' + group + '_' + binsize + '.pkl'):
        data = data_processing(init_dict)
    else:
        data = pd.read_pickle(data_path + '/pkl/replication_data/rep_' + group + '_' + binsize + '.pkl')

    result_transitions, result_fixp = estimate(init_dict, data)
    return result_transitions, result_fixp
