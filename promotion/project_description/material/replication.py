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
    init_dict['replication']['binsize'] = 5000
    init_dict['replication']['states'] = 90
    init_dict['replication']['beta'] = 0.9999
    result_trans_5000, result_fixp_5000 = get_repl_result(init_dict['replication'])
    init_dict['replication']['beta'] = 0
    result_trans_5000_0, result_fixp_5000_0 = get_repl_result(init_dict['replication'])
    init_dict['replication']['binsize'] = 2571
    init_dict['replication']['states'] = 175
    result_trans_2571_0, result_fixp_2571_0 = get_repl_result(init_dict['replication'])
    init_dict['replication']['beta'] = 0.999
    result_trans_2571, result_fixp_2571 = get_repl_result(init_dict['replication'])
    table = '\\begin{tabular}{lrrrrr} \\toprule States & 90 & 90 & 175 & 175 \\\\ \\midrule Binsize & 5000 & 5000 & '
    table +='2571 & 2571 \\\\ \\midrule $\\theta_{30}$ & ' + str(round(result_trans_5000['x'][0], 4)) + ' & '
    table += str(round(result_trans_5000_0['x'][0], 4)) + ' & ' + str(round(result_trans_2571['x'][0], 4)) + ' & '
    table += str(round(result_trans_2571_0['x'][0], 4)) + '\\\\' + '$\\theta_{31}$ & '
    table += str(round(result_trans_5000['x'][1], 4)) + ' & ' + str(round(result_trans_5000_0['x'][1], 4)) + ' & '
    table += str(round(result_trans_2571['x'][1], 4)) + ' & ' + str(round(result_trans_2571_0['x'][1], 4)) + ' \\\\ '
    table += '$\\theta_{32}$ & ' + str(round(result_trans_5000['x'][2], 4)) + ' & '
    table += str(round(result_trans_5000_0['x'][2], 4)) + ' & ' + str(round(result_trans_2571['x'][2], 4)) + ' & '
    table += str(round(result_trans_2571_0['x'][2], 4)) + ' \\\\ ' + '$\\theta_{33}$ &  &  & '
    table += str(round(result_trans_2571['x'][3], 4)) + ' & ' + str(round(result_trans_2571_0['x'][3], 4)) + ' \\\\ '
    table += 'RC & ' + str(round(result_fixp_5000['x'][0], 4)) + ' & ' + str(round(result_fixp_5000_0['x'][0], 4))
    table += ' & ' + str(round(result_fixp_2571['x'][0], 4)) + ' & ' + str(round(result_fixp_2571_0['x'][0], 4))
    table += ' \\\\ $\\theta_{11}$ & ' + str(round(result_fixp_5000['x'][1], 4)) + ' & '
    table += str(round(result_fixp_5000_0['x'][1], 4)) + ' & ' + str(round(result_fixp_2571['x'][1], 4)) + ' & '
    table += str(round(result_fixp_2571_0['x'][1], 4)) + ' \\\\ LL & '
    table += str(round(result_trans_5000['fun'] + result_fixp_5000['fun'], 4)) + ' & '
    table += str(round(result_trans_5000_0['fun'] + result_fixp_5000_0['fun'], 4)) + ' & '
    table += str(round(result_trans_2571['fun'] + result_fixp_2571['fun'], 4)) + ' & '
    table += str(round(result_trans_2571_0['fun'] + result_fixp_2571_0['fun'], 4)) + ' \\\\ \\midrule $\\beta$ & 0.9999'
    table += ' & 0 & 0.9999 & 0 \\\\ \\bottomrule \\end{tabular}'
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
