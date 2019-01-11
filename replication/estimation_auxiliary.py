import numpy as np
from math import log
import scipy.optimize as opt


# The first part are functions for estimating the transition probabilities.
def estimate_transitions_5000(df):
    """
    A function to estimate the transition probabilities.
    :param df: A DataFrame with columns Bus_ID, state and decision containing the observations.
    :return: A dictionary with the results of the observation.
    """
    transition_list = count_transitions_5000(df)
    result_transitions = opt.minimize(loglike, args=transition_list, x0=[0.3, 0.5, 0.01],
                                      bounds=[(1e-6, 1), (1e-6, 1), (1e-6, 1)],
                                      constraints=({'type': 'eq', "fun": apply_p_constraint}))
    return result_transitions


def count_transitions_5000(df):
    """
    A function to count the transitions.
    :param df: A DataFrame with columns Bus_ID, state and decision containing the observations.
    :return: A list with the count state increases by 0,1 or 2.
    """
    n = 0
    e = 0
    z = 0
    for i in df.Bus_ID.unique():
        df2 = df[df['Bus_ID'] == i].reset_index()
        for j in df2.index.values[::-1]:
            if j > 0:
                if df2.iloc[j - 1]['decision'] == 0:
                    if df2.iloc[j]['state'] - df2.iloc[j - 1]['state'] == 0:
                        n = n + 1
                    elif df2.iloc[j]['state'] - df2.iloc[j - 1]['state'] == 1:
                        e = e + 1
                    elif df2.iloc[j]['state'] - df2.iloc[j - 1]['state'] == 2:
                        z = z + 1
                elif df2.iloc[j - 1]['decision'] == 1:
                    e = e + 1  # Weird convention, but gives the results of the paper.
    return [n, e, z]


def apply_p_constraint(inputs):
    """
    A constraint which checks the sum of the transition probabilities.
    :param inputs: A array of transition probabilities.
    :return: Should return 0.
    """
    total = 1 - np.sum(inputs)
    return total


def loglike(params, transition_list):
    """
    The loglikelihood function for estimating the transition probabilities.
    :param params: An array of choice probabilities.
    :param transition_list:
    :return: The negative loglikelihood function for minimizing.
    """
    ll = 0
    for i in range(len(params)):
        ll = ll + transition_list[i] * log(params[i])
    return -ll


# The second part contains functions to maximize the likelihood of the Zurcher's decision probabilities.
def create_transition_matrix(num_states, trans_prob):
    """
    This function creates a transition matrix.
    :param num_states: The size of the state space.
    :param trans_prob: The transition probabilities for an increase of the state.
    :return: A Markov transition matrix.
    """
    trans_mat = np.zeros((num_states, num_states))
    for i in range(num_states):  # Loop over all states.
        for j, p in enumerate(trans_prob):  # Loop over the possible increases.
            if i + j < num_states - 1:
                trans_mat[i + j][i] = p
            elif i + j == num_states - 1:
                trans_mat[num_states - 1][i] = trans_prob[j:].sum()  # The probability to reach the last state.
            else:
                pass
    return trans_mat


def create_state_matrix(exog, num_states, num_obs):
    """
    This function constructs a auxiliary matrix for the likelihood.
    :param exog: The observation data on the states.
    :param num_states: The size of the state space s.
    :param num_obs: The total number of observations n.
    :return:  A nxs matrix containing TRUE in the row for each observation, if the bus was in that state.
    """
    state_mat = np.array([[exog[i] == s for i in range(num_obs)]
                          for s in range(num_states)])
    return state_mat


def loglike_opt_rule(params, num_states, trans_mat, state_mat, decision_mat, beta):
    """
    This is the logliklihood function for the estimation of the cost parameters.
    :param params: The cost parameters for replacing or maintaining the bus engine.
    :param num_states: The size of the state space s.
    :param trans_mat: The Markov transition matrix.
    :param state_mat: A nxs matrix containing TRUE in the row for each observation, if the bus was in that state.
    :param decision_mat: A nx2 vector containing 1 in the first row for maintaining and 1 in the second for replacement.
    :param beta: The discount factor.
    :return: The negative loglikelihood function for minimizing
    """
    ev = calc_fixp(num_states, trans_mat, lin_cost, params, beta)
    p_choice = choice_prob(ev, params, beta)
    ll_prob = np.log(np.dot(p_choice.T, state_mat))
    return -np.sum(decision_mat * ll_prob)


def lin_cost(s, params):
    """
    This function describes a linear cost function, which Rust concludes is the most realistic maintenance function.
    :param s: The state s.
    :param params: The slope of the cost function.
    :return: The maintenance cost for state s.
    """
    states = np.arange(s)
    return states * 0.001 * params[0]


def myopic_costs(s, maint_func, params):
    """
    This function calculates a vector containing the costs for the two alternatives, without recognizing the future.
    :param s: The size of the state space.
    :param maint_func: The name of the maintenance function.
    :param params: The cost parameters for replacing or maintaining the bus engine.
    :return: A vector containing the costs of a non-forward looking agent.
    """
    rc = params[0]
    maint_cost = maint_func(s, params[1:])
    repl_cost = np.full(maint_cost.shape, rc + maint_cost[0])
    return np.vstack((maint_cost, repl_cost)).T


def choice_prob(cost_array, params, beta):
    """
    This function calculates the choice probabilities to maintain or replace for each state.
    :param cost_array: An array containing the expected future value of maintaining or replacing the bus engine.
    :param params: The cost parameters for replacing or maintaining the bus engine.
    :param beta: The discount factor.
    :return: A array containing the choice probabilities for each state.
    """
    s = cost_array.shape[0]
    costs = myopic_costs(s, lin_cost, params)
    util_main = np.exp(beta * cost_array - costs[:, 0])  # Utility to maintain the bus
    util_repl = [np.exp(beta * cost_array[0] - costs[0][0] - costs[0][1]) for state in
                 range(0, s)]  # Utility to replace the bus
    util = np.vstack((util_main, util_repl)).T
    pchoice = util / (np.sum(util, axis=1).reshape(s, -1))
    return pchoice


def calc_fixp(num_states, trans_mat, maint_func, params, beta, threshold=1e-6):
    """
    The function to calculate the nested fix point.
    :param num_states: The size of the state space.
    :param trans_mat: The Markov transition matrix.
    :param maint_func: The name of the maintenance function.
    :param params: The cost parameters for replacing or maintaining the bus engine.
    :param beta: The discount factor.
    :param threshold: A threshold for the convergence.
    :return: A vector with the fix point.
    """
    k = 0
    ev = np.zeros((num_states, 1))
    costs = myopic_costs(num_states, maint_func, params)  # The myopic costs are the starting point.
    ev_new = np.dot(trans_mat.T, np.log(np.sum(np.exp(-costs), axis=1)))
    while abs(ev_new - ev).max() > threshold:
        ev = ev_new
        main_cost = (beta * ev - costs[:, 0])
        repl_cost = [(beta * ev[0] - costs[0][1] - costs[0][0]) for state in range(0, num_states)]
        ev_ = np.vstack((main_cost, repl_cost)).T
        ev_new = np.dot(trans_mat.T, np.log(np.sum(np.exp(ev_), axis=1)))
        k = k + 1
        if k == 1000:  # Maximum number of iterations.
            break
    return ev_new
