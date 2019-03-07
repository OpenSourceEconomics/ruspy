import numpy as np
from math import log
import scipy.optimize as opt
import numba


# The first part are functions for estimating the transition probabilities.
def estimate_transitions_5000(df):
    """The sub function for estimating the transition probabilities.

    :param df:
    :return:
    """
    transition_count = [0]
    num_bus = len(df['Bus_ID'].unique())
    num_periods = int(df.shape[0] / num_bus)
    states = df['state'].values.reshape(num_bus, num_periods)
    decisions = df['decision'].values.reshape(num_bus, num_periods)
    transition_count = count_transitions_5000(transition_count, num_bus, num_periods, states, decisions)
    dim = len(transition_count)
    x_0 = np.full(dim, 0.1)
    result_transitions = opt.minimize(loglike, args=transition_count, x0=x_0,
                                      bounds=[(1e-6, 1)] * dim,
                                      constraints=({'type': 'eq', "fun": apply_p_constraint}))
    return result_transitions


@numba.jit(nopython=True)
def count_transitions_5000(transition_count, num_bus, num_periods, states, decisions):
    """

    :param df:
    :return:
    """

    for bus in range(num_bus):
        for period in range(num_periods - 1):
            if decisions[bus, period] == 0:
                increase = states[bus, period + 1] - states[bus, period]
            else:
                increase = 1
            if increase >= len(transition_count):
                transition_count_new = [0] * (increase + 1)
                for i in range(len(transition_count)):
                    transition_count_new[i] = transition_count[i]
                transition_count = transition_count_new
            transition_count[increase] += 1
    return transition_count


def apply_p_constraint(inputs):
    """A constraint which checks the sum of the transition probabilities.

    :param inputs: A array of transition probabilities.
    :return: Should return 0.
    """
    total = 1 - np.sum(inputs)
    return total


def loglike(params, transition_list):
    """The loglikelihood function for estimating the transition probabilities.

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
    """This function creates a transition matrix.

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
    """This function constructs a auxiliary matrix for the likelihood.

    :param exog: The observation data on the states.
    :param num_states: The size of the state space s.
    :param num_obs: The total number of observations n.
    :return:  A nxs matrix containing TRUE in the row for each observation, if the bus was in that state.
    """
    state_mat = np.full((num_states, num_obs), False, dtype=bool)
    for i, value in enumerate(exog):
        state_mat[value, i] = True
    return state_mat


def loglike_opt_rule(params, maint_func, num_states, trans_mat, state_mat, decision_mat, beta):
    """This is the logliklihood function for the estimation of the cost parameters.

    :param params: The cost parameters for replacing or maintaining the bus engine.
    :param num_states: The size of the state space s.
    :param trans_mat: The Markov transition matrix.
    :param state_mat: A nxs matrix containing TRUE in the row for each observation, if the bus was in that state.
    :param decision_mat: A nx2 vector containing 1 in the first row for maintaining and 1 in the second for replacement.
    :param beta: The discount factor.
    :return: The negative loglikelihood function for minimizing
    """
    costs = myopic_costs(num_states, maint_func, params)
    ev = calc_fixp(num_states, trans_mat, costs, beta)
    p_choice = choice_prob(ev, params, beta)
    ll_prob = np.log(np.dot(p_choice.T, state_mat))
    return -np.sum(decision_mat * ll_prob)


def lin_cost(s, params):
    """This function describes a linear cost function, which Rust concludes is the most realistic maintenance function.

    :param s: The number of states.
    :param params: The slope of the cost function.
    :return: The maintenance cost for state s.
    """
    states = np.arange(s)
    return states * 0.001 * params[0]


def myopic_costs(s, maint_func, params):
    """This function calculates a vector containing the costs for the two alternatives, without recognizing the future.

    :param s: The size of the state space.
    :param maint_func: The name of the maintenance function.
    :param params: The cost parameters for replacing or maintaining the bus engine.
    :return: A vector containing the costs of a non-forward looking agent.
    """
    rc = params[0]
    maint_cost = maint_func(s, params[1:])
    repl_cost = np.full(maint_cost.shape, rc + maint_cost[0])
    return np.vstack((maint_cost, repl_cost)).T


def choice_prob(ev, params, beta):
    """This function calculates the choice probabilities to maintain or replace for each state.

    :param ev: An array containing the expected future value of maintaining or replacing the bus engine.
    :param params: The cost parameters for replacing or maintaining the bus engine.
    :param beta: The discount factor.
    :return: A array containing the choice probabilities for each state.
    """
    s = ev.shape[0]
    costs = myopic_costs(s, lin_cost, params)
    util_main = beta * ev - costs[:, 0]  # Utility to maintain the bus
    util_repl = np.full(util_main.shape, beta * ev[0] - costs[0, 0] - costs[0, 1]) # Utility to replace the bus
    util = np.vstack((util_main, util_repl)).T
    util_min = ev[0]
    util = util - util_min
    pchoice = np.exp(util) / (np.sum(np.exp(util), axis=1).reshape(s, -1))
    return pchoice


@numba.jit(nopython=True)
def calc_fixp(num_states, trans_mat, costs, beta, threshold=1e-8, max_it=1000000):
    """The function to calculate the nested fix point.

    :param num_states: The size of the state space.
    :param trans_mat: The Markov transition matrix.
    :param costs: The cost parameters for replacing or maintaining the bus engine.
    :param beta: The discount factor.
    :param threshold: A threshold for the convergence.
    :param max_it: Maximum number of iterations.
    :return: A vector with the fix point.
    """
    ev = np.zeros(num_states)
    ev_new = np.dot(trans_mat.T, np.log(np.sum(np.exp(-costs), axis=1)))
    while (np.max(np.abs(ev_new - ev)) > threshold) & (max_it != 0):
        ev = ev_new
        maint_cost = beta * ev - costs[:, 0]
        repl_cost = beta * ev[0] - costs[0, 1] - costs[0, 0]
        ev_min = maint_cost[0]
        log_sum = ev_min + np.log(np.exp(maint_cost - ev_min) + np.exp(repl_cost - ev_min))
        ev_new = np.dot(trans_mat.T, log_sum)
        max_it += -1
    return ev_new
