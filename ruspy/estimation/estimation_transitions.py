"""
This module contains the functions necessary for the estimation process of transition
probabilities.
"""
import numpy as np
from math import log
import numba


def estimate_transitions(df, repl_4=True):
    """
    The sub function for estimating the transition probabilities. This function
    manages the estimation process of the transition probaiblities and calls the
    necessary subfunctions.

    :param df: A pandas dataframe, which contains for each observation the Bus ID,
    the current state of the bus, the current period and the decision made in this
    period.
    :param repl_4: Auxiliary variable for the convention of the replacement increase.
    :return: The optimization result of the transition probabilities estimation as a
             dictionary.
    """
    transition_count = [0]
    result_transitions = dict()
    num_bus = len(df["Bus_ID"].unique())
    num_periods = int(df.shape[0] / num_bus)
    states = df["state"].values.reshape(num_bus, num_periods)
    decisions = df["decision"].values.reshape(num_bus, num_periods)
    space_state = states.max() + 1
    state_count = np.zeros(shape=(space_state, space_state), dtype=int)
    if repl_4:
        count_func = count_transitions

    else:
        count_func = count_transitions_alt

    transition_count, state_count = count_func(
        transition_count, state_count, num_bus, num_periods, states, decisions
    )
    result_transitions["state_count"] = state_count
    trans_probs = np.array(transition_count) / sum(transition_count)
    ll = loglike(trans_probs, transition_count)
    result_transitions.update(
        {"x": trans_probs, "fun": ll, "trans_count": transition_count}
    )
    return result_transitions


@numba.jit(nopython=True)
def count_transitions(
    transition_count, state_count, num_bus, num_periods, states, decisions
):
    """
    This function counts how often the buses increased their state by 0, by 1 etc.

    :param transition_count: A list containing only one integer zero.
    :param num_bus:          The number of buses in the samples.
    :type num_bus:           int
    :param num_periods:      The number of periods the buses drove.
    :type num_periods:       int
    :param states:           A two dimensional numpy array containing for each bus in
                             each period the state as an integer.
    :param decisions:        A two dimensional numpy array containing for each bus in
                             each period the decision as an integer.

    :return: A list with the highest increase as maximal index and the increase
             counts as entries.
    """

    for bus in range(num_bus):
        for period in range(num_periods - 1):
            if decisions[bus, period] == 0:
                increase = states[bus, period + 1] - states[bus, period]
                state_count[states[bus, period], states[bus, period + 1]] += 1
            else:
                increase = 1
                state_count[0, increase] += 1
            if increase >= len(transition_count):
                transition_count_new = [0] * (increase + 1)
                for i in range(len(transition_count)):
                    transition_count_new[i] = transition_count[i]
                transition_count = transition_count_new
            transition_count[increase] += 1
    return transition_count, state_count


@numba.jit(nopython=True)
def count_transitions_alt(
    transition_count, state_count, num_bus, num_periods, states, decisions
):
    """
    This function counts how often the buses increased their state by 0, by 1 etc.

    :param transition_count: A list containing only one integer zero.
    :param state_count:      A list containing the observations per state.
    :param num_bus:          The number of buses in the samples.
    :type num_bus:           int
    :param num_periods:      The number of periods the buses drove.
    :type num_periods:       int
    :param states:           A two dimensional numpy array containing for each bus in
                             each period the state as an integer.
    :param decisions:        A two dimensional numpy array containing for each bus in
                             each period the decision as an integer.

    :return: A list with the highest increase as maximal index and the increase
             counts as entries.
    """

    for bus in range(num_bus):
        for period in range(num_periods - 1):
            if decisions[bus, period] == 0:
                increase = states[bus, period + 1] - states[bus, period]
                state_count[states[bus, period], states[bus, period + 1]] += 1
            else:
                increase = states[bus, period + 1]
                state_count[0, increase] += 1
            if increase >= len(transition_count):
                transition_count_new = [0] * (increase + 1)
                for i in range(len(transition_count)):
                    transition_count_new[i] = transition_count[i]
                transition_count = transition_count_new
            transition_count[increase] += 1
    return transition_count, state_count


@numba.jit(nopython=True)
def loglike(trans_probs, transition_count):
    """
    The loglikelihood function for estimating the transition probabilities.

    :param trans_probs:      A numpy array containing transition probabilities.
    :param transition_count: A list with the highest state increase as maximal index
                             and the increase counts as entries.

    :return: The negative loglikelihood value for minimizing the second liklihood
             function.
    """
    ll = 0
    for i, p in enumerate(trans_probs):
        ll = ll + transition_count[i] * log(p)
    return -ll
