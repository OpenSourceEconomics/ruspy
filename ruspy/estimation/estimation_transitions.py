"""
This module contains the functions necessary for the estimation process of transition
probabilities.
"""
import numba
import numpy as np


def estimate_transitions(df, repl_4=False):
    """
    The sub function for estimating the transition probabilities. This function
    manages the estimation process of the transition probaiblities and calls the
    necessary subfunctions.

    :param df: A pandas dataframe, which contains for each observation the Bus ID,
    the current state of the bus, the current period and the decision made in this
    period.

    :param repl_4: Auxiliary variable indicating the transition estimation as in Rust
        (1987). The treatment of the engine replacement decision for the state
        transitions is hardcoded below.

    :return: The optimization result of the transition probabilities estimation as a
             dictionary.
    """
    result_transitions = {}
    num_bus = len(df["Bus_ID"].unique())
    num_periods = int(df.shape[0] / num_bus)
    states = df["state"].values.reshape(num_bus, num_periods)
    decisions = df["decision"].values.reshape(num_bus, num_periods)
    space_state = states.max() + 1
    state_count = np.zeros(shape=(space_state, space_state), dtype=int)
    increases = np.zeros(shape=(num_bus, num_periods - 1), dtype=int)

    increases, state_count = create_increases(
        increases, state_count, num_bus, num_periods, states, decisions, repl_4
    )
    result_transitions["state_count"] = state_count
    transition_count = np.bincount(increases.flatten())
    trans_probs = np.array(transition_count) / np.sum(transition_count)
    ll = loglike(trans_probs, transition_count)
    result_transitions.update(
        {"x": trans_probs, "fun": ll, "trans_count": transition_count}
    )
    return result_transitions


@numba.jit(nopython=True)
def create_increases(
    increases, state_count, num_bus, num_periods, states, decisions, repl_4=False
):
    """
    This function counts how often the buses increased their state by 0, by 1 etc.

    :param increases:        An array containing zeros and be filled with the increases.
    :param state_count:      An array containing the number of observations per state.
    :param num_bus:          The number of buses in the samples.
    :type num_bus:           int
    :param num_periods:      The number of periods the buses drove.
    :type num_periods:       int
    :param states:           A two dimensional numpy array containing for each bus in
                             each period the state as an integer.
    :param decisions:        A two dimensional numpy array containing for each bus in
                             each period the decision as an integer.

    :param repl_4: Auxiliary variable indicating the transition estimation as in Rust
        (1987). The treatment of the engine replacement decision for the state
        transitions is hardcoded below.

    :return: A list with the highest increase as maximal index and the increase
             counts as entries.
    """

    for bus in range(num_bus):
        for period in range(num_periods - 1):
            if decisions[bus, period] == 0:
                increases[bus, period] = states[bus, period + 1] - states[bus, period]
                state_count[states[bus, period], states[bus, period + 1]] += 1
            else:
                if repl_4:
                    increases[bus, period] = 1  # This is the setting from Rust (1987)
                else:
                    increases[bus, period] = states[bus, period + 1]
                state_count[0, increases[bus, period]] += 1
    return increases, state_count


def loglike(trans_probs, transition_count):
    """
    The loglikelihood function for estimating the transition probabilities.

    :param trans_probs:      A numpy array containing transition probabilities.
    :param transition_count: A list with the highest state increase as maximal index
                             and the increase counts as entries.

    :return: The negative loglikelihood value for minimizing the second liklihood
             function.
    """
    ll = np.sum(np.multiply(transition_count, np.log(trans_probs)))
    return -ll


@numba.jit(nopython=True)
def create_transition_matrix(num_states, trans_prob):
    """
    This function creates a markov transition matrix. By the assumptions of the
    underlying model, only the diagonal and elements to the right can have a non-zero
    entry.

    :param num_states:  The size of the state space s.
    :type num_states:   int
    :param trans_prob:  A numpy array containing the transition probabilities for a
                        state increase.

    :return: A two dimensional numpy array containing a s x s markov transition matrix.
    """
    trans_mat = np.zeros((num_states, num_states))
    for i in range(num_states):  # Loop over all states.
        for j, p in enumerate(trans_prob):  # Loop over the possible increases.
            if i + j < num_states - 1:
                trans_mat[i, i + j] = p
            elif i + j == num_states - 1:
                trans_mat[i, num_states - 1] = trans_prob[j:].sum()
            else:
                pass
    return trans_mat
