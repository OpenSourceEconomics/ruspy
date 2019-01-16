import numpy as np
from estimation_auxiliary import calc_fixp
from estimation_auxiliary import create_transition_matrix
from estimation_auxiliary import lin_cost
from estimation_auxiliary import myopic_costs


def decide(s, ev, costs, unobs, beta):
    """
    Takes a state s and calculates the according decision probability
    """
    if (- costs[s, 0] + unobs[0] + beta * ev[s]) > (- costs[0, 0] - costs[0, 1]+ unobs[1] + beta * ev[0]):
        decision = 0
        utility = - costs[s, 0] + unobs[0]
    else:
        decision = 1
        utility = - costs[0, 0] - costs[0, 1] + unobs[1]
    return decision, utility



def transition(old_state, decision, trans_prob):
    increment = np.random.choice(np.arange(len(trans_prob)), p=trans_prob)
    if decision == 0:
        return old_state + increment
    else:
        return increment


def simulate_strategy(known_trans, real_trans, num_buses, num_periods, num_states, params, beta, unobs, maint_func):
    known_trans_mat = create_transition_matrix(num_states, known_trans)
    ev = calc_fixp(num_states, known_trans_mat, lin_cost, params, beta)
    costs = myopic_costs(num_states, maint_func, params)
    states = np.zeros((num_buses, 1), dtype=int)
    decisions = np.empty((num_buses, 0), dtype=int)
    utilities = np.empty((num_buses, 0), dtype=int)
    for i in range(0, num_periods):
        print(i)
        if i > 0:
            states = np.append(states, new_states.reshape(num_buses, 1), axis=1)
        new_decisions = np.empty((0,0), dtype=int)
        new_utilities = np.empty((0,0), dtype=int)
        new_states = np.empty((0, 0), dtype=int)
        for bus in range(num_buses):
            decision, utility = decide(states[bus, -1], ev, costs, unobs[i , :], beta)
            new_decisions = np.append(new_decisions, decision)
            new_utilities = np.append(new_utilities, utility)
            new_states = np.append(new_states, transition(states[bus, -1], decision, real_trans))
        decisions = np.append(decisions, new_decisions.reshape(num_buses, 1), axis=1)
        utilities = np.append(utilities, new_utilities.reshape(num_buses, 1), axis=1)
    return states, decisions, utilities


