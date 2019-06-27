import yaml
import numpy as np
import matplotlib.pyplot as plt
from ruspy.estimation.estimation_cost_parameters import create_transition_matrix
from ruspy.estimation.estimation_cost_parameters import cost_func
from ruspy.estimation.estimation_cost_parameters import lin_cost
from ruspy.estimation.estimation_cost_parameters import calc_fixp
from ruspy.simulation.simulation import simulate
from ruspy.plotting.value_zero import discount_utility
from ruspy.simulation.robust_sim import get_worst_trans


with open("init_long.yml") as y:
    init_dict = yaml.safe_load(y)

beta = init_dict["simulation"]["beta"]
init_dict["simulation"]["roh"] = 2
init_dict["simulation"]["states"] = 90
worst_trans = get_worst_trans(init_dict["simulation"])
init_dict["simulation"]["real trans"] = worst_trans

df, unobs, utilities, num_states = simulate(init_dict["simulation"])


costs = cost_func(num_states, lin_cost, init_dict["simulation"]["params"])

num_buses = init_dict["simulation"]["buses"]
num_periods = init_dict["simulation"]["periods"]
gridsize = init_dict["plot"]["gridsize"]
num_points = int(num_periods / gridsize)

real_trans_probs = np.array(init_dict["simulation"]["real trans"])
real_trans_mat = create_transition_matrix(num_states, real_trans_probs)
ev_real = calc_fixp(num_states, real_trans_mat, costs, beta)

known_trans_probs = np.array(init_dict["simulation"]["known trans"])
known_trans_mat = create_transition_matrix(num_states, known_trans_probs)
ev_known = calc_fixp(num_states, known_trans_mat, costs, beta)


v_calc = 0
for i in range(num_buses):
    v_calc = v_calc + unobs[i, 0, 0] + ev_known[0]
v_calc = v_calc / num_buses
v_exp_known = list(np.full(num_points, v_calc))

v_calc = 0
for i in range(num_buses):
    v_calc = v_calc + unobs[i, 0, 0] + ev_real[0]
v_calc = v_calc / num_buses
v_exp_real = list(np.full(num_points, v_calc))

v_start = np.zeros(num_points)
v_disc = list(
    discount_utility(v_start, num_buses, gridsize, num_periods, utilities, beta)
)
np.savetxt("figure_2_1_prelim_long.txt", np.array(v_disc))
np.savetxt("worst_case_figure_2_long.txt", worst_trans)

periods = list(np.arange(0, num_periods, gridsize))

ax = plt.figure(figsize=(14, 6))

ax1 = ax.add_subplot(111)

ax1.set_ylim([0, 1.3 * v_disc[-1]])

ax1.set_ylabel(r"Value at time 0    ")
ax1.set_xlabel(r"Periods")

l1 = ax1.plot(periods, v_disc, color="blue")
l2 = ax1.plot(periods, v_exp_known, color="green")
l3 = ax1.plot(periods, v_exp_real, color="red")

plt.tight_layout()

plt.savefig("figure_2_1_long.png", dpi=300)
