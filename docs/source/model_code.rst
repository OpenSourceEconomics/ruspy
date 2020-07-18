Model code
==========
This part documents the different functions for the calculation of the model objects
determining the decision of Harold Zurcher. Following Rust (1987), the code does not
estimate the discount factor and it needs to be externally set.

.. _costs:

Observed costs
--------------

The observed costs are saved in :math:`num\_states \times 2` dimensional numpy array.
The first column contains the maintenance and the second the replacement costs for each
state. The function to calculate the observed costs is:

.. currentmodule:: ruspy.model_code.cost_functions

.. autosummary::
    :toctree: _generated/

    calc_obs_costs

The inputs are besides the size of the state space, the type of the maintenance cost
function as well as the cost parameters and the scale. The inputs will be explained in
the following:

.. _maint_func:

-------------------------
Maintenance cost function
-------------------------

So far the code allows for five functional forms. The following table reports the
different functional forms for an arbitrary state :math:`x`. Afterwards I list the APIs
of each function and their derivatives. :math:`states` is the size of the state space.

+-------------+------------------------------------------------------------------------+
| Name        | Functional form                                                        |
+-------------+------------------------------------------------------------------------+
| linear      | :math:`c(x,\theta_1) = \theta_{11} x`                                  |
+-------------+------------------------------------------------------------------------+
| square root | :math:`c(x,\theta_1) = \theta_{11} \sqrt{x}`                           |
+-------------+------------------------------------------------------------------------+
| cubic       | :math:`c(x,\theta_1) = \theta_{11}x+\theta_{12} x**2+\theta_{13} x**3` |
+-------------+------------------------------------------------------------------------+
| hyperbolic  | :math:`c(x,\theta_1) = (\theta_{11} / ((states + 1) - x))`             |
+-------------+------------------------------------------------------------------------+
| quadratic   | :math:`c(x,\theta_1) = (\theta_{11} x +\theta_{12} x**2)`              |
+-------------+------------------------------------------------------------------------+

Linear cost function
x
.. currentmodule:: ruspy.model_code.cost_functions

.. autosummary::
    :toctree: _generated/

    lin_cost
    lin_cost_dev

Square root function

.. currentmodule:: ruspy.model_code.cost_functions

.. autosummary::
    :toctree: _generated/

    sqrt_costs
    sqrt_costs_dev


Cubic cost function

.. currentmodule:: ruspy.model_code.cost_functions

.. autosummary::
    :toctree: _generated/

    cubic_costs
    cubic_costs_dev

Quadratic cost function

.. currentmodule:: ruspy.model_code.cost_functions

.. autosummary::
    :toctree: _generated/

    quadratic_costs
    quadratic_costs_dev

hyperbolic cost function

.. currentmodule:: ruspy.model_code.cost_functions

.. autosummary::
    :toctree: _generated/

    hyperbolic_costs
    hyperbolic_costs_dev


.. _params:

---------------
Cost parameters
---------------
The second in put are the cost parameters, which are stored as a one dimensional
*numpy.array*. At the first position always the replacement cost :math:`RC` is stored.
The next positions are subsequently filled with :math:`\theta_{11}, \theta_{12}, ...`.
The exact number depends on the functional form.

.. _scale:

-----
Scale
-----

The maintenance costs are, due to feasibility of the fixed point algorithm scaled. The
scaling varies across functional forms. The following table contains an overview of the
minimal scale needed for each form:

+---------------+-----------------+
| Cost function | Scale           |
+---------------+-----------------+
| linear        | :math:`10^{-3}` |
+---------------+-----------------+
| square root   | :math:`10^{-2}` |
+---------------+-----------------+
| cubic         | :math:`10^{-8}` |
+---------------+-----------------+
| hyperbolic    | :math:`10^{-1}` |
+---------------+-----------------+
| quadratic     | :math:`10^{-5}` |
+---------------+-----------------+



Fixed Point Algorithm
----------------------------

This part documents the core contribution to research of the Rust (1987) paper, the
Fixed Point Algorithm (FXP). It allows to consequently calculate the log-likelihood
value for each cost parameter and thus, to estimate the model and hence builds
the corner stone of the Nested Fixed Point Algorithm (NFXP).
The computation of the fixed point is managed by:

.. currentmodule:: ruspy.model_code.fix_point_alg

.. autosummary::
    :toctree: _generated/

    calc_fixp


The algorithm is set up as a polyalgorithm combining contraction and Newton-Kantorovich
(Kantorovich, 1948) iterations. It starts by executing contraction iterations, until it
reaches some convergence threshold and then switches to Newton-Kantorovich iterations.
The exact mathematical deviation of the separate steps are very nicely illustrated in
`Rust (2000) <https://editorialexpress.com/jrust/nfxp.pdf>`_. The function of these two
steps are the following in ruspy:

.. currentmodule:: ruspy.model_code.fix_point_alg

.. autosummary::
    :toctree: _generated/

    contraction_iteration
    kantorovich_step


.. _alg_details:

-------------------
Algorithmic details
-------------------

In the following the variable keys are presented, which allow to specify the algorithmic
behavior. The parameters can be grouped into two categories. Switching parameters, which
allow to specify, when the algorithm switches from contraction to Newton-Kantorovich
iterations and general parameters, which let the algorithm stop. So far, there is no
switching back implemented.

**max_cont_steps :** *(int)* The maximum number of contraction iterations before
switching to Newton-Kantorovich iterations. Default is 20.

**switch_tol :** *(float)* If this threshold is reached by contraction iterations, then
the algorithm switches to Newton-Kantorovich iterations. Default is :math:`10^{-3}`.

**max_newt_kant_steps :** *(int)* The maximum number of Newton-Kantorovich iterations
before the algorithm stops. Default is 20.

**threshold :** *(float)* If this threshold is reached by Newton-Kantorovich iterations,
then the algorithm stops. Default is :math:`10^{-12}`.

.. _ev:

-----------------------------
Expected value of maintenance
-----------------------------

In ruspy the expected value of maintenance is stored in a state space sized numpy array.
Thus, the exected value of replacement can be found in the zero entry. It is generally
denoted by *ev*, except in the simulation part of the package where it is denoted by
*ev_known*. This illustrates that the expected value is created by the agent on his
beliefs of the process.


Common model objects
--------------------

Here are some common objects with a short description documented.

.. _trans_mat:

-----------------
Transition matrix
-----------------

The transition matrix for the Markov process are stored in a :math:`num\_states \times
num\_states` dimensional numpy array. As transition in the case of the replacement
corresponds to a transition from state 0, it exists only matrix.


.. _pchoice:

--------------------
Choice probabilities
--------------------

The choice probabilities are stored in a :math:`num\_states \times 2` dimensional numpy
array. So far only choice probabilities, resulting from an unobserved shock with i.i.d.
gumbel distribution are implemented. The multinomial logit formula is herefore
implemented in:

.. currentmodule:: ruspy.model_code.choice_probabilities

.. autosummary::
    :toctree: _generated/

    choice_prob_gumbel


.. _disc_fac:

---------------
Discount factor
---------------

The discount factor, as described in the economic model section, is stored as a float in
ruspy. It needs to be set externally for the simulation as well as for the estimation
process. The key in the dictionary herefore is always *disc_fac*.
