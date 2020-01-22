Model code
==========
This part documents the different functions for the calculation of the model objects
determining the decision of Harold Zurcher. Following Rust (1987), the code does not
estimate the discount factor and it needs to be externally set.

.. _disc_fac:

Discount factor
---------------

.. _costs:

Observed costs
--------------

The observed costs are saved in an The function to calculate the observed costs is:

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
different functional forms for an arbitrary state :math`x`. Afterwards I list the APIs of
each function and their derivatives. :math:`states` is the size of the state space.

+-------------+-------------------------------------------------------------------------+
| Name        | Functional form                                                         |
+-------------+-------------------------------------------------------------------------+
| linear      | :math:`c(x,\theta_1) = \theta_{11} x`                                   |
+-------------+-------------------------------------------------------------------------+
| square root | :math:`c(x,\theta_1) = \theta_{11} \sqrt{x}`                            |
+-------------+-------------------------------------------------------------------------+
| cubic       | :math:`c(x,\theta_1) = \theta_{11}x+\theta_{12} x**2 + \theta_{13} x**3`|
+-------------+-------------------------------------------------------------------------+
| hyperbolic  | :math:`c(x,\theta_1) = (\theta_{11} / ((states + 1) - x))`              |
+-------------+-------------------------------------------------------------------------+
| quadratic   | :math:`c(x,\theta_1) = (\theta_{11} x +\theta_{12} x**2)`               |
+-------------+-------------------------------------------------------------------------+

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
The second in put are the cost parameters, which are sored as a one dimension
*numpy.array*. At the first position always the replacement cost :math:`RC` is stored.
The next positions are subsequently filled with :math:`theta_{11}, \theta_{12}, ...`. The
exact number depends on the functional form.

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



Fixed point algorithm
---------------------

This part documents the core contribution to research of the Rust (1987) paper, the Fixed
Point Algorithm (FXP). It allows to consequently calculate the log-likelihood value for
each cost parameter and thus, to estimate the model. The computation of the fixed point
is managed by:

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

In the following I describe the conditions, for my algorithm when to switch between those
iterations.

.. _alg_details:

-------------------
Algorithmic details
-------------------

There are several conditions to switch between contraction iterations to
Newton-Kantorovich iterations. In the following the variable keys are presented, which
allow to specify the algorithmic behavior:




.. _ev:

--------------
Expected value
--------------








Other model objects
-------------------

.. _trans_mat:

-----------------
Transition matrix
-----------------

The transition matrix for the Markov process stored in a two dimensional numpy array. As
transition in the case of the replacement corresponds to a transition from state 0, there
is only matrix with number of rows and columns equally to the size of the state space.


.. _pchoice:

--------------------
Choice probabilities
--------------------
