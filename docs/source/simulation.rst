Simulation
========================


The simulation package contains the functions to simulate a single agent in a dynamic
discrete choice model. It is structured into two modules. The simulation module with
the main function and the ``simulation_auxiliary`` with all supplementary functions.

The simulate function
---------------------

The simulate function can be found in ``ruspy.simulation.simulation`` and coordinates
the whole simulation process.

.. currentmodule:: ruspy.simulation.simulation

.. autosummary::
    :toctree: _generated/

    simulate

Besides the input :ref:`costs`, :ref:`ev` and :ref:`trans_mat`, there is more input
specific to the simulation function.

.. _sim_init_dict:

Simulation initialization dictionary
------------------------------------

The initialization dictionary contains the following keys:

**seed :** *(int)* A positive integer setting the random seed for drawing random
numbers. If none given, some random seed is drawn.

**discount_factor :** *(float)* See :ref:`disc_fac` for more details.

**buses :** *(int)* The number of buses to be simulated.

**periods :** *(int)* The number of periods to be simulated.


The simulation process
----------------------

After all inputs are read in, the actual simulation starts. This is coordinated by:

.. currentmodule:: ruspy.simulation.simulation_functions

.. autosummary::
    :toctree: _generated/

    simulate_strategy

The function calls in each period for each bus the following function, to choose the
optimal decision:

.. currentmodule:: ruspy.simulation.simulation_model

.. autosummary::
    :toctree: _generated/

    decide

Then the mileage state increase is drawn:

.. currentmodule:: ruspy.simulation.simulation_model

.. autosummary::
    :toctree: _generated/

    draw_increment

.. _sim_results:

The simulation
--------------
After the simulation process the observed states, decisions and mileage uses are
returned. Additionally the agent's utility. They are all stored in pandas.DataFrame with
column names **states**, **decisions**, **utilities** and **usage**. Hence, the observed
data is a subset of the returned Dataframe.

Demonstration
-------------

The `simulation <notebooks/simulation_convergence.ipynb>`_ notebook allows to easily
experiment with the estimation methods described here. The notebook can be downloaded
from the `repository <https://github.com/OpenSourceEconomics/ruspy/blob/master/promotion
/simulation/simulation_convergence.ipynb>`_. If you have have everything
setup, then it should be easy to run it.
