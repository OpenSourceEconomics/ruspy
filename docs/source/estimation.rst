######################
Estimation
######################

On this page the estimation process of the ruspy package is documented. The structure
is the following: The first part documents in detail which format is required
on the input data and how you can easily access this for the original data. Then we
explain how the initialization dictionary specified, before explaining the different
estimation steps and methods.


.. _df:

****************
The input data
****************

The estimation package works with a `pandas.DataFrame
<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
as input. Each observation in the DataFrame has to be indexed by the "Bus_ID" and
the "period" of observation. These two identifiers have to be combined by the
`pandas.MultiIndex
<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html>`_.
Furthermore the DataFrame has to include the following information as columns:

+----------+-----------------------------------------------------------------------+
| Variable | Description                                                           |
+==========+=======================================================================+
| state    | Discretized mileage state of the bus.                                 |
+----------+-----------------------------------------------------------------------+
| decision | Containing 0 for decision of maintenance and 1 for replacement.       |
+----------+-----------------------------------------------------------------------+
| usage    | Last month mileage usage as discretized state increase.               |
+----------+-----------------------------------------------------------------------+

If you want to replicate Rust (1987) you can download the raw data from John Rust`s
`website <https://editorialexpress.com/jrust/research.html>`_ and prepare it yourself it
the desired manner. Or you can use the functions in `zurcher-data
<https://github.com/OpenSourceEconomics/zurcher-data>`_ which are provided by the
`OpenSourceEconomics <https://github.com/OpenSourceEconomics>`_ community and
tailored to the ruspy package.


**********************
The estimation process
**********************

The estimation process is not directly implemented in ruspy. The package only contains
the likelihood functions and in case of MPEC also the constraints. For the
(minimization) maximization of the (negative) loglikelihood function, an external
optimization library has to be used. Hence ruspy is a so called model package.
OpenSourceEconomics offers several of these model packages, all for different models.
More information can be found on our `homepage <https://open-econ.org>`_. The central
function to get the criterion function, it's derivative and if applicable the constraint
is the ``get_criterion_function`` function. It's source code can be found here:

.. currentmodule:: ruspy.estimation.criterion_function

.. autosummary::
    :toctree: _generated/

    get_criterion_function


The returns of the function will be explained below. First, the second input besides the
input data :ref:`df`, the initialization dictionary is documented:


.. _init_dict:

*************************************
Estimation initialization dictionary
*************************************

The initialization dictionary contains model, method and optional algorithmic
specific information. The information on theses three categories is are saved under the
dictionary keys **method**, **model_specifications**
and **alg_details**.
The keys **method** and **model_specifications** are mandatory. In the following the
entries saved under the three keys is explained

Under the key **model_specifications** a subdictionary has to be provided, with the
following mandatory keys:

- **discount_factor :** *(float)* The discount factor. See :ref:`disc_fac` for details.

- **num_states :** *(int)* The size of the state space as integer.

- **maint_cost_func :** *(string)* Name of the cost function. See :ref:`maint_func`
  for details.

- **cost_scale :** *(float)* The scale for the maintenance costs. See :ref:`scale`
  for details.


Under the key **method** the method of estimation has to be specified as a *(string)*:
Ruspy supports the following keys: "NFXP", "NFXP_BHHH" or "MPEC".


If "NFXP" or "NFXP_BHHH" are chosen as **method**, then the additional subdictionairy
**alg_details** can be used to specify options for the fixed point algorithm.
See :ref:`alg_details` for the possible keys and the default values.


Before explaining the cost parameter estimation with the likelihood function from
``get_criterion_function``, the transition probability estimation is documented. This
estimation can be completely separated from the cost parameter estimation.

**********************************
Transition probability estimation
**********************************

The functions for estimating the transition probabilities can be found in
``estimation.estimation_transitions``. The main function, which coordinates this process
is:

.. currentmodule:: ruspy.estimation.estimation_transitions

.. autosummary::
    :toctree: _generated/

    estimate_transitions

So far, there is only the pooled transition estimation from Rust (1987) implemented. The
function filters missing values from the usage data from the DataFrame and then counts,
how often each increase occurs. With this transition count the log-likelihood function
for the transition estimation can be constructed. Note that this is the log-likelihood
function of a multinomial distribution:

.. math::

    \begin{align}
     l^1 = - \sum a_i \log(p_i)
    \end{align}

where :math:`a_i` is the number of occurrences for an increase by :math:`i` states and
:math:`p_i` their probability. Note that the minus is introduced, such that a
maximization of the likelihood corresponds to a minimization of this function. The
corresponding function in the code is:

.. currentmodule:: ruspy.estimation.estimation_transitions

.. autosummary::
    :toctree: _generated/

    loglike_trans


The ``estimate_transitions`` function does not minimize ``loglike_trans`` directly.
Instead ruspy uses the formula for estimating a multinomial distribution and then
calculates the likelihood value from the estimate. Therefore we don't provide standard
errors of the transitions probabilities at the moment.


The collected results of the transition estimation are collected in a dictionary
described below and returned to the function ``get_criterion_function`` in which
then the respective criterion function for the cost parameter estimation is
specified. The transition result is also returned as the second output from
``get_criterion_function``.

.. _result_trans:

Transition results
====================

The dictionary containing the transition estimation results has the following keys:

- **fun :** *(numpy.float)* Log-likelihood of transition estimation.

- **x :** *(numpy.array)* Estimated transition probabilities.

- **trans_count :** *(numpy.array)* Counted state increases for each array index.


So far only a pooled estimation of the transitions is possible. Hence, ``ruspy``
uses the estimated probabilities to construct a transition matrix with the same
nonzero probabilities in each row. This function is:

.. currentmodule:: ruspy.estimation.estimation_transitions

.. autosummary::
    :toctree: _generated/

    create_transition_matrix


The transition matrix is then used for the cost parameter estimation irrespective
of using NFXP or MPEC.

.. _func_dict:

***************************
Cost parameter estimation
***************************

The cost parameters are now estimated differently for NFXP, NFXP_BHHH and MPEC.
``get_criterion_function`` returns independet of the specified method two objects.
A dictionary of functions and the transition results. Only the keys and the functions
in the function dictionary are different by each method and described below. Note,
that all inputs are fixed for the functions in function dictionary dependent on the
specifications given in the initialization dictionary and the functions only take the
cost parameters as only input.

NFXP
=========================

The cost parameters for the NFXP are estimated by minimizing the negative
log-likelihood. The criterion function as well as its analytical derivative
are returned by the function ``get_criterion_function`` in a dictonary with keys
"criterion_function" and "criterion_derivative". Their source code can be found
in ``ruspy.estimation.nfxp``:

.. currentmodule:: ruspy.estimation.nfxp

.. autosummary::
  :toctree: _generated/

  loglike_cost_params
  derivative_loglike_cost_params


The minimization of the criterion function is not directly implemented in the
ruspy package, so an minimization routine is needed. In the provided
tutorials, we use the minimize function from the
`estimagic library <https://estimagic.readthedocs.io>`_.
Beside the criterion function and its derivative, an `algorithm
<https://estimagic.readthedocs.io/en/stable/algorithms.html>`_ used for optimization
has to be entered and a first guess of the cost params can be provided as inputs
of the ``minimize``function. Note, again that only the cost parameters are needed in
the minimization, as all other inputs of the functions are fixed.
Depending on the form of the cost functions, the params argument is a vector of
length ``num_params``, i.e. if we specify a linear cost function in
the initialization dictionary, there are two cost parameters, which are :math:`RC`
and :math:`\theta_1`, respectively. For any other cost function see ref:`cost_func`.

In the minimization procedure the optimizer calls the likelihood functions and its
derivative with different cost parameters. Together with the constant held
arguments, the expected value is calculated by fixed point algorithm. Double
calculation of the same fixed point is avoided by the following function:

.. currentmodule:: ruspy.estimation.nfxp

.. autosummary::
    :toctree: _generated/

    get_ev


NFXP_BHHH
=========================

The cost parameter estimation for "NFXP_BHHH" is similar to the one for
"NFXP" by using the individual log likelihood contributions of a bus at each
time period. The criterion function as well as its analytical derivative
are returned by the function ``get_criterion_function`` in a dictonary with keys
"criterion_function" and "criterion_derivative". Their source code can also be found
in ``ruspy.estimation.nfxp``:

.. currentmodule:: ruspy.estimation.nfxp

.. autosummary::
  :toctree: _generated/

  loglike_cost_params_individual
  derivative_loglike_cost_params_individual



The BHHH is a quasi-Newton method, which uses the individual likelihood contributions
instead of their sum. You can find a BHHH implementation in the overview of
`estimagic algorithms <https://estimagic.readthedocs.io/en/stable/algorithms.html>`_.
Everything else is the same as in the NFXP implementation using the sum
of the likelihood contributions.

.. _mpec_params:

MPEC
==============================

In the case of MPEC there the expected value fixed point is not calculated for a set of
cost parameters and instead the fixed point mapping is implemented as a constraint.
We provide besides the criterion function and its derivative, also the constraint and
its derivative via the ``get_criterion_function``. They are returned in a dictionary of
functions with keys "criterion_function", "criterion_derivative", "constraint" and
"constraint_derivative". The source code of the four functions can be found in
``ruspy.estimation.mpec``:

.. currentmodule:: ruspy.estimation.mpec

.. autosummary::
    :toctree: _generated/

    mpec_loglike_cost_params
    mpec_loglike_cost_params_derivative
    mpec_constraint
    mpec_constraint_derivative



For estimating the model, one can use the optimizers for non-linear constraint
optimizers implemented in `estimagic <https://estimagic.readthedocs.io>`_.
The ``minimize`` function of estimagic takes the criterion function, its derivative,
the constraint function and its derivative as inputs. The constraint can be given to the
``minimize`` function via a dictionairy under the argument ``constraint``
(see `constraint argument
<https://estimagic.readthedocs.io/en/stable/how_to_guides/optimization/
how_to_specify_constraints.html>`_) Note that the starting values ``params`` for MPEC
consist of the cost parameters and starting values for the :math:`EV` fixed point. The
array has therefore a length of :math:`num\_states + num\_params`.

Imagine the grid size is 90 and we have linear cost
which means there are two cost parameters. Then the first 90 values are the
starting values for the expected values in order of increasing state. The last two
elements are :math:`RC` and :math:`\theta_1`, respectively.




Auxiliary objects
=====================

.. _state_mat:


State matrix
--------------------

A :math:`num\_obs \times num\_states` dimensional *bool numpy.array* containing a
single TRUE in each row at the column in which the bus was in that observation. It is
used in the matrix multiplication of the likelihood function. It is created by

.. currentmodule:: ruspy.estimation.nfxp

.. autosummary::
    :toctree: _generated/

    create_state_matrix


.. _decision_mat:


Decision Matrix
---------------------

A :math:`num\_obs \times 2` dimensional numpy array containing 1 in the first row for
maintaining and 1 in the second for replacement. It is used in the matrix multiplication
of the likelihood function.

****************
Demonstration
****************

In the tutorials are two demonstration jupyter notebooks of the cost estimation process.
The `replication <tutorials/replication/replication.ipynb>`_ notebook allows to easily
experiment with the methods described here as well as the implied demand function.
The notebook can also be downloaded from the tutorials folder of the
`repository <https://github.com/OpenSourceEconomics/ruspy/tree/master/docs/source/replication>`_.
If you have have everything setup, then it should be easy to run it.
For a more advanced set up have a look at the `replication of Iskhakov et al. (2016)
<tutorials/replication/replication_iskhakov_et_al_2016.ipynb>`_.
