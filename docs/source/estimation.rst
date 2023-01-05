######################
Estimation
######################

Here the estimation process of the ruspy package is documented. The structure
is the following: The first part documents in detail which format is required
on the input data and how you can easily access this for the original data.
Then the estimation process is documented and an introduction to the
demonstration notebooks closes this part. Throughout this part, there are
references to the functions in the ruspy package and in the end a summary of
all APIs.


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
tailored to the ruspy package. In the demonstration section these functions are used
to replicate the results documented.

**********************
The estimation process
**********************

The estimation process is coordinated by the function ``get_criterion_function``:

.. currentmodule:: ruspy.estimation.criterion_function

.. autosummary::
    :toctree: _generated/

    get_criterion_function


Besides the :ref:`df`, the function needs the following initialization dictionary
**init_dict**:


.. _init_dict:

*************************************
Estimation initialization dictionary
*************************************

The initialization dictionary contains model, method and optional algorithmic
specific information. The information on theses three categories is saved under
the key **method** and in subdictionairies under the keys **model_specifications**
and **alg_details**.
The model specific information as well as the method key are mandatory.

The following inputs for the **model_specifications** are mandatory:

**discount_factor :** *(float)* The discount factor. See :ref:`disc_fac` for details.

**num_states :** *(int)* The size of the state space as integer.

**maint_cost_func :** *(string)* The name of the maintenance cost function. See
:ref:`maint_func` for details.

**cost_scale :** *(float)* The scale for the maintenance costs. See :ref:`scale` for
details.

In the key **method** the following has to be specified:

**method:** *(string)* The general approach chosen which is either "NFXP",
"NFXP_BHHH" or "MPEC".

If "NFXP" or "NFXP_BHHH" are chosen as **method**, then the additional subdictionairy
**alg_details** can be used to specify options for the fixed point algorithm.
See :ref:`alg_details` for the possible keys and the default values.






For both NFXP and MPEC, following the separability of the estimation process
the function ``get_criterion_function`` first calls the estimation function for
the transition probabilities.

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

    loglike_trans_individual


The ``estimate_transitions`` function minimizes now the ``loglike_trans_individual``
function by calling the `BHHH of estimagic
<https://estimagic.readthedocs.io/en/stable/algorithms.html>`_.
The transition probabilities need to add up to 1 and have to be positive
which is conveniently implemented in estimagic using the `constraints argument
<https://estimagic.readthedocs.io/en/stable/how_to_guides/optimization/
how_to_specify_constraints.html>`_.


The collected results of the transition estimation are collected in a dictionary
descibed below and returned to the function ``get_criterion_function`` in which
then the respective criterion function for the cost parameter estimation is
specified.

.. _result_trans:

Transition results
====================

The dictionary containing the transition estimation results has the following keys:

**fun :** *(numpy.float)* Log-likelihood of transition estimation.

**x :** *(numpy.array)* Estimated transition probabilities.


So far only a pooled estimation of the transitions is possible. Hence, ``ruspy``
uses the estimated probabilities to construct a transition matrix with the same
nonzero probabilities in each row. This function is:

.. currentmodule:: ruspy.estimation.estimation_transitions

.. autosummary::
    :toctree: _generated/

    create_transition_matrix


The transition matrix is then used for the cost parameter estimation irrespective
of using NFXP or MPEC.

***************************
Cost parameter estimation
***************************

The cost parameters are now estimated differently for NFXP and MPEC.

NFXP
=========================

The cost parameters for the NFXP are estimated by minimizing the negative
log-likelihood. The criterion function as well as its analytical derivative
are returned by the function ``get_criterion_function`` and can be found
in ``ruspy.estimation.nfxp``:

.. currentmodule:: ruspy.estimation.nfxp

.. autosummary::
  :toctree: _generated/

  loglike_cost_params_individual
  derivative_loglike_cost_params_individual
  loglike_cost_params
  derivative_loglike_cost_params


The minimization of the criterion function is not directly implemented in the
ruspy package, so that any minimization rountine can be used. However, we recommend
using the minimize function from the
`estimagic library <https://estimagic.readthedocs.io/en/v0.0.28/optimization/interface.html>`_,
as estimagic offers to use an implementation of the BHHH also used by Rust (1987).
They work with the individual
log likelihood contributions of a bus at each time period. The two lower functions
are needed for other algorithms such as the L-BFGS-B provided by estimagic.
For this, the previous functions are summed up to obtain to latter ones. The selection
of the correct functions is done by ruspy automatically depending on your choice
of method ("NFXP" or "NFXP_BHHH").

When calling the minimize function from estimagic, the following inputs are needed:

**criterion :** *(callable)* (Negative) log-likelihood of the cost parameter
estimation returned by ``get_criterion_function``.

**algorithm :** *(string)* Algorithm used for optimization. If method is "NFXP_BHHH",
then algorithm has to be "bhhh". If method is "NFXP", then any algorithm offered
by `estimagic <https://estimagic.readthedocs.io/en/latest/>`_ can be used. Here, only one
the names of one of `those
<https://estimagic.readthedocs.io/en/v0.0.28/optimization/
algorithms.html#list-of-algorithms>`_
has to be entered.

**params :** *(numpy.float)* (optional) The first guess of the cost parameter vector
can be supplied. This has to be done according to the `conventions of estimagic
<https://estimagic.readthedocs.io/en/v0.0.28/optimization/params.html?highlight=params>`_.
Note that the size of the vector has to match the number of the cost parameters
of the considered cost function, i.e. if we specify a linear cost function in
the initialization dictionary, there are two cost parameters, which are :math:`RC`
and :math:`\theta_1`, respectively.

**derivative :** *(numpy.float)* Derivative of the criterion function returned
by ``get_criterion_function``.

In the minimization procedure the optimizer calls the likelihood functions and its
derivative with different cost parameters. Together with the constant held
arguments, the expected value is calculated by fixed point algorithm. Double
calculation of the same fixed point is avoided by the following function:

.. currentmodule:: ruspy.estimation.nfxp

.. autosummary::
    :toctree: _generated/

    get_ev


MPEC
==============================

In the case of MPEC there is no need to calculate the fixed point via the function
``get_ev`` but rather a constraint to the likelihood function with the contraction
mapping has to be specified.
The functions needed for MPEC are hence the four below being the log likelihood
function that is now also dependent on :math:`EV`, the constraints as well as
the analytical derivatives of the two.

.. currentmodule:: ruspy.estimation.mpec

.. autosummary::
    :toctree: _generated/

    mpec_loglike_cost_params
    mpec_loglike_cost_params_derivative
    mpec_constraint
    mpec_constraint_derivative


As for NFXP, the minimization of the criterion function is not directly implemented
in the ruspy package, but again we recommend using the minimize function from the
`estimagic library
<https://estimagic.readthedocs.io/en/v0.0.28/optimization/interface.html>`_.

For MPEC, the function ``get_criterion_function`` additionally returns the constraint
function and its derivative, that can be passed to the ``minimize`` function in a
dictionairy under the argument ``constraint`` (see `constraint argument
<https://estimagic.readthedocs.io/en/stable/how_to_guides/optimization/
how_to_specify_constraints.html>`_)
beside the criterion function, its derivative, the algorithm and starting
values ``params``. Note that the starting values for MPEC consist of the cost
parameters and the discretized expected values. The array has therefore a length
of num_states plus num_params. Imagine the grid size is 90 and we have linear cost
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

In the promotion folder of the repository are two demonstration jupyter notebooks. The
`replication <https://github.com/OpenSourceEconomics/ruspy/blob/master/promotion
/replication/replication.ipynb>`_ notebook allows to easily experiment with the
methods described here as well as the implied demand function. If you have have
everything setup, then it should be easy to run it. For a more advanced set up have
a look at the `replication of Iskhakov et al. (2016)
<https://github.com/OpenSourceEconomics/ruspy/blob/master/promotion/replication
/replication_iskhakov_et_al_2016.ipynb>`_.
