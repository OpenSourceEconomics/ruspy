######################
Estimation
######################

Here the estimation process of the ruspy package is documented. The structure is the
following: The first part documents in detail which format is required on the input data
and how you can easily access this for the original data. Then the estimation process is
documented and an introduction to the demonstration notebooks closes this part.
Throughout this part, there are references to the functions in the ruspy package and in
the end a summary of all APIs.


.. _df:

****************
The input data
****************

The estimation package works with a `pandas.DataFrame
<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_ as
input. Each observation in the DataFrame has to be indexed by the "Bus_ID" and the
"period" of observation. These two identifiers have to be combined by the
`pandas.MultiIndex
<https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html>`_.
Furthermore the DataFrame has to include the following information as columns:

+----------+-----------------------------------------------------------------------+
| Variable | Description                                                           |
+==========+=======================================================================+
| state    | Discretized mileage state of the bus                                  |
+----------+-----------------------------------------------------------------------+
| decision | Containing 0 for decision of maintenance and 1 for replacement        |
+----------+-----------------------------------------------------------------------+
| usage    | Last month mileage usage as discretized state increase                |
+----------+-----------------------------------------------------------------------+

If you want to replicate Rust (1987) you can download the raw data from John Rust`s
`website <https://editorialexpress.com/jrust/research.html>`_ and prepare it yourself it
the desired manner. Or you can use the functions in `zurcher-data
<https://github.com/OpenSourceEconomics/zurcher-data>`_ which are provided by the
`OpenSourceEconomics <https://github.com/OpenSourceEconomics>`_ community and
tailored to the ruspy package. In the demonstration section these functions are used
to replicate the results documented.

************************
The estimation function
************************

The estimation process is coordinated by the function estimate:

.. currentmodule:: ruspy.estimation.estimation

.. autosummary::
    :toctree: _generated/

    estimate


Besides the :ref:`df`, the function needs the following initialization dictionary
**init_dict**:


.. _init_dict:

*************************************
Estimation initialization dictionary
*************************************

The initialization dictionary contains model, optimizer and algorithmic specific
information. The information on theses three categories is saved in subdictionaries
under the keys **model_specifications**, **optimizer** and **alg_details**.  The
model specific information as well as the optimizer key are mandatory.
The options differ depending on whether NFXP or MPEC are chosen as estimation procedure.
The following inputs for the **model_specifications** are mandatory for both
estimation approaches:

**discount_factor :** *(float)* The discount factor. See :ref:`disc_fac` for details.

**number_states :** *(int)* The size of the state space as integer.

**maint_cost_func :** *(string)* The name of the maintenance cost function. See
:ref:`maint_func` for details.

**cost_scale :** *(float)* The scale for the maintenance costs. See :ref:`scale` for
details.

In the subdictionary **optimizer** at least the following has to be specified:

**approach :** *(string)* The general approach chosen which is either "NFXP" or
"MPEC".

**algorithm :** *(string)* The name of the optimization algorithm used to estimate
the model with one of the above approaches. More details below as the selection
is specific to the general approach.

Optionally, irrespective of the approach chosen, one can specify the following:

**derivative :** *(string)* (optional) Information on whether to use analytical or
numerical derivatives. Enter "Yes" for analytical and "No" or numerical derivatives.
*Default is "Yes"*.

As the optimization problem in the NFXP and MPEC are quite different, also in the
**init_dict** many different options are available depending on the approach.

NFXP
==========================

In the **optimizer** subdictionary the following options are implemented:

**algorithm :** *(string)* The algorithms available are those that are offered
by `estimagic <https://estimagic.readthedocs.io/en/latest/>`_. Here, only one
the names of one of `those
<https://estimagic.readthedocs.io/en/latest/optimization/algorithms.html>`_
has to be entered.

**params :** *(pd.DataFrame)* (optional) The first guess of the cost parameter vector
can be supplied. This has to be done according to the `conventions of estimagic
<https://estimagic.readthedocs.io/en/latest/optimization/params.html>`_.

In general any argument that can be chosen in the estimagic function `minimize
<https://estimagic.readthedocs.io/en/latest/optimization/interface.html>`_ can be
passed in as a key in the **optimizer** subdictionary. For example one could specify
the key "logging" and specify the name of the logging database ("logging_nfxp.db").
For performance reasons the logging in ruspy is switched off if not specified
differently. Only the arguments criterion, criterion_kwargs, derivative and
derivative_kwargs cannot be set by the user.

Additionally, the subdictionairy **alg_details** can be used to specify options
for the fixed point algorithm. See :ref:`alg_details` for the possible keys
and the default values.


.. _mpec_params:

MPEC
======================

The **optimizer** subdictionary can contain the following:

**algorithm :** *(string)* The constrained optimization algorithm chosen which
can handle nonlinear equality constraints. So far, there is the option to use **IPOPT**
by specifying "ipopt". For this `cyipopt <https://github.com/matthias-k/cyipopt>`_
is built on which uses `the interface of scipy
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
. If you want to add extra options to the optimizer then you can do so by e.g.
adding a key "options" which is itself a dictionairy as requested by the scipy
interface. The arguments fun, x0, bounds, jac and constraints cannot be specified.
As a second option one can use **NLOPT** by specifying the specific algorithm from
the following `list <https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/>`_
without the prefix "NLOPT", i.e. for example "LD_SLSQP" is a valid choice. Again,
one can pass in extra options to the algorithm. The options available can be found
`here <https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/>`_. The
options are passed in by their name without the prefix "opt.", i.e. a key could
be for instance "set_lower_bounds" and the value is a numpy array specifying the
lower bound. What cannot be used are set_min_objective and
add_equality_mconstraint.

**params :** *(numpy.array)* The starting values for MPEC consist of the cost
parameters and the discretized expected values. The array has therefore a length
of num_states plus num_params. Imagine the grid size is 90 and we have linear cost
which means there are two cost parameters. Then the first 90 values are the
starting values for the expected values in order of increasing state. The last two
elements are :math:`RC` and :math:`\theta_1`, respectively.

There is one special case regarding the interface for **IPOPT**. If you want to
specify bounds for IPOPT then use also the notation of NLOPT as outlined above.

For further details see the selection function itself:


.. currentmodule:: ruspy.estimation.estimation_interface

.. autosummary::
    :toctree: _generated/

    select_optimizer_options


For both NFXP and MPEC, following the separability of the estimation process
the ``estimate`` function first calls the estimation function for the transition
probabilities.

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
<https://estimagic.readthedocs.io/en/latest/optimization/algorithms.html>`_.
The transition probabilities need to add up to 1 and have to be positive
which is conveniently implemented in estimagic using the `constraints argument
<https://estimagic.readthedocs.io/en/latest/optimization/constraints/index.html>`_.


The collected results of the transition estimation are collected in a dictionary
descibed below and returned to the ``estimate`` function in which then the
cost parameters are estimated using either NFXP or MPEC.

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

The cost parameters for the NFXP are estimated directly by minimizing the log-likelihood
the minimize function from the `estimagic library
<https://estimagic.readthedocs.io/en/latest/index.html>`_. The objective function
as well as its analytical derivative can be found in
``ruspy.estimation.est_cost_params``:


.. currentmodule:: ruspy.estimation.est_cost_params

.. autosummary::
  :toctree: _generated/

  loglike_cost_params_individual
  derivative_loglike_cost_params_individual
  loglike_cost_params
  derivative_loglike_cost_params

As estimagic offers to use an implementation of the BHHH also used by Rust (1987)
the first two functions above are needed. They work with the individual
log likelihood contributions of a bus at each time period. The two lower functions
are needed for other algorithms such as the L-BFGS-B provided by estimagic.
For this the previous functions are summed up to obtain to latter ones. The selection
of the correct functions is done by ruspy automatically depending on your choice
of algorithm.

In the minimization proedure the optimizer calls the likelihood functions and its
derivative with different cost parameters. Together with the constant held
arguments, the expected value is calculated by fixed point algorithm. Double
calculation of the same fixed point is avoided by the following function:

.. currentmodule:: ruspy.estimation.est_cost_params

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


For both NFXP and MPEC some of the results from the estimators estimagic, ipopt
and nlopt are passed on to the user. The results are presented below.

.. _result_costs:

******************************
Cost parameters results
******************************

Again there are slight differences for NFXP and MPEC.
The dictionary containing the cost parameter results has the following keys for both
NFXP and MPEC:

**fun :** *(numpy.float)* Log-likelihood of the cost parameter estimation.

**x :** *(numpy.array)* Estimated cost parameters and in the case of MPEC also
the estimated expected values.

**status :** *(bool)* Evaluates to True if the optimizer converged and False
if not.

**n_iterations :** *(int)* Gives out the number of iterations needed by the
algorithm.

**n_evaluations :** *(int)* Gives out the number of function evaluations needed
by the algorithm.

**time :** *(float)* Indicates the time needed by the optimizer to obtain the final
cost parameter estimates.

For the **NFXP** there are also the following keys:

**jac :** *(numpy.array)* The value of the estimates' jacobian.

**message :** *(string)* The convergence message of estimagic.

**n_contraction_steps :** *(int)* The number of contraction iterations needed in
total during the optimization to calculate the fixed points.

**n_newt_kant_steps :** *(int)* The number of Newton-Kantorovich iterations needed in
total during the optimization to calculate the fixed points.

When using **IPOPT** for **MPEC** the following key is included:

**n_evaluations_total :** *(int)* The number of total function evaluations needed
which is also including function evaluations made to approximate the derivatives
of the log likelihood function and the constraints.

The function ``estimate`` calls some sub functions depending on whether NFXP, MPEC
with IPOPT or MPEC with NLOPT is selected. Those functions can be inspected below:

.. currentmodule:: ruspy.estimation.estimation

.. autosummary::
    :toctree: _generated/

    estimate_nfxp
    estimate_mpec_ipopt
    estimate_mpec_nlopt


Auxiliary objects
=====================

.. _state_mat:


State matrix
--------------------

A :math:`num\_obs \times num\_states` dimensional *bool numpy.array* containing a
single TRUE in each row at the column in which the bus was in that observation. It is
used in the matrix multiplication of the likelihood function. It is created by

.. currentmodule:: ruspy.estimation.est_cost_params

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
