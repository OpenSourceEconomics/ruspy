Estimation
========================

Here the estimation process of the ruspy packacge is documented. The structure is the
following: The first part documents in detail which format is required on the input data
and how you can easily access this for the original data. Then the estimation process is
documented and an introduction to the demonstration notebooks closes this part.
Throughout this part, there are references to the functions in the ruspy package and in
the end a summary of all APIs.


.. _df:

The input data
--------------

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
`OpenSourceEconomics <https://github.com/OpenSourceEconomics>`_ community and tailored to
the ruspy package. In the demonstration section these functions are used to replicate the
results documented.


The estimation function
-----------------------
The estimation process is coordinated by the function estimate:

.. currentmodule:: ruspy.estimation.estimation

.. autosummary::
    :toctree: _generated/

    estimate

Following the separability of the estimation process the `estimate` function first calls
the estimation function for the transition probabilities.

---------------------------------
Transition probability estimation
---------------------------------

The funtions for estimating the transition probabilities can be found in
``estimation.estimation_transitions``. The main function, which coordinates this process
is:

.. currentmodule:: ruspy.estimation.estimation_transitions

.. autosummary::
    :toctree: _generated/

    estimate_transitions

So far, there is only the pooled transition from Rust (1987) implemented. The function
filters missing values from the usage data from the DataFrame and then counts, how often
each increase occurs. With this transition count the log-likelihood function for the
transition estimation can be constructed. Note that this is the log-likelihood function
of a multinomial distribution:

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


The ``estimate_transitions`` function minimizes now the ``loglike_trans`` function by
calling a minimize routine from the scipy library. Even though transition probabilities
need to add up to 1 and have to be positive, there are no constraints on the minimized
parameters. The constraints are applied inside ``loglike_trans`` by a reparametrization
function:

.. currentmodule:: ruspy.estimation.estimation_transitions

.. autosummary::
    :toctree: _generated/

    reparam_trans

This allows to create the 95\% interval of the estimated to bootstrap transition
probabilities from the asymptotic distribution provided by
the scipy minimizer. The core function for this can be found in
``ruspy.estimation.bootstrapping``:

.. currentmodule:: ruspy.estimation.bootstrapping

.. autosummary::
    :toctree: _generated/

    calc_95_conf

The collected results of the transition estimation are collected in a dictionary and
returned to the ``estimate`` function.

.. _result_trans:

Transition results
""""""""""""""""""
The dictionary containing the transition estimation results has the following keys:

**fun :** *(numpy.float)* Log-likelihood of transition estimation.

**x :** *(numpy.array)* Estimated transition probabilities.

**trans_count :** *(numpy.array)* Number of transitions for an increase of 0, 1, 2, ...

**95_conf_interv :** *(numpy.array)*
2 x dim(x) matrix containing the bootstrapped (1000 replications) 95% confidence interval
bounds.

---------------------------------
Cost parameter estimation
---------------------------------

The



Cost Parameters
"""""""""""""""

.. currentmodule:: ruspy.estimation.est_cost_params

.. autosummary::
  :toctree: _generated/

  loglike_cost_params
  derivative_loglike_cost_params

.. _result_costs:

Cost parameters results
"""""""""""""""""""""""
