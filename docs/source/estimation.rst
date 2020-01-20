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

Besides the :ref:`df`, the function needs the following initialization dictionary:


------------------------------------
Estimation initialization dictionary
------------------------------------

The initialization dictionary contains data and model specific information. Besides that it let allows to specify the optimizer from the `scipy library
<http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.optimize.minimize.html>`_. All inputs are either strings or numbers in any format:

*groups :* Specify which groups are








Following the separability of the estimation process the `estimate` function first calls
the estimation function for the transition probabilities.

---------------------------------
Transition probability estimation
---------------------------------

The functions for estimating the transition probabilities can be found in
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
calling a minimize routine from the `scipy library
<http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.optimize.minimize.html>`_.
Even though transition probabilities need to add up to 1 and have to be positive, there
are no constraints on the minimized parameters. The constraints are applied inside
``loglike_trans`` by a reparametrization function:

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

    bootstrapp


The collected results of the transition estimation are collected in a dictionary and
returned to the ``estimate`` function.

Transition results
""""""""""""""""""
The dictionary containing the transition estimation results has the following keys:

**fun :** *(numpy.float)* Log-likelihood of transition estimation.

**x :** *(numpy.array)* Estimated transition probabilities.

**trans_count :** *(numpy.array)* Number of transitions for an increase of 0, 1, 2, ...

**95_conf_interv :** *(numpy.array)*
2 x dim(x) matrix containing the bootstrapped (1000 replications) 95% confidence interval
bounds.

**std_errors :** *(numpy.array)*
dim(X) numpy array with bootstrapped standard errors for each parameter.


So far only a pooled estimation of the transitions is possible. Hence, ``ruspy``
uses the estimated probabilities to construct a transition matrix with the same
nonzero probabilities in each row. This function is:

.. currentmodule:: ruspy.estimation.estimation_transitions

.. autosummary::
    :toctree: _generated/

    create_transition_matrix


The transition matrix is then used for the cost parameter estimation.

.. _result_trans:

---------------------------------
Cost parameter estimation
---------------------------------

The cost parameters are estimated directly by minimizing the log-likelihood and the corresponding jacobian function with a minimize function from the `scipy library <http://lagrange.univ-lyon1.fr/docs/scipy/0.17.1/generated/scipy.optimize.minimize.html>`_ . The functions can be found in ``ruspy.estimation.est_cost_params``:


.. currentmodule:: ruspy.estimation.est_cost_params

.. autosummary::
  :toctree: _generated/

  loglike_cost_params
  derivative_loglike_cost_params

For the construction of the likelihood function, ruspy uses matrix algebra and
therefore creates a :ref:`state_mat` with:

.. currentmodule:: ruspy.estimation.est_cost_params

.. autosummary::
    :toctree: _generated/

    create_state_matrix

In the minimization the scipy optimizer calls the likelihood functions and its
derivative with different cost parameters. Together with the constant held
arguments, the expected value is calculated by fixed point algorithm. Double
calculation of the same fixed point is avoided by the following function:

.. currentmodule:: ruspy.estimation.est_cost_params

.. autosummary::
    :toctree: _generated/

    get_ev

After successful minimization, some results of the scipy result dictionary are used
to construct ruspys cost parameter results:


.. _result_costs:

Cost parameters results
"""""""""""""""""""""""
The dictionary containing the cost parameter results has the following keys:

**fun :** *(numpy.float)* Log-likelihood of cost parameter estimation.

**x :** *(numpy.array)* Estimated cost parameters.

**message :** *(string)* The optimizer message of the scipy optimizer.

**jac :** *(numpy.array)* The value of the estimates' jacobian.

**95_conf_interv :** *(numpy.array)*
2 x dim(x) matrix containing the bootstrapped (1000 replications) 95% confidence
interval bounds.

**std_errors :** *(numpy.array)*
dim(X) numpy array with bootstrapped standard errors for each parameter.
