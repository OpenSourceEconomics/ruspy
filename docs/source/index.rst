Welcome to ruspy's documentation!
=================================

Ruspy is an open-source software package for estimating and simulating an infinite
horizon single agent discrete choice model in the setting of Rust (1987).
This package offers to choose whether to estimate the model using the nested fixed
point algorithm suggested by Rust (1987) or by employing the mathematical programming
with equilibrium constraints based on Su and Judd (2012). It serves
as a foundation for teaching and research in this particular model and can be used
freely by everyone. For a full understanding of the mechanisms in this package it is
advisable to first read the two papers:

  Rust, J.  (1987). `Optimal replacement of GMC bus engines: An empirical model of Harold
  Zurcher. <https://doi.org/10.2307/1911259>`_ *Econometrica, 55* (5), 999-1033.

  Su, C. L., & Judd, K. L. (2012).  `Constrained optimization approaches to estimation of
  structural models. <https://www.jstor.org/stable/23271445>`_ *Econometrica, 80* (5), 2213-2230.

and the documentation provided by John Rust on his website:

  Rust, J. (2000). `Nested fixed point algorithm documentation manual.
  <https://editorialexpress.com/jrust/nfxp.pdf>`_ *Unpublished Manuscript.*

as well as the comment by Iskakhov et al. (2016) on Su and Judd (2012):

  Iskhakov, F., Lee, J., Rust, J., Schjerning, B., & Seo, K. (2016). `Comment on
  “constrained optimization approaches to estimation of structural models”. <https://doi.org/10.3982/ECTA12605>`_
  *Econometrica, 84* (1), 365-370.

So far, there has been only one research project based on this code. The promotional
material for this project can be found
`here. <https://github.com/robustzurcher/promotion>`_

ruspy can be installed via conda with:

.. code-block:: bash

      $ conda config --add channels conda-forge
      $ conda install -c opensourceeconomics ruspy


After installing ruspy, you can familiarize yourself with ruspy's tools and
interface by exploring multiple tutorial notebooks. Note that for a full
comprehension, you should read the papers above or study at least the economics
section of this documentation. We provide a `simulation <https://github
.com/OpenSourceEconomics/ruspy/blob/master/promotion
/simulation/simulation_convergence.ipynb>`_ and `replication <https://
github.com/OpenSourceEconomics/ruspy/blob/master/promotion
/replication/replication.ipynb>`_. The first one puts more focus on the simulation
function of ruspy while the latter has a closer look at the estimation function.
Lastly, for a combination of both you can further dive into the `replication of
Iskhakov et al. (2016) <https://github
.com/OpenSourceEconomics/ruspy/blob/master/promotion/replication
/replication_iskhakov_et_al_2016.ipynb>`_ notebook which allows to replicate this paper
using ruspy.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   economics
   model_code
   estimation
   simulation
   references
   credits
   api