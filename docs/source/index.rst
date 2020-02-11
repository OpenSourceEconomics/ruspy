Welcome to ruspy's documentation!
=================================

Ruspy is an open-source software package for estimating and simulating infinite
horizon single agent discrete choice model in the setting of Rust (1987). It serves
as a foundation for teaching and research in this particular model and can be used
freely by everyone. For a full understanding of the mechanisms in this package it is
advisable to first read the paper:

Rust, J.  (1987). `Optimal replacement of GMC bus engines: An empirical model of Harold
Zurcher. <https://doi.org/10.2307/1911259>`_ *Econometrica, 55* (5), 999-1033.

and the documentation provided by John Rust on his website:

  Rust, J. (2000). `Nested fixed point algorithm documentation manual.
  <https://editorialexpress.com/jrust/nfxp.pdf>`_ *Unpublished Manuscript.*


So far, there has been only one research project based on this code. Numerical
experiments for a robust decision rule for Harold Zurcher can be found in this online
`organisation. <https://github.com/robustzurcher>`_

Using the code needs this `dependencies
<https://github.com/OpenSourceEconomics/ruspy/blob/master/environment.yml>`_ After
cloning the repository, you can install the ruspy package via pip locally. This can
easily be done by

.. code-block:: bash

      $ pip install -e ruspy




.. toctree::
   :maxdepth: 2
   :caption: Contents:

   economics
   model_code
   estimation
   simulation
   references
