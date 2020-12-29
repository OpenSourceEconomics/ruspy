ruspy
======

.. image:: https://anaconda.org/opensourceeconomics/ruspy/badges/version.svg
    :target: https://anaconda.org/OpenSourceEconomics/ruspy/

.. image:: https://anaconda.org/opensourceeconomics/ruspy/badges/platforms.svg
    :target: https://anaconda.org/OpenSourceEconomics/ruspy/

.. image:: https://readthedocs.org/projects/ruspy/badge/?version=latest
    :target: https://ruspy.readthedocs.io/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

.. image:: https://github.com/OpenSourceEconomics/ruspy/workflows/Continuous%20Integration%20Workflow/badge.svg
    :target: https://github.com/OpenSourceEconomics/ruspy/actions

.. image:: https://codecov.io/gh/OpenSourceEconomics/robupy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/OpenSourceEconomics/robupy

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

``ruspy`` is an open-source package for the simulation and estimation of a prototypical
infinite-horizon dynamic discrete choice model based on

    Rust, J. (1987). `Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher. <https://doi.org/10.2307/1911259>`_ *Econometrica, 55* (5), 999-1033.

You can install ``ruspy`` via conda with

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -c opensourceeconomics ruspy

Please visit our `online documentation <https://ruspy.readthedocs.io/>`_ for
tutorials and other information.


Citation
--------

If you use ruspy for your research, do not forget to cite it with

.. code-block:: bash

    @Unpublished{ruspy-1.1,
        Author = {Maximilian Blesch},
        Title  = {ruspy - An open-source package for the simulation and estimation of a prototypical infinite-horizon dynamic discrete choice model based on Rust (1987)},
        Year   = {2020},
        Url    = {https://github.com/OpenSourceEconomics/ruspy},
        }
