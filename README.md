# Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/91ce9e983dea4403b986f0ca69564818)](https://app.codacy.com/app/OpenSourceEconomics/ruspy?utm_source=github.com&utm_medium=referral&utm_content=OpenSourceEconomics/ruspy&utm_campaign=Badge_Grade_Dashboard)

This repository contains code to replicate some descriptives and the major results of
> Rust, J. (1987). [Optimal Replacement of GMC Bus Engines: An empirical model of Harold Zurcher.](https://doi.org/10.2307/1911259) *Econometrica*, Vol. 55, No.5, 999-1033.

Setup
-----
To run the code you first need to setup a envoirement fulffiling these [dependencies](https://github.com/OpenSourceEconomics/ruspy/blob/master/environment.yml)
After cloning the repository, you can install the ruspy package. This can easily be done by

  $ pip install -e ruspy

Exploring
---------
If you now first of all, want to get a little introduction and see some descriptives of the paper, please refer to this [notebook](https://github.com/OpenSourceEconomics/ruspy/blob/master/replicate%20descriptives.ipynb).

For a longer introduction to my project please first run this [creation script](https://github.com/OpenSourceEconomics/ruspy/blob/master/create_project.py) and then you can find a more detailed description on my project in [here](https://github.com/OpenSourceEconomics/ruspy/tree/master/promotion/project_description).

Also if you ran the creation script you can have a look at the detailed documentation of my code in the build/latex folder in the [documentation section](https://github.com/OpenSourceEconomics/ruspy/tree/master/documentation).

If you would like to replicate the paper yourself, then please either leave the settings in [here](https://github.com/OpenSourceEconomics/ruspy/blob/master/promotion/replication/init_replication.yml) or change them and then run this [red button](https://github.com/OpenSourceEconomics/ruspy/blob/master/promotion/replication/red_button_replication.py).

If you want to create a similar figure as the one in the project description proving the convergence of the mathematical model of the expected value and the discounted utility, please change or leave the settings in [here](https://github.com/OpenSourceEconomics/ruspy/blob/master/promotion/simulation/init.yml) and then run the script for the [plot](https://github.com/OpenSourceEconomics/ruspy/blob/master/promotion/simulation/figure_1.py).
