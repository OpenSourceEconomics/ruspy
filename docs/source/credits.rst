Contributing
============

Team
----

----
BDFL
----

`Philipp Eisenhauer <https://github.com/peisenha>`_

----------------
Development Lead
----------------

`Maximilian Blesch <https://github.com/MaxBlesch>`_

------------
Contributors
------------

`Sebastian Becker <https://github.com/sebecker>`_, `Pascal Heid <https://github
.com/Pascalheid>`_, `Viktoria Kleinschmidt <https://github.com/viktoriakleinschmidt>`_



Master Theses
-------------

Below you find a list with past Master Theses, that used ruspy. If you think of using
ruspy in your Master Thesis, please reach out to us and view the issues with a
Master-Thesis tag on `github. <https://github.com/OpenSourceEconomics/ruspy/issues>`_

------------------------------------------------------
Decision rule performance under model misspecification
------------------------------------------------------
by `Maximilian Blesch <https://github.com/MaxBlesch>`_

I incorporate techniques from distributionally robust optimization into a dynamic
investment model. This allows to explicitly account for ambiguity in the decision-
making process. I outline an economic, mathematical, and computational model
to study the seminal bus replacement problem (Rust, 1987) under potential model
misspecification. I specify ambiguity sets for the transition dynamics of the model.
These are based on empirical estimates, statistically meaningful, and computation-
ally tractable. I analyze alternative policies in a series of computational exper-
iments. I find that, given the structure of the model and the available data on
past transitions, a policy simply ignoring model misspecification often outperforms
its alternatives that are designed to explicitly account for it.


---------------------------------------------------------------------------------
Mathematical Programming with Equilibrium Constraints: An Uncertainty Perspective
---------------------------------------------------------------------------------
by `Pascal Heid <https://github.com/Pascalheid>`_

This thesis explores to which extent the Nested Fixed Point Algorithm (NFXP) as
suggested by Rust (1987) differs from the Mathematical Programming with Equilibrium
Constraints as introduced by Su and Judd (2012) by revisiting the Optimal Bus Engine
Replacement Problem posed by the previous author. While previous studies focus on
quantitative measures of speed and convergence rate, my focus lies on how the two
approaches actually recover the true model when the simulation setup is less clean
and more closely to what applied researchers typically face. For this comparison I
draw on some techniques from the Uncertainty Quantification literature. I run a large
scale simulation study in which I compare the two approaches among different model
specifications by checking how accurate their counterfactual demand level predictions
are. I can show that under realistic circumstances, the two approaches can yield
considerably different predictions suggesting that they should be regarded as
complements rather than competitors.
