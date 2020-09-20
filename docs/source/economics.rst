Economic Model & Calibration
==============================

Here the economic model of Rust (1987) is documented and two ways of calibrating
its parameters are introduced. The first one is the nested fixed point algorithm
(NFXP) initially suggested by Rust (1987) and the second one is mathematical
programming with equilibrium constraints (MPEC) based on Su and Judd (2012).
This builds the theoretic background for the estimation and simulation modules of ruspy.


The Economic Model
------------------

The model is set up as an infinite horizon regenerative optimal stopping problem. It
considers the dynamic decisions by a maintenance manager, Harold Zurcher, for a fleet of
buses. As the buses are all identical and the decisions are assumed to be independent
across buses, there are no indications of the bus in the following notation. Harold
Zurcher makes repeated decisions :math:`a` about their maintenance in order to maximize
his expected total discounted utility with respect to the expected mileage usage of the
bus. Each month :math:`t`, a bus arrives at the bus depot in state :math:`s_t = (x_t,
\epsilon_t(a_t))` containing the mileage since last engine replacement :math:`x_t` and
other signs of wear and tear plus decision specific information :math:`\epsilon_t(a_t)`.
Harold Zurcher is faced with the decision to either conduct a complete engine
replacement :math:`(a_t = 1)` or to perform basic maintenance work :math:`(a_t = 0)`.
The cost of maintenance :math:`c(x_t, \theta_1)` increases with the mileage state,
while the cost of replacement :math:`RC` remains constant. Notationwise
:math:`\theta_1` captures the structural parameters shaping the maintenance cost
function. In the case of an engine replacement, the mileage state is reset to zero.

The immediate utility of each action in month :math:`t` is assumed to be additively
separable and given by:

.. math::

    \begin{align}
    u(a_t, x_t, \theta_1, RC) + \epsilon_t(a_t) \quad \text{with} \quad u(a_t, x_t,
    \theta_1, RC) = \begin{cases}
    -RC - c(0, \theta_1)   & a_t = 1 \\
    -c(x_t, \theta_1) & a_t = 0.
    \end{cases}
    \end{align}


The objective of Harold Zurcher is to choose a strategy :math:`\pi` of all strategies
:math:`\Pi` to maximize the utility over infinite horizon and therefore the current
value in period :math:`t` and state :math:`s_t` is given by:

.. math::

    \begin{align} \tilde{v}^{\pi}(s_t) \equiv \max_{\pi\in\Pi}
    E^\pi\left[\sum^{\infty}_{i = t}  \delta^{i - t} u(a_t, x_t, \theta_1, RC)) +
    \epsilon_t(a_t) \right]. \end{align}

The discount factor :math:`\delta` weighs the utilities over all periods and therefore
captures the preference of utility in current and future time periods. As the model
assumes stationary utility, as well as stationary transition probabilities the future
looks the same, whether the agent is at time :math:`t` in state :math:`s` or at any
other time. Therefore the optimal decision in every period can be captures by the
Bellman equation:

.. math::

    \begin{equation}
    v_\theta(x_t, \epsilon_t) = \max_{a_t \in \{0,1\}} \biggl[u(x_t,
    a_t, \theta_1, RC) + \epsilon_t(a_t) + \delta EV_\theta(x_t, \epsilon_t,
    a_t)\biggr],
    \end{equation}

where

.. math::

    \begin{equation} EV_\theta(x_t, \epsilon_t, a_t) =
    \int \int v_\theta(\gamma, \eta) p(d\gamma, d\eta | x_t, \epsilon_t, a_t, \theta_2,
    \theta_3)
    \end{equation}

and :math:`\theta` captures the parametrization of the model given by :math:`\{\delta,
RC, \theta_1, \theta_2, \theta_3 \}`. Thus Harold Zurcher makes his decision in light of
uncertainty about next month's state realization captured by the their conditional
distribution :math:`p(x_{t+1}, \epsilon_{t+1} | x_t, \epsilon_t, a_t, \theta_2,
\theta_3)`.

Rust (1987) imposes conditional independence between the probability densities of the
observable and unobservable state variables, i.e.

.. math::

    \begin{equation}
    p(x_{t+1}, \epsilon_{t+1}| x_t, a_t, \epsilon_t, \theta_2, \theta_3) = p(x_{t+1}|
    x_t, a_t, \theta_3) p(\epsilon_{t+1}|\epsilon_t, \theta_2)
    \end{equation}

and furthermore assumes that the unobservables :math:`\epsilon_t(a_t)` are independent
and identically distributed according to an extreme value distribution with mean zero
and scale parameter one, i.e.:

.. math::

     \begin{equation}
      p(\epsilon_{t+1}| \theta_2) = \exp\{-\epsilon_{t+1} + \theta_2\}
      \exp\{-\exp\{-\epsilon_{t+1} + \theta_2 \}\}
      \end{equation}

where :math:`\theta_2 = 0.577216`, i.e. the Euler-Mascheroni constant.

Rust (1988) shows that these two assumptions, together with the additive separability
between the observed and unobserved state variables in the immediate utilities, imply
that :math:`EV_\theta` is a function independent of :math:`\epsilon_t` and the unique
fixed point of a contraction mapping on the reduced space of all state action pairs
:math:`(x,a)`. Furthermore, the regenerative property of the process yields for all
states :math:`x`, that the expected value of replacement corresponds to the expected
value of maintenance in state :math:`0`, i.e. :math:`EV_\theta(x, 1) = EV_\theta(0,
0)`. Thus :math:`EV_\theta` is the unique fixed point on the observed mileage state
:math:`x` only. Therefore in the following :math:`EV_\theta(x)` refers to
:math:`EV_\theta(x, 0)`. The contraction mapping is then given by:

.. math::

    \begin{equation}
      EV_\theta(x) = \sum_{x' \in X} p(x'|x, \theta_3) \log \sum_{a \in \{0, 1\}} \exp(
      u(x' , a, \theta_1, RC) + \delta EV_\theta(x'))
      \end{equation}

This gives rise to the shorthand notation of the above formula:

.. math::
    \begin{equation}
      EV_\theta(x) = T_\theta(EV_\theta(x))
    \end{equation}

In addition, the conditional choice probabilities :math:`P(a| x, \theta)` have a
closed-form solution given by the multinomial logit formula (McFadden, 1973):

.. math::

    \begin{equation}
    P(a|x, \theta) = \frac{\exp(u(a, x, RC, \theta_1) + \delta EV_\theta((a-1) \cdot
    x))}{\sum_{i \in \{0, 1\}} \exp(u(i, x, RC, \theta_1) + \delta EV_\theta((i - 1)x))}
    \end{equation}

These closed form solutions allow to estimate the structural parameters driving
Zurcher's decisions. Given the data :math:`\{a_0, ....a_T, x_0, ..., x_T\}` for a
single bus, one can form the likelihood function :math:`l^f(a_1, ..., a_T, x_1, ....,
x_T | a_0, x_0, \theta)` and estimate the parameter vector :math:`\theta` by maximum
likelihood. Rust (1988) proofs that this function has due to the conditional
independence assumption a simple form:

.. math::

    \begin{equation}
    l^f(a_1, ..., a_T, x_1, ...., x_T | a_0, x_0, \theta) = \prod_{t=1}^T P(a_t|x_t,
    \theta) p(x_t| x_{t-1}, a_{t-1}, \theta_3)
    \end{equation}


Therefore the estimation can be split into two separate partial likelihood functions,
given by:

.. math::

    \begin{equation}
    l^1(a_1, ..., a_T, x_1, ...., x_T | a_0, x_0, \theta_3) = \prod_{t=1}^T p(x_t|
    x_{t-1}, a_{t-1}, \theta_3)
    \end{equation}

and

.. math::

    \begin{equation}
      l^2(a_1, ..., a_T, x_1, ...., x_T | \theta) = \prod_{t=1}^T P(a_t|x_t, \theta)
    \end{equation}


Nested Fixed Point Algorithm
----------------------------

The calibration strategy employed by Rust (1987) involves handing the logarithm
of the above :math:`l^f(a_1, ..., a_T, x_1, ...., x_T | a_0, x_0, \theta)`
to an unconstrained optimization algorithm.
Rust originally suggests a polyalgorithm of the BHHH and the BFGS for this purpose.
This optimizer fixes a guess of the structural parameter vector :math:`\hat\theta`
for which the unique fixed point of the economic model is found.
Through this the conditional choice probabilities :math:`P(a|x, \hat\theta)`
are obtained which in turn are used to evaluate the log likelihood function.
On the basis of this, the optimization algorithm comes up with a new guess for
the structural parameters and the procedure starts over until a certain
convergence criteria is met.

The algorithm consequently corresponds to solving the following optimization
problem in an outer loop:

.. math::

    \begin{equation}
      \max_{\theta} \; log \; l^f(a_1, ..., a_T, x_1, ...., x_T | a_0, x_0, \theta)
    \end{equation}

while finding the unique fixed point of :math:`EV_\theta(x) = T_\theta(EV_\theta(x))`
in an inner loop for a given parameter guess produced in the outer loop.


Mathematical Programming with Equilibrium Constraints
-----------------------------------------------------

The approach developed by Su and Judd (2012) casts this unconstrained nested problem
into a constrained optimization problem. For this they plug the conditional
choice probabilities :math:`P(a|x, \theta)` into the likelihood function :math:`l^f(.)`:

.. math::

    \begin{equation}
    \begin{split}
    l^f_{aug}(. | a_0, x_0, \theta, EV) = & \prod_{t=1}^T \frac{
    \exp(u(a, x, RC, \theta_1) + \delta EV((a-1) \cdot x))}{
    \sum_{a \in \{0, 1\}} \exp(u(a, x, RC, \theta_1) + \delta EV((a - 1)x))} \\
    \\
    & \times p(x_t| x_{t-1}, a_{t-1}, \theta_3).
    \end{split}
    \end{equation}

They coin the term augmented likelihood function for :math:`l^f_{aug}`.
The particular feature now is that the likelihood depends explicitly on both the
structural parameter vector :math:`\theta` as well as the choice of :math:`EV`.
In order to ensure that guesses of both vectors are consistent
in the spirit of the economic model, the contraction mapping of the expected value
function is imposed as a constraint to the augmented likelihood function.
Consequently, the calibration problem boils down to a constrained optimization
looking like the following:

.. math::

    \begin{equation}
      \max_{(\theta, EV)} \; log \; l^f_{aug}(a_1, ..., a_T, x_1, ...., x_T | a_0, x_0, \theta, EV) \\
      \text{subject to } \; EV = T(EV, \theta).
    \end{equation}

The constraints are generally nonlinear functions which restricts the use of
optimization algorithms. An non-exhaustive list of optimizers that can handle
the above problem are the commercial KNITRO (see Byrd et al. (2006)), as well
as the open source IPOPT (see Wächter and Biegler (2006)) and the SLSQP (see
Kraft (1994)) provided by NLOPT.


The Implied Demand Function
---------------------------

Rust (1987) shortly describes a way to uncover an implied demand function of engine
replacement from his model and its estimated parameters. Theoretically, for Harold
Zurcher the random annual implied demand function takes the following form:

.. math::

    \begin{equation*}
      \tilde{d}(RC) = \sum_{t=1}^{12} \sum_{m=1}^{M} \tilde{a}^m_t
    \end{equation*}

where :math:`\tilde{a}^m_t` is the replacement decision for a certain bus :math:`m`
in a certain month :math:`t` derived from the process {:math:`a^m_t, x^m_t`}.

For convenience I will drop the index for the bus in the following. Its probability
distribution is therefore the result of the process described by
:math:`P(a_t|x_t; \theta)p(x_t|x_{t-1}, a_{t-1}; \theta_3)`. For simplification
Rust actually derives the expected demand function :math:`d(RC)=E[\tilde{d}(RC)]`.
Assuming that :math:`\pi` is the long-run stationary distribution of the process
{:math:`a_t, x_t`} and that the observed initial state {:math:`a_0, x_0`} is in
the long run equilibrium, :math:`\pi` can be described by the following functional
equation:

.. math::

    \begin{equation}
      \pi(x, a; \theta) = \int_{y} \int_{j} P(a|x; \theta)p_3(x|y, j, \theta_3)
      \pi(dy, dj; \theta).
    \end{equation}

Further assuming that the processes of {:math:`a_t, x_t`} are independent across
buses the annual expected implied demand function boils down to:

.. math::

    \begin{equation}
      d(RC) = 12 M \int_{0}^{\infty} \pi(dx, 1; \theta).
    \end{equation}

Given some estimated parameters :math:`\hat\theta` from calibrating the Rust Model
and parametrically varying :math:`RC` results in different estimates of
:math:`P(a_t|x_t; \theta)p(x_t|x_{t-1}, a_{t-1}; \theta_3)` which in turn affects
the probability distribution :math:`\pi` which changes the implied demand.
In the representation above it is clearly assumed that both the mileage state
:math:`x` and the replacement decision :math:`a` are continuous. The replacement
decision is actually discrete, though, and the mileage state has to be discretized
again which in the end results in a sum representation of the function :math:`d(RC)`
that is taken to calculate the expected annual demand.

This demand function can be calculated in the ruspy package for a given
parametrization of the model. A description how to do this can be found in
:ref:`demand_function_calculation`.
