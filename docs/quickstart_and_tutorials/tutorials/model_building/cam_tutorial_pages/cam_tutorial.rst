.. _cam_tutorial:

**********************************************************************
Tutorial on Conditional Abundance Matching
**********************************************************************

Conditional Abundance Matching (CAM) is a technique that you can use to
model a variety of correlations between galaxy and halo properties,
such as the dependence of galaxy quenching upon both halo mass and
halo formation time, or the dependence of galaxy disk size upon halo spin.
This tutorial explains CAM by applying the technique to a few different problems.


.. _cam_basic_idea:

Basic Idea Behind CAM
======================

CAM is designed to answer questions of the following form:
*does halo property A correlate with galaxy property B?*
The Halotools approach to answering such questions is via forward modeling:
a mock universe is created in which the A--B correlation exists;
comparing the mock universe to the real one allows you to evaluate the
success of the A--B correlation hypothesis.

Forward-modeling the galaxy-halo connection requires specifying
some statistical distribution of the galaxy property being modeled,
so that Monte Carlo realizations can be drawn from the distribution.
CAM uses the most ubiquitous approach to generating Monte Carlo realizations,
*inverse transformation sampling,* in which the statistical distribution
is specified in terms of the cumulative distribution function (CDF),
:math:`{\rm CDF}(z) \equiv {\rm Prob}(< z).`
Briefly, the way this work is that once you specify the CDF,
you only need to generate a realization of a random uniform distribution,
and pass the values of that realization to the CDF inverse,  :math:`{\rm CDF}^{-1}(p),`
which evaluates to the variable :math:`z` being painted on the model galaxies.
See the `Transformation of Probability tutorial <https://github.com/jbailinua/probability/>`_
for pedagogical derivations associated with inverse transformation sampling,
and the `~halotools.utils.monte_carlo_from_cdf_lookup` function
for a convenient one-liner syntax.

In ordinary applications of inverse transformation sampling,
the use of a random uniform variable guarantees
that the output variables :math:`z` will be distributed according to
:math:`{\rm Prob}(z),` and that each individual :math:`z` will be purely stochastic.
CAM generalizes this common technique so that :math:`{\rm Prob}(z)`
is still recovered exactly, and moreover :math:`z` exhibits residual correlations
with some other variable, :math:`h`. Operationally, the way this works is that
rather than evaluating :math:`{\rm CDF}^{-1}(p)` with random uniform variables,
instead you evaluate with :math:`p = {\rm CDF}(h) = {\rm Prob}(< h),`
introducing a monotonic correlation between :math:`z` and :math:`h`.
In most applications, :math:`h` is some halo property like mass accretion rate,
and :math:`z` is some galaxy property like star-formation rate.
In this way, the galaxy property you paint on to your halos will
trace the distribution :math:`{\rm Prob}(z)`, such that above-average
values of :math:`z` will be painted onto halos with above average values of
:math:`h`, and conversely.

Finally, the "Conditional" part of CAM is that this technique naturally generalizes to
introduce a galaxy property correlation while holding some other property fixed.
For example, at fixed stellar mass, it is natural to hypothesize that
star-forming galaxies live in halos that are rapidly accreting mass,
and that quiescent galaxies live in halos that have already built up most of their mass.
In this kind of CAM application, we have:
:math:`{\rm Prob}(z)\rightarrow{\rm Prob}(<SFR\vert M_{\ast})`,
and :math:`{\rm Prob}(h)\rightarrow{\rm Prob}(<\dot{M}_{\rm sub}\vert M_{\rm sub})`.
That is, SFR at fixed stellar mass is hypothesized to correlate with
halo accretion rate at fixed (sub)halo mass.


Quick Demonstration
=====================================

This tutorial gives many different examples of how to apply CAM when modeling
the galaxy-halo connection, including several different variations of the technique.
Before going into the details, we'll first give a brief demo of a simple example.

Suppose we live in a very simple universe in which every galaxy lives on the star-forming
sequence, with SFR distributed like a log-normal. Let's create a Monte Carlo realization
of such a galaxy population:

.. code:: python

    import numpy as np
    ngals = int(1e4)
    log_mstar = np.random.uniform(10, 12, ngals)
    galaxy_mstar = 10**log_mstar
    mean_log_sfr = np.interp(log_mstar, [10, 11, 12], [0, 1, 2])
    log_sfr = np.random.normal(loc=mean_log_sfr, scale=0.2, size=ngals)
    galaxy_sfr = 10**log_sfr

Now let's suppose that some simple stellar-to-halo mass relation accurately
maps stellar mass onto dark matter halos, and that SFR is just determined by the
CAM hypothesis that halos with large accretion rates host galaxies with
large star-formation rates. First, we grab a halo catalog and map stellar mass onto it:

.. code:: python

    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname='bolplanck', redshift=0)
    halo_mass = halocat.halo_table['halo_mvir']
    from halotools.empirical_models import Behroozi10SmHm
    model = Behroozi10SmHm(redshift=0)
    halo_mstar = model.mc_stellar_mass(prim_haloprop=halo_mass)
    halo_acc_rate = halocat.halo_table['halo_dmvir_dt_100myr']

Now we show how to map star-formation rate onto these halos using CAM.

.. code:: python

    from halotools.empirical_models import conditional_abunmatch
    nwin = 101
    halo_sfr = conditional_abunmatch(halo_mstar, halo_acc_rate, galaxy_mstar, galaxy_sfr, nwin)

The rest of this tutorial will help you understand the details behind this example, as well as other applications that involve different variations of the technique.


Worked Examples
========================================

CAM applications take on a slightly different form depending on whether or not you can analytically evaluate the inverse CDF of the galaxy property you are modeling. So the examples in this tutorial are divided into two categories:

Modeling a galaxy property with a simple analytical distribution
----------------------------------------------------------------

Many galaxy properties are well-described by straightforward statistical distributions.
For example, if your distribution can be approximated by a log-normal or power law,
then the functions implemented in `scipy.stats` can be used to analytically evaluate
the inverse CDF. Each of the following tutorials gives an example of how to apply CAM
in such a situation:

.. toctree::
   :maxdepth: 1

   cam_decorated_clf
   cam_disk_bulge_ratios
   cam_quenching_gradients


Modeling a galaxy property without a known analytical distribution
------------------------------------------------------------------

In many cases, evaluating the inverse CDF analytically is intractible,
and it can only be numerically tabulated from some sample data. The examples
below illustrate a few CAM applications for such galaxy properties:


.. toctree::
   :maxdepth: 1

   cam_complex_sfr
