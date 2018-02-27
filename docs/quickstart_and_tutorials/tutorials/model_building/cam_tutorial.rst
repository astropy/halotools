:orphan:

.. _cam_tutorial:

**********************************************************************
Tutorial on Conditional Abundance Matching
**********************************************************************

Conditional Abundance Matching (CAM) is a technique that you can use to
model a variety of correlations between galaxy and halo properties,
such as the dependence of galaxy quenching upon both halo mass and
halo formation time. This tutorial explains CAM by applying
the technique to a few different problems.
Each of the following worked examples are independent from one another,
and illustrate the range of applications of the technique.


Basic Idea
=================

Forward-modeling the galaxy-halo connection requires specifying
some statistical distribution of the galaxy property being modeled,
so that Monte Carlo realizations can be drawn from the distribution.
The most convenient distribution to use for this purpose is the cumulative
distribution function (CDF), :math:`{\rm CDF}(x) = {\rm Prob}(< x).`
Once the CDF is specified, you only need to generate
a realization of a random uniform distribution and pass those draws to the
CDF inverse,  :math:`{\rm CDF}^{-1}(p),` which evaluates to the variable
:math:`x` being painted on the model galaxies.

CAM introduces correlations between the
galaxy property :math:`x` and some halo property :math:`h,`
without changing :math:`{\rm CDF}(x)`. Rather than evaluating :math:`{\rm CDF}^{-1}(p)`
with random uniform variables,
instead you evaluate with :math:`p = {\rm CDF}(h) = {\rm Prob}(< h),`
introducing a monotonic correlation between :math:`x` and :math:`h`.

The function `~halotools.empirical_models.noisy_percentile` can be used to
add controllable levels of noise to :math:`p = {\rm CDF}(h).`
This allows you to control the correlation coefficient
between :math:`x` and :math:`h,`
always exactly preserving the 1-point statistics of the output distribution.


The "Conditional" part of CAM is that this technique naturally generalizes to
introduce a galaxy property correlation while holding some other property fixed.
Age Matching in `Hearin and Watson 2013 <https://arxiv.org/abs/1304.5557/>`_
is an example of this: the distribution :math:`{\rm Prob}(<SFR\vert M_{\ast})`
is modeled by correlating draws from the observed distribution with
:math:`{\rm Prob}(<\dot{M}_{\rm sub}\vert M_{\rm sub})` in a simulation,
so that galaxies which have
large SFR for their stellar mass are associated with subhalos that have
large mass accretion rates for their mass dark matter mass.

Each of the sections below illustrates a different application of the same underlying method.
Each section has an accompanying annotated Jupyter notebook with the code used to generate the plots.

Satellite Galaxy Quenching Gradients
=====================================

Observations indicate that satellite galaxies are redder in the
inner regions of their host dark matter halos. One way to model this phenomenon is to use CAM
to correlate the quenching probability with host-centric position.
For example, `Zu and Mandelbaum 2016 <https://arxiv.org/abs/1509.06758/>`_ model satellite
quenching with a simple analytical function :math:`{\rm Prob(\ quenched}\ \vert\ M_{\rm host})`,
where :math:`M_{\rm host}` is the dark matter mass of the satellite's parent halo.
For a standard implementation of this model, you can draw from a random uniform number generator
of the unit interval, and evaluate whether those draws are above or below :math:`{\rm Prob(\ quenched)}`.

Alternatively, to implement CAM you would compute
:math:`p={\rm Prob(< r/R_{vir}}\ \vert\ M_{\rm host})` for each simulated subhalo,
and then evaluate whether each :math:`p`
is above or below :math:`{\rm Prob(\ quenched}\ \vert\ M_{\rm host})`.
This technique lets you generate a series of mocks with exactly the same
:math:`{\rm Prob(\ quenched}\ \vert\ M_{\rm host})`,
but with tunable levels of quenching gradient, ranging from zero gradient
to the statistical extrema.
The `~halotools.utils.sliding_conditional_percentile` function can be used to
calculate :math:`p={\rm Prob(< r/R_{vir}}\ \vert\ M_{\rm host}).`


The plot below demonstrates three different mock catalogs made with CAM in this way.
The left hand plot shows how the quenched fraction of satellites varies
with intra-halo position. The right hand plot confirms that all three mocks have
statistically indistinguishable "halo mass quenching", even though their gradients
are very different.

.. image:: /_static/quenching_gradient_models.png

The next plot compares the 3d clustering between these models.

.. image:: /_static/quenching_gradient_model_clustering.png

For implementation details, the code producing these plots
can be found in the following Jupyter notebook:

    **halotools/docs/notebooks/galcat_analysis/intermediate_examples/quenching_gradient_tutorial.ipynb**





