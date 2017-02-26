:orphan:

.. _whats_new:

*****************************
What's New in Halotools v0.5?
*****************************

All Halotools official releases include minor bug-fixes and performance enhancements. All such changes appear in the :ref:`changelog`, and can also be reviewed reviewed in detail by filtering the `GitHub Issues page <https://github.com/astropy/halotools/issues/>`_ for issues that have been tagged with Milestone matching the release you are interested in.

New Features
============

.. currentmodule:: halotools.empirical_models

Abundance Matching and Age Matching Functions
---------------------------------------------

The `conditional_abunmatch` function provides a Numpy-based kernel for mapping galaxy properties onto halo properties in such a way that the observed (one-point) distribution is exactly correct, and a correlation of variable strength is introduced between the galaxy and halo properties. This function is the core of the age matching technique used in `Hearin and Watson 2013 <https://arxiv.org/abs/1304.5557/>`_ to correlate galaxy color and halo age. The `conditional_abunmatch` function generalizes this technique to construct mappings between any two variables and any level of stochasticity in the mapping.


HOD models using subhalos for satellites
----------------------------------------

In HOD-stye models of the galaxy--halo connection, there is parametric freedom in the number of satellites assigned to a host halo. Once the satellite count in a halo is determined, the intra-halo positions and velocities must also be modeled in some way. One common parametric choice is to assume an NFW profile, as in `Zehavi et al 2005 <https://arxiv.org/abs/astro-ph/0408569/>`_, and related works. Alternatively, satellites may be assigned to randomly chosen dark matter particles in the halo, as in `Reid and White 2014 <https://arxiv.org/abs/1404.3742/>`_. Instead of dark matter particles, the `SubhaloPhaseSpace` class allows for satellites to placed onto subhalos in the host halo. As described in the documentation, `SubhaloPhaseSpace` allows users to preferentially select *which* subhalos are assigned satellites, e.g., most massive first.


Conditional Luminosity Function models
---------------------------------------

The Conditional Luminosity Function (CLF) is a closely related class of empirical models to HODs. Whereas the HOD only specifies the number of galaxies in a halo brighter than some threshold, the CLF specifies the luminosity function of galaxies in a halo brighter than some threshold (e.g., `van den Bosch et al 2012 <https://arxiv.org/abs/1206.6890/>`_, and related works). The :ref:`cacciato09_composite_model` is a composite CLF model introduced in Halotools v0.5 that implements the version of the CLF introduced in `Cacciato et al 2009 <https://arxiv.org/abs/0807.4932/>`_.



