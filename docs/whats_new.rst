:orphan:

.. _whats_new:

*****************************
What's New in Halotools v0.5?
*****************************

Halotools ``v0.5`` will soon be available for pip installation. The main changes to the code are summarized in the sections below.


New Models
============

.. currentmodule:: halotools.empirical_models

.. _abundance_matching_v05:

Abundance Matching and Age Matching functions
---------------------------------------------

The `conditional_abunmatch` function provides a Numpy-based kernel for mapping galaxy properties onto halo properties in such a way that the observed (one-point) distribution is exactly correct, and a correlation of variable strength is introduced between the galaxy and halo properties. This function is the core of the age matching technique used in `Hearin and Watson 2013 <https://arxiv.org/abs/1304.5557/>`_ to correlate galaxy color and halo age. The `conditional_abunmatch` function generalizes this technique to construct mappings between any two variables and any level of stochasticity in the mapping.


.. _subhalo_phase_space_v05:

HOD models using subhalos for satellites
----------------------------------------

In HOD-stye models of the galaxy--halo connection, there is parametric freedom for the positions and velocities of satellite galaxies within their halos. One common parametric choice is to assume an NFW profile, as in `Zehavi et al 2005 <https://arxiv.org/abs/astro-ph/0408569/>`_, and related works. Alternatively, satellites may be assigned to randomly chosen dark matter particles in the halo, as in `Reid and White 2014 <https://arxiv.org/abs/1404.3742/>`_. Instead of dark matter particles, the `~halotools.empirical_models.SubhaloPhaseSpace` class allows for satellites to be placed onto subhalos in the host halo. See :ref:`subhalo_phase_space_model_tutorial` for further information.


.. _clf_models_v05:

Conditional Luminosity Function models
---------------------------------------

The Conditional Luminosity Function (CLF) is a closely related class of empirical models to HODs. Whereas the HOD only specifies the number of galaxies in a halo brighter than some threshold, the CLF specifies the luminosity function of galaxies in a halo brighter than some threshold (e.g., `van den Bosch et al 2012 <https://arxiv.org/abs/1206.6890/>`_, and related works). The :ref:`cacciato09_composite_model` is a composite CLF model that implements the version of the CLF introduced in `Cacciato et al 2009 <https://arxiv.org/abs/0807.4932/>`_.


.. _zu_mandelbaum_v05:

Zu & Mandelbaum (2015/2016)
---------------------------
There are two new HOD models based on `Zu and Mandelbaum 2015 <https://arxiv.org/abs/1505.02781/>`_
and the follow-up paper `Zu and Mandelbaum 2016 <https://arxiv.org/abs/1509.06758/>`_.
The first model describes an HOD fit to z=0 SDSS clustering and lensing of stellar mass threshold galaxy
samples, the second model additionally predicts whether each model galaxy is quiescent or star-forming. See :ref:`zu_mandelbaum15_composite_model` and :ref:`zu_mandelbaum16_composite_model` tutorials on these models.


.. _biased_nfw_phase_space_v05:

Satellite profiles with biased values
-------------------------------------

The `BiasedNFWPhaseSpace` class provides modeling for the distribution of satellite galaxies orbiting in their host halos, where the NFW concentration governing the satellites is allowed to differ from host halo concentration. Galaxy concentration bias is permitted to vary as a function of mass with `BiasedNFWPhaseSpace`, while the `SFRBiasedNFWPhaseSpace` class permits dependence of galaxy concentration on both host mass and whether or not the model galaxy is quiescent.


New Mock Observations
======================

.. currentmodule:: halotools.mock_observables

.. _galaxy_lensing_v05:

Calculating galaxy-galaxy lensing in a hydro simulation
--------------------------------------------------------

The `delta_sigma` function has had a complete overhaul, including a change to the function signature. The new implementation is faster and more accurate than the previous version, and now supports calculating the lensing signal for cases where the simulation particles have variable mass, such as hydro simulations or boxes with massive neutrinos.

Calculating galaxy-galaxy lensing from pre-computed pairs
-----------------------------------------------------------
There is also a new `delta_sigma_from_precomputed_pairs` that allows users to pre-compute the mass surrounding each model galaxy and then compute :math:`\Delta\Sigma` directly from an input mask; for cases where the *candidate* positions of galaxies are known in advance, the `delta_sigma_from_precomputed_pairs` will generally improve runtimes for calculating :math:`\Delta\Sigma` by orders of magnitude.


Calculating the HOD directly from a mock
-----------------------------------------

The `hod_from_mock` function provides a convenient interface for calculating
the mean number of galaxies per halo as a function of, for example, halo mass.
The API permits arbitrary subsampling; supports any input independent variable,
and is performant enough to apply to MCMC applications of, e.g., group statistics.
Thus the `hod_from_mock` function could be used for example,
to calculate :math:`\langle N_{\rm cen}|M_{\rm vir}\rangle`,
:math:`\langle N_{\rm red}|V_{\rm max}\rangle`,
or :math:`\langle N_{\rm blue-sat}|M_{\rm group}\rangle`.


Further Details
================

All Halotools official releases include minor bug-fixes and performance enhancements. All such changes appear in the :ref:`changelog`, and can also be reviewed in detail by filtering the `GitHub Issues page <https://github.com/astropy/halotools/issues/>`_ for issues that have been tagged with the Milestone matching the release you are interested in.



