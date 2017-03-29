.. _zu_mandelbaum15_composite_model:

*********************************************
Zu & Mandelbaum et al. (2015) Composite Model
*********************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes the basic behavior of
the ``zu_mandelbaum15`` composite HOD model. To see how this composite
model is built by the `~halotools.empirical_models.PrebuiltHodModelFactory` class,
see `~halotools.empirical_models.zu_mandelbaum15_model_dictionary`.

Overview of the Zu & Mandelbaum et al. (2015) Model Features
=============================================================
This HOD-style model is based on `Zu & Mandelbaum et al (2015) <https://arxiv.org/abs/1505.02781/>`_.
The behavior of this model is governed by the
`Behroozi et al. (2010) <https://arxiv.org/abs/1001.0015/>`_, but with parameters
that have been refit to z=0 data, and with scatter that is allowed to
vary with halo mass. The occupation statistics have the same functional
form as the :ref:`leauthaud11_composite_model` introduced in
`Leauthaud et al (2011) <https://arxiv.org/abs/1103.2077/>`_.

In this model, there are two populations, centrals and satellites.
Central occupation statistics are given by a nearest integer distribution
with first moment given by an `~scipy.special.erf` function; the class governing this
behavior is `~halotools.empirical_models.ZuMandelbaum15Cens`.
Central galaxies are assumed to reside at the exact center of the host halo;
the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`.

Satellite occupation statistics are given by a Poisson distribution
with first moment given by a power law that has been truncated at the low-mass end;
the class governing this behavior is `~halotools.empirical_models.ZuMandelbaum15Sats`;
satellites in this model follow an (unbiased) NFW profile, as governed by the
`~halotools.empirical_models.NFWPhaseSpace` class.

Building the Zu & Mandelbaum et al. (2015) Model
=================================================
You can build an instance of this model using the
`~halotools.empirical_models.PrebuiltHodModelFactory` class as follows:

>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> model = PrebuiltHodModelFactory('zu_mandelbaum15')


Customizing the Zu & Mandelbaum et al. (2015) Model
=====================================================

There are two keyword arguments you can use to customize
the instance returned by the factory:

First, the ``threshold`` keyword argument pertains to the minimum
stellar mass of the galaxy sample, in solar mass units with h=1:

>>> model = PrebuiltHodModelFactory('zu_mandelbaum15', threshold=10.75)

Second, the ``prim_haloprop_key`` keyword argument determines which
halo mass definition will be used to populate a mock with this model.
You are free to choose any halo mass definition you like, but you should
be aware that the best-fit parameters of the Zu & Mandelbaum model are
based on ``halo_m200m``:

>>> model = PrebuiltHodModelFactory('zu_mandelbaum15', threshold=11, haloprop_key='halo_mvir')

The `Colossus python package <https://bitbucket.org/bdiemer/colossus/>`_
written by Benedikt Diemer can be used to
convert between different halo mass definitions. This may be useful if you wish to use an
existing halo catalog for which the halo mass definition you need is unavailable.

As described in :ref:`altering_param_dict`, you can always change the model parameters
after instantiation by changing the values in the ``param_dict`` dictionary. For example,

>>> model.param_dict['alphasat'] = 1.1

The above line of code changes the power law slope between
halo mass and satellite occupation number, :math:`\langle N_{\rm sat} \rangle \propto M_{\rm halo}^{\alpha}`.
See :ref:`zu_mandelbaum15_parameters` for a description of all parameters of this model.

Populating Mocks and Generating Zu & Mandelbaum et al. (2015) Model Predictions
================================================================================

As with any Halotools composite model, the model instance
can populate N-body simulations with mock galaxy catalogs.
In the following, we'll show how to do this
with fake simulation data via the ``halocat`` argument.

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> model = PrebuiltHodModelFactory('zu_mandelbaum15')
>>> model.populate_mock(halocat)  # doctest: +SKIP

See `ModelFactory.populate_mock` for information about how to
populate your model into different simulations.
See :ref:`galaxy_catalog_analysis_tutorial` for a sequence of worked examples
on how to use the `~halotools.mock_observables` sub-package
to study a wide range of astronomical statistics predicted by your model.

Studying the Zu & Mandelbaum et al. (2015) Model Features
===========================================================

In addition to populating mocks, the ``zu_mandelbaum15`` model also gives you access to
its underlying analytical relations. Here are a few examples:

>>> import numpy as np
>>> halo_mass = np.logspace(11, 15, 100)

To compute the mean number of each galaxy type as a function of halo mass:

>>> mean_ncen = model.mean_occupation_centrals(prim_haloprop=halo_mass)
>>> mean_nsat = model.mean_occupation_satellites(prim_haloprop=halo_mass)

To compute the mean stellar mass of central galaxies as a function of halo mass:

>>> mean_sm_cens = model.mean_stellar_mass_centrals(prim_haloprop=halo_mass)

Now suppose you wish to know the mean halo mass of a central galaxy with known stellar mass:

>>> stellar_mass = np.logspace(9, 12, 100)
>>> inferred_halo_mass = model.mean_halo_mass_centrals(stellar_mass)

.. _zu_mandelbaum15_parameters:

Parameters of the Zu & Mandelbaum et al. (2015) model
======================================================

The best way to learn what the parameters of a model do is to
just play with the code: change parameter values, make plots of how the
underying analytical relations vary, and also of how the
mock observables vary. Here we just give a simple description of the meaning
of each parameter. You can also refer to the original publications
`Leauthaud et al (2011) <https://arxiv.org/abs/1103.2077/>`_,
`Behroozi et al. (2010) <https://arxiv.org/abs/1001.0015/>`_
and `Zu & Mandelbaum et al (2015) <https://arxiv.org/abs/1505.02781/>`_
for more detailed descriptions of the meaning of each parameter.

To see how the following parameters are implemented, see `ZuMandelbaum15Cens.mean_occupation` and `Behroozi10SmHm.mean_stellar_mass`.

* param_dict['smhm_m0'] - Characteristic stellar mass at redshift-zero in :math:`\langle M_{\ast} \rangle(M_{\rm halo})`.

* param_dict['smhm_m1'] - Characteristic halo mass at redshift-zero in :math:`\langle M_{\ast} \rangle(M_{\rm halo})`.

* param_dict['smhm_beta'] - Low-mass slope at redshift-zero of :math:`\langle M_{\ast} \rangle(M_{\rm halo})`.

* param_dict['smhm_delta'] - High-mass slope at redshift-zero of :math:`\langle M_{\ast} \rangle(M_{\rm halo})`.

* param_dict['smhm_gamma'] - Transition between low- and high-mass behavior at redshift-zero of :math:`\langle M_{\ast} \rangle(M_{\rm halo})`.

* param_dict['u'smhm_sigma'] - Normalization of the log-normal scatter about the mean relation :math:`\langle M_{\ast} \rangle(M_{\rm halo})`.

* param_dict['u'smhm_sigma_slope'] - Parameter controlling halo mass-dependence of the log-normal scatter.

To see how the following parameters are implemented, see `ZuMandelbaum15Sats.mean_occupation` and `Behroozi10SmHm.mean_stellar_mass`.

* param_dict['alphasat'] - Power law slope of the relation between halo mass and :math:`\langle N_{\rm sat} \rangle`.

* param_dict['betasat'] - Controls the amplitude of the power law slope :math:`\langle N_{\rm sat} \rangle`.

* param_dict['bsat'] - Also controls the amplitude of the power law slope :math:`\langle N_{\rm sat} \rangle`.

* param_dict['betacut'] - Controls the low-mass cutoff of :math:`\langle N_{\rm sat} \rangle`.

* param_dict['bcut'] - Also controls the low-mass cutoff of :math:`\langle N_{\rm sat} \rangle`.


