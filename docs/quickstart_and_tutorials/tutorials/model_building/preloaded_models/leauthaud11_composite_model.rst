.. _leauthaud11_composite_model:

*********************************************
Leauthaud et al. (2011) Composite Model
*********************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes the basic behavior of
the ``leauthaud11`` composite HOD model. To see how this composite
model is built by the `~halotools.empirical_models.PrebuiltHodModelFactory` class,
see `~halotools.empirical_models.leauthaud11_model_dictionary`.

Overview of the Leauthaud et al. (2011) Model Features
========================================================
This HOD-style model is based on Leauthaud et al. (2011), arXiv:1103.2077.
The behavior of this model is governed by an assumed underlying stellar-to-halo-mass relation.

There are two populations, centrals and satellites.
Central occupation statistics are given by a nearest integer distribution
with first moment given by an ``erf`` function; the class governing this
behavior is `~halotools.empirical_models.Leauthaud11Cens`.
Central galaxies are assumed to reside at the exact center of the host halo;
the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`.

Satellite occupation statistics are given by a Poisson distribution
with first moment given by a power law that has been truncated at the low-mass end;
the class governing this behavior is `~halotools.empirical_models.Leauthaud11Sats`;
satellites in this model follow an (unbiased) NFW profile, as governed by the
`~halotools.empirical_models.NFWPhaseSpace` class.

Building the Leauthaud et al. (2011) Model
============================================
You can build an instance of this model using the
`~halotools.empirical_models.PrebuiltHodModelFactory` class as follows:

>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> model = PrebuiltHodModelFactory('leauthaud11')


Customizing the Leauthaud et al. (2011) Model
=================================================

There are two keyword arguments you can use to customize
the instance returned by the factory:

First, the ``threshold`` keyword argument pertains to the minimum
stellar mass of the galaxy sample, in logarithmic units of Msun in h=1 units:

>>> model = PrebuiltHodModelFactory('leauthaud11', threshold = 10.75)

Second, the ``redshift`` keyword argument must be set to the redshift of the
halo catalog you might populate with this model.

>>> model = PrebuiltHodModelFactory('leauthaud11', threshold = 11, redshift = 2)

It is not permissible to dynamically change the ``threshold`` and ``redshift``
of the model instance. If you want to explore the effects of different
thresholds and redshifts, you should instantiate multiple models.

As described in :ref:`altering_param_dict`, you can always change the model parameters
after instantiation by changing the values in the ``param_dict`` dictionary. For example,

>>> model.param_dict['alphasat'] = 1.1

The above line of code changes the power law slope between
halo mass and satellite occupation number, :math:`\langle N_{\rm sat} \rangle \propto M_{\rm halo}^{\alpha}`.
See :ref:`leauthaud11_parameters` for a description of all parameters of this model.

Populating Mocks and Generating Leauthaud et al. (2011) Model Predictions
===========================================================================

As with any Halotools composite model, the model instance
can populate N-body simulations with mock galaxy catalogs.
In the following, we'll show how to do this
with fake simulation data via the ``halocat`` argument.

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> model = PrebuiltHodModelFactory('leauthaud11')
>>> model.populate_mock(halocat)  # doctest: +SKIP

See `ModelFactory.populate_mock` for information about how to
populate your model into different simulations.
See :ref:`galaxy_catalog_analysis_tutorial` for a sequence of worked examples
on how to use the `~halotools.mock_observables` sub-package
to study a wide range of astronomical statistics predicted by your model.

Studying the Leauthaud et al. (2011) Model Features
======================================================

In addition to populating mocks, the ``leauthaud11`` model also gives you access to
its underlying analytical relations. Here are a few examples:

>>> import numpy as np
>>> halo_mass = np.logspace(11, 15, 100)

To compute the mean number of each galaxy type as a function of halo mass:

>>> mean_ncen = model.mean_occupation_centrals(prim_haloprop = halo_mass)
>>> mean_nsat = model.mean_occupation_satellites(prim_haloprop = halo_mass)

To compute the mean stellar mass of central galaxies as a function of halo mass:

>>> mean_sm_cens = model.mean_stellar_mass_centrals(prim_haloprop = halo_mass)

Now suppose you wish to know the mean halo mass of a central galaxy with known stellar mass:

>>> log_stellar_mass = np.linspace(9, 12, 100)
>>> inferred_log_halo_mass = model.mean_log_halo_mass_centrals(log_stellar_mass)

.. _leauthaud11_parameters:

Parameters of the Leauthaud et al. (2011) model
=================================================

The best way to learn what the parameters of a model do is to
just play with the code: change parameter values, make plots of how the
underying analytical relations vary, and also of how the
mock observables vary. Here we just give a simple description of the meaning
of each parameter. You can also refer to the original
Leauthaud et al. (2011) publication, arXiv:1103.2077, and also the original
Behroozi et al. (2010) publication, arXiv:1001.0015,
for further details. A succinct summary also appears in Section 2.4 of arXiv:1512.03050.

To see how the following parameters are implemented, see `Leauthaud11Cens.mean_occupation` and `Behroozi10SmHm.mean_stellar_mass`.

* param_dict['smhm_m0_0'] - Characteristic stellar mass at redshift-zero in the :math:`\langle M_{\ast} \rangle(M_{\rm halo})` map.

* param_dict['smhm_m0_a'] - Redshift evolution of the characteristic stellar mass.

* param_dict['smhm_m1_0'] - Characteristic halo mass at redshift-zero in the :math:`\langle M_{\ast} \rangle(M_{\rm halo})` map.

* param_dict['smhm_m1_a'] - Redshift evolution of the characteristic halo mass.

* param_dict['smhm_beta_0'] - Low-mass slope at redshift-zero of the :math:`\langle M_{\ast} \rangle(M_{\rm halo})` map.

* param_dict['smhm_beta_a'] - Redshift evolution of the low-mass slope.

* param_dict['smhm_delta_0'] - High-mass slope at redshift-zero of the :math:`\langle M_{\ast} \rangle(M_{\rm halo})` map.

* param_dict['smhm_delta_a'] - Redshift evolution of the high-mass slope.

* param_dict['smhm_gamma_0'] - Transition between low- and high-mass behavior at redshift-zero of the :math:`\langle M_{\ast} \rangle(M_{\rm halo})` map.

* param_dict['smhm_gamma_a'] - Redshift evolution of the transition.

* param_dict['u'scatter_model_param1'] - Log-normal scatter in the stellar-to-halo mass relation.

To see how the following parameters are implemented, see `Leauthaud11Sats.mean_occupation` and `Behroozi10SmHm.mean_stellar_mass`.

* param_dict['alphasat'] - Power law slope of the relation between halo mass and :math:`\langle N_{\rm sat} \rangle`.

* param_dict['betasat'] - Controls the amplitude of the power law slope :math:`\langle N_{\rm sat} \rangle`.

* param_dict['bsat'] - Also controls the amplitude of the power law slope :math:`\langle N_{\rm sat} \rangle`.

* param_dict['betacut'] - Controls the low-mass cutoff of :math:`\langle N_{\rm sat} \rangle`.

* param_dict['bcut'] - Also controls the low-mass cutoff of :math:`\langle N_{\rm sat} \rangle`.


