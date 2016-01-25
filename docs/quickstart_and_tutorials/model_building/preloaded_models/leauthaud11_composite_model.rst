.. _leauthaud11_composite_model:

*********************************************
Leauthaud et al. (2011) Composite Model
*********************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes the basic behavior of 
the ``leauthaud11`` composite HOD model. To see how this composite 
model is built by the `~halotools.empirical_models.PrebuiltHodModelFactory` class, 
see `~halotools.empirical_models.leauthaud11_model_dictionary`. 

Overview of the Model Features
=================================
HOD-style based on Leauthaud et al. (2011), arXiv:1103.2077. 
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

Building the Model 
=====================
You can build an instance of this model using the 
`~halotools.empirical_models.PrebuiltHodModelFactory` class as follows:

>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> model = PrebuiltHodModelFactory('leauthaud11')


Customizing the Model
=================================

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

Populating Mocks and Generating Model Predictions
======================================================

As with any Halotools composite model, the above line of code 
will return a model instance that can populate N-body simulations 
with mock galaxy catalogs. In the following, we'll show how to do this 
with fake simulation data via the ``halocat`` argument. 

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> model = PrebuiltHodModelFactory('leauthaud11')
>>> model.populate_mock(halocat = halocat) 

See `ModelFactory.populate_mock` for information about how to  
populate your model into different simulations.  
See :ref:`mock_observation_quickstart` for a quick reference on 
generating common model predictions such as galaxy clustering and lensing, 
and :ref:`mock_observation_overview` for more detailed information on how the 
`~halotools.mock_observables` sub-package can be used to study 
a wide range of astronomical statistics predicted by your model. 


Studying the Model Features 
==============================

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








