.. _behroozi10_composite_model:

*********************************************
Behroozi et al. (2010) Composite Model
*********************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes the basic behavior of
the ``behroozi10`` composite subhalo model. To see how this composite
model is built by the `~halotools.empirical_models.PrebuiltSubhaloModelFactory` class,
see `~halotools.empirical_models.behroozi10_model_dictionary`.

Overview of the Behroozi et al. (2010) Model Features
========================================================
This subhalo-based model is an implementation of
Behroozi et al. (2010), arXiv:1001.0015.
There is a one-to-one mapping between stellar mass and subhalo mass
governed by a parameterized form for the stellar-to-halo-mass relation (SMHM).
The class where the SMHM behavior is defined is `Behroozi10SmHm`.

Building the Behroozi et al. (2010) Model
============================================
You can build an instance of this model using the
`~halotools.empirical_models.PrebuiltSubhaloModelFactory` class as follows:

>>> from halotools.empirical_models import PrebuiltSubhaloModelFactory
>>> model = PrebuiltSubhaloModelFactory('behroozi10')


Customizing the Behroozi et al. (2010) Model
=================================================

There are several keyword arguments you can use to customize
the instance returned by the factory:

First, the ``redshift`` keyword argument must be set to the redshift of the
halo catalog you might populate with this model.

>>> model = PrebuiltSubhaloModelFactory('behroozi10', redshift = 2)

It is not permissible to dynamically change the ``redshift``
of the ``behroozi10`` composite model instance.
If you want to explore the model variations with redshift,
you should instantiate multiple models.
Or, alternatively, if you only want to study
the underlying analytical SMHM relation, and not populate mocks,
you can just build an instance of the `Behroozi10SmHm` component model
class without specifying a redshift, in which case you can
call the methods of the `Behroozi10SmHm` instance for any redshift.

Second, the ``prim_haloprop_key`` keyword argument allows you to choose
which subhalo property regulates mean stellar mass.
In principle, you can choose any column name in the halo catalog you will
be populating, but this key should be a mass-like variable in order to get
sensible results, e.g., ``halo_mpeak``, ``halo_macc``, etc.
It is not permissible to dynamically change the ``prim_haloprop_key``
of the ``behroozi10`` composite model instance.
If you want to explore the effects of choosing different
halo properties, you should instantiate multiple models.

Finally, you can choose how stochasticity between halo and stellar mass is modeled
with the ``scatter_abscissa`` and ``scatter_ordinates`` keywords.
These arguments determine the level of scatter in stellar mass, given in dex.
The abscissa serve as control points and the ordinates the values of the scatter
at those control points. So, for example, if you wanted to have 0.3 dex of
scatter at :math:`M_{\rm halo} = 10^{12}M_{\odot}` and 0.1 dex of scatter
at :math:`M_{\rm halo} = 10^{15}M_{\odot}`:

>>> model = PrebuiltSubhaloModelFactory('behroozi10', scatter_abscissa = [1e12, 1e15], scatter_ordinates = [0.3, 0.1])

For constant scatter, use a one-element python list.
It is not permissible to dynamically change the abscissa after instantiation,
though you can vary the ordinates by changing the appropriate values in the ``param_dict``.
For example, in the above model, the following line will change the
scatter to 0.2 dex in :math:`M_{\rm halo} = 10^{12}M_{\odot}` halos:

>>> model.param_dict['scatter_model_param1'] = 0.2

Populating Mocks and Generating Behroozi et al. (2010) Model Predictions
===========================================================================

As with any Halotools composite model, the model instance
can populate N-body simulations with mock galaxy catalogs.
In the following, we'll show how to do this
with fake simulation data via the ``halocat`` argument.

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> model = PrebuiltSubhaloModelFactory('behroozi10')
>>> model.populate_mock(halocat)  # doctest: +SKIP

See `ModelFactory.populate_mock` for information about how to
populate your model into different simulations.
See :ref:`galaxy_catalog_analysis_tutorial` for a sequence of worked examples
on how to use the `~halotools.mock_observables` sub-package
to study a wide range of astronomical statistics predicted by your model.

Studying the Behroozi et al. (2010) Model Features
======================================================

In addition to populating mocks, the ``behroozi10`` model also gives you access to
its underlying analytical relations. Here are a few examples:

>>> import numpy as np
>>> halo_mass = np.logspace(11, 15, 100)

To compute the mean stellar mass as a function of halo mass:

>>> mean_sm = model.mean_stellar_mass(prim_haloprop = halo_mass)

.. _behroozi10_parameters:

Parameters of the Behroozi et al. (2010) model
=================================================

The best way to learn what the parameters of a model do is to
just play with the code: change parameter values, make plots of how the
underying analytical relations vary, and also of how the
mock observables vary. Here we just give a simple description of the meaning
of each parameter. You can also refer to the original
Behroozi et al. (2010) publication, arXiv:1001.0015,
for further details.

To see how the following parameters are implemented, see `Behroozi10SmHm.mean_stellar_mass`.

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






