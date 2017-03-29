.. _zheng07_composite_model:

*********************************************
Zheng et al. (2007) Composite Model
*********************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes the basic behavior of
the ``zheng07`` composite HOD model. To see how this composite
model is built by the `~halotools.empirical_models.PrebuiltHodModelFactory` class,
see `~halotools.empirical_models.zheng07_model_dictionary`.

Overview of the Zheng et al. (2007) Model Features
==================================================
This HOD-style model is based on Zheng et al. (2007), arXiv:0703457.
There are two populations, centrals and satellites.
Central occupation statistics are given by a nearest integer distribution
with first moment given by an ``erf`` function; the class governing this
behavior is `~halotools.empirical_models.Zheng07Cens`.
Central galaxies are assumed to reside at the exact center of the host halo;
the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`.

Satellite occupation statistics are given by a Poisson distribution
with first moment given by a power law that has been truncated at the low-mass end;
the class governing this behavior is `~halotools.empirical_models.Zheng07Sats`;
satellites in this model follow an (unbiased) NFW profile, as governed by the
`~halotools.empirical_models.NFWPhaseSpace` class.


Building the Zheng et al. (2007) Model
=========================================
You can build an instance of this model using the
`~halotools.empirical_models.PrebuiltHodModelFactory` class as follows:

>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> model = PrebuiltHodModelFactory('zheng07')


Customizing the Zheng et al. (2007) Model
===============================================

There are three keyword arguments you can use to customize
the instance returned by the factory, ``threshold``, ``redshift`` and
``modulate_with_cenocc``. In this section, we cover each of these keywords in turn.

Role of the ``threshold`` keyword
-----------------------------------
The ``threshold`` keyword argument pertains to the r-band absolute magnitude
of the luminosity of the galaxy sample:

>>> model = PrebuiltHodModelFactory('zheng07', threshold = -20)

The only purpose of this keyword is to allow you
to instantiate your model according to the best-fit values of the parameters
taken from Table 1 of Zheng et al. (2007). After instantiation, the
``threshold`` attribute has no impact whatsoever on the behavior of the model.
If you choose a different value for ``threshold`` than one of the values in Table 1
of Zheng et al. (2007), the model behavior will be set to the best-fit parameters
of the ``default_luminosity_threshold`` variable set in the
`~halotools.empirical_models.model_defaults` module, and you can proceed to
alter the ``param_dict`` however you like (see below).

As described in :ref:`altering_param_dict`, you can always change the model parameters
after instantiation by changing the values in the ``param_dict`` dictionary. For example,

>>> model.param_dict['logMmin'] = 12.5

The above line of code changes the minimum mass for
a halo to host a central galaxy to :math:`10^{12.5}M_{\odot}`.
See :ref:`zheng07_parameters` for a description of all parameters of this model.

Role of the ``redshift`` keyword
-----------------------------------
The ``redshift`` keyword argument must be set to the redshift of the
halo catalog you might populate with this model.

>>> model = PrebuiltHodModelFactory('zheng07', threshold = -20, redshift = 2)

For the ``zheng07`` model, the ``redshift`` attribute has no impact whatsoever on
the behavior of the model; the purpose of this keyword for factory standardization purposes only,
and also to remind users that this model has only been calibrated against *z=0* observational data.


Role of the ``modulate_with_cenocc`` keyword
----------------------------------------------

.. note::

	As described in the docstring, if you are trying to strictly reproduce
	the results in `Zheng et al 2007 <http://arxiv.org/abs/astro-ph/0703457>`_.
	you should set the ``modulate_with_cenocc`` keyword to True.
	Note that the default value is False, which for typical regions of parameter
	space gives very similar behavior to the published results.

The ``modulate_with_cenocc`` keyword argument controls how the
`Zheng07Sats.mean_occupation` function is calculated.
The default value is False, in which case the `Zheng07Sats.mean_occupation`
function is calculated according to the following equation:

.. math::

	\langle N_{\mathrm{sat}} \rangle_{M} = \left( \frac{M - M_{0}}{M_{1}} \right)^{\alpha}

When ``modulate_with_cenocc`` is True, then the value of
:math:`\langle N_{\rm sat}\rangle(M_{\rm vir})` as calculated above
will be multiplied by an overall factor of
:math:`\langle N_{\rm cen}\rangle(M_{\rm vir})`,
where the factor of :math:`\langle N_{\rm cen}\rangle` is defined
according to `Zheng07Cens.mean_occupation`:

.. math::

        \langle N_{\mathrm{cen}} \rangle(M_{\rm vir}) = \frac{1}{2}\left( 1 + \mathrm{erf}\left( \frac{\log_{10}M - \log_{10}M_{min}}{\sigma_{\log_{10}M}} \right) \right)

If you wish to modulate your satellite occupation statistics using
alternative models for :math:`\langle N_{\rm cen}\rangle(M_{\rm vir})`,
see :ref:`zheng07_using_cenocc_model_tutorial`.


Populating Mocks and Generating Zheng et al. (2007) Model Predictions
=======================================================================

As with any Halotools composite model, the model instance
can populate N-body simulations with mock galaxy catalogs.
In the following, we'll show how to do this
with fake simulation data via the ``halocat`` argument.

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> model = PrebuiltHodModelFactory('zheng07')
>>> model.populate_mock(halocat = halocat)  # doctest: +SKIP

See `ModelFactory.populate_mock` for information about how to
populate your model into different simulations.
See :ref:`galaxy_catalog_analysis_tutorial` for a sequence of worked examples
on how to use the `~halotools.mock_observables` sub-package
to study a wide range of astronomical statistics predicted by your model.

Studying the Zheng et al. (2007) Model Features
====================================================

In addition to populating mocks, the ``zheng07`` model also gives you access to
its underlying analytical relations. Here are a few examples:

>>> import numpy as np
>>> halo_mass = np.logspace(11, 15, 100)

To compute the mean number of each galaxy type as a function of halo mass:

>>> mean_ncen = model.mean_occupation_centrals(prim_haloprop = halo_mass)
>>> mean_nsat = model.mean_occupation_satellites(prim_haloprop = halo_mass)


.. _zheng07_parameters:

Parameters of the Zheng et al. (2007) model
=================================================

The best way to learn what the parameters of a model do is to
just play with the code: change parameter values, make plots of how the
underying analytical relations vary, and also of how the
mock observables vary. Here we just give a simple description of the meaning
of each parameter. You can also refer to the original publication, arXiv:0703457,
for further details.

To see how the following parameters are implemented, see `Zheng07Cens.mean_occupation`.

* param_dict['logMmin'] - Minimum mass required for a halo to host a central galaxy.

* param_dict['sigma_logM'] - Rate of transition from :math:`\langle N_{\rm cen} \rangle = 0 \Rightarrow \langle N_{\rm cen} = 1 \rangle`.

To see how the following parameters are implemented, see `Zheng07Sats.mean_occupation`.

* param_dict['alpha'] - Power law slope of the relation between halo mass and :math:`\langle N_{\rm sat} \rangle`.

* param_dict['logM0'] - Low-mass cutoff in :math:`\langle N_{\rm sat} \rangle`.

* param_dict['logM1'] - Characteristic halo mass where :math:`\langle N_{\rm sat} \rangle` begins to assume a power law form.


