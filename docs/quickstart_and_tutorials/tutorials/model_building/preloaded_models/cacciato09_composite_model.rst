.. _cacciato09_composite_model:

*********************************************
Cacciato et al. (2009) Composite Model
*********************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes the basic behavior of
the ``cacciato09`` composite CLF model. To see how this composite
model is built by the `~halotools.empirical_models.PrebuiltHodModelFactory` class,
see `~halotools.empirical_models.cacciato09_model_dictionary`.

For brevity, we describe this model as a "Conditional Luminosity Function" model. However,
the associated classes work equally well as a model for the "Conditional Stellar Mass Function".
The only difference will be a change in the ``prim_galprop_key`` from ``luminosity`` to ``stellar_mass``,
and an accompanying change to the parameter values in the model dictionary.

Overview of the Cacciato et al. (2009) Model Features
========================================================
This CLF-style model is based on Cacciato et al. (2009), arXiv:0807.4932.
The behavior of this model is governed by an assumed mass-to-light relation for the centrals
and a modified Schechter function for the satellites.

There are two populations, centrals and satellites.
Central occupation statistics are given by a log-normal distribution;
the class governing this behavior is `~halotools.empirical_models.Cacciato09Cens`.
Central galaxies are assumed to reside at the exact center of the host halo;
the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`.

Satellite occupation statistics are given by a Poisson distribution, with luminosities
given by a modified Schechter function.
the class governing this behavior is `~halotools.empirical_models.Cacciato09Sats`;
satellites in this model follow an (unbiased) NFW profile, as governed by the
`~halotools.empirical_models.NFWPhaseSpace` class.

Building the Cacciato et al. (2009) Model
============================================
You can build an instance of this model using the
`~halotools.empirical_models.PrebuiltHodModelFactory` class as follows:

>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> model = PrebuiltHodModelFactory('cacciato09')

Customizing the Cacciato et al. (2009) Model
=================================================

The ``threshold`` keyword argument allows you to customize the
luminosity threshold of the galaxy sample in units of Lsun with h=1 units:

>>> model = PrebuiltHodModelFactory('cacciato09', threshold = 11)

It is not permissible to dynamically change the ``threshold``
of the model instance. If you want to explore the effects of different
thresholds, you should instantiate multiple models. Alternatively, you can
always impose a higher threshold on an already existing galaxy catalog produced
with a given model instance. The resulting reduced catalog will have the same
statistical properties as if you ran the model with the higher threshold and
same parameters.

As described in :ref:`altering_param_dict`, you can always change the model parameters
after instantiation by changing the values in the ``param_dict`` dictionary. For example,

>>> model.param_dict['sigma'] = 0.2

The above line of code changes the scatter in the mass-to-light ratio.
See :ref:`cacciato09_parameters` for a description of all parameters of this model.

Populating Mocks and Generating Cacciato et al. (2009) Model Predictions
===========================================================================

As with any Halotools composite model, the model instance
can populate N-body simulations with mock galaxy catalogs.
In the following, we'll show how to do this
with fake simulation data via the ``halocat`` argument.

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> model = PrebuiltHodModelFactory('cacciato09')
>>> model.populate_mock(halocat)  # doctest: +SKIP

See `ModelFactory.populate_mock` for information about how to
populate your model into different simulations.
See :ref:`galaxy_catalog_analysis_tutorial` for a sequence of worked examples
on how to use the `~halotools.mock_observables` sub-package
to study a wide range of astronomical statistics predicted by your model.

Studying the Cacciato et al. (2009) Model Features
======================================================

In addition to populating mocks, the ``cacciato09`` model also gives you access to
its underlying analytical relations. Here are a few examples:

>>> import numpy as np
>>> halo_mass = np.logspace(11, 15, 100)

To compute the median luminosity of central galaxies as a function of halo mass:

>>> median_lum = model.median_prim_galprop_centrals(prim_haloprop = halo_mass)

To compute the average number of satellites per halo as a function of halo mass:

>>> mean_nsat = model.mean_occupation_satellites(prim_haloprop=halo_mass)

By modifying the parameters stored in the ``param_dict``, the underlying analytical
relations such as those above allow you to study how the model behaves without the
need to create Monte Carlo realizations of the Universe.

.. _cacciato09_parameters:

Parameters of the Cacciato et al. (2009) model
=================================================

The best way to learn what the parameters of a model do is to
just play with the code: change parameter values, make plots of how the
underying analytical relations vary, and also of how the
mock observables vary. Here we just give a simple description of the meaning
of each parameter. You can also refer to the original
Cacciato et al. (2009) publication, arXiv:0807.4932. The fiducial values of the
``cacciato09`` model instance implemented in Halotools are drawn from the WMAP3
analysis of that publication.

The model also generalizes the CLF model of Cacciato et al. (2009) by allowing
modifications of the high-luminosity cut-off of the satellite population.
Briefly, changing the delta parameters should only affect the abundance of
satellites that have luminosities similar to the central luminosity. On the other
hand, faint satellites should be unaffected. The details of the 2 delta parameters
are described in Lange et al. (2017), arXiv:1705.05043. Setting both to 0, as
done by default, is equivalent to the model of Cacciato et al. (2009).

* param_dict['log_L_0'] -  Normalization of central mass-to-light ratio.

* param_dict['log_M_1'] - Characteristic mass of central mass-to-light ratio.

* param_dict['gamma_1'] - Low-mass slope of central mass-to-light ratio.

* param_dict['gamma_2'] - High-mass slope of central mass-to-light ratio.

* param_dict['sigma'] - Scatter in the log-normal mass-to-light distribution of centrals.

* param_dict['a_1'] - Sets the faint-end slope of the satellite luminosity function.

* param_dict['a_2'] - Determines the mass-dependence of the faint-end slope of the satellite luminosity function.

* param_dict['log_M_2'] - Determines the mass-dependence of the faint-end slope of the satellite luminosity function.

* param_dict['b_0'] - Modifies normalization of the satellite luminosity function.

* param_dict['b_1'] - Modifies normalization of the satellite luminosity function.

* param_dict['b_2'] - Modifies normalization of the satellite luminosity function.

* param_dict['delta_1'] - Modifies the high-luminosity exponential cut-off of the satellite luminosity function.

* param_dict['delta_2'] - Modifies the high-luminosity exponential cut-off of the satellite luminosity function.



