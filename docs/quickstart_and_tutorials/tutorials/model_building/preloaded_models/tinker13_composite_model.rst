.. _tinker13_composite_model:

*********************************************
Tinker et al. (2013) Composite Model
*********************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes the basic behavior of
the ``tinker13`` composite HOD model. To see how this composite
model is built by the `~halotools.empirical_models.PrebuiltHodModelFactory` class,
see `~halotools.empirical_models.tinker13_model_dictionary`.

.. _tinker13_model_features_overview:

Overview of the Tinker et al. (2013)  Model Features
========================================================
This HOD-style model is based on Tinker et al. (2013), arXiv:1308.2974.
The behavior of this model is governed by an assumed underlying stellar-to-halo-mass relation
that is distinct for star-forming and quiescent populations.

There are two populations, centrals and satellites.
Central occupation statistics are given by a nearest integer distribution
with first moment given by an ``erf`` function; the class governing this
behavior is `~halotools.empirical_models.Tinker13Cens`.
Central galaxies are assumed to reside at the exact center of the host halo;
the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`.

Satellite occupation statistics are given by a Poisson distribution
with first moment given by a power law that has been truncated at the low-mass end;
the classes governing this behavior are `~halotools.empirical_models.Tinker13QuiescentSats`
and `~halotools.empirical_models.Tinker13ActiveSats`;
satellites in this model follow an (unbiased) NFW profile, as governed by the
`~halotools.empirical_models.NFWPhaseSpace` class.

Building the Tinker et al. (2013) Model
==========================================
You can build an instance of this model using the
`~halotools.empirical_models.PrebuiltHodModelFactory` class as follows:

>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> model = PrebuiltHodModelFactory('tinker13')


Customizing the Tinker et al. (2013) Model
=============================================

There are several keyword arguments you can use to customize
the instance returned by the factory.

The ``threshold`` keyword argument and the ``redshift`` keyword
argument behave in the exact same way as they do in the ``leauthaud11`` model.
See :ref:`leauthaud11_composite_model` for further details.

In the ``tinker13`` model, the quiescent fraction of central galaxies is
specified at a set of control points via the ``quiescent_fraction_abscissa``
and ``quiescent_fraction_ordinates`` keywords. Linear interpolation is used to
for the values of the quenched fraction evaluated at distinct values from the control points.
So, for example, if you wanted to initialize your model so that the quenched fraction
at :math:`M_{\rm vir}/M_{\odot} = 10^{12}, 10^{13}, 10^{14}, 10^{15}` is
:math:`0.25, 0.5, 0.75, 0.9`:

>>> model = PrebuiltHodModelFactory('tinker13', quiescent_fraction_abscissa = [1e12, 1e13, 1e14, 1e15], quiescent_fraction_ordinates = [0.25, 0.5, 0.75, 0.9])

As described in :ref:`altering_param_dict`, you can always change the model parameters
after instantiation by changing the values in the ``param_dict`` dictionary. For example,

>>> model.param_dict['quiescent_fraction_ordinates_param1'] = 0.35

There will be one ``param_dict`` parameter for each entry of
the input ``quiescent_fraction_ordinates``.
Once you instantiate the model you are not permitted to change the abscissa.
To do that, you need to instantiate another model.

Populating Mocks and Generating Tinker et al. (2013) Model Predictions
=========================================================================

As with any Halotools composite model, the model instance
can populate N-body simulations with mock galaxy catalogs.
In the following, we'll show how to do this
with fake simulation data via the ``halocat`` argument.

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> model = PrebuiltHodModelFactory('tinker13')
>>> model.populate_mock(halocat)  # doctest: +SKIP

See `ModelFactory.populate_mock` for information about how to
populate your model into different simulations.
See :ref:`galaxy_catalog_analysis_tutorial` for a sequence of worked examples
on how to use the `~halotools.mock_observables` sub-package
to study a wide range of astronomical statistics predicted by your model.

Studying the Tinker et al. (2013) Model Features
===================================================

In addition to populating mocks, the ``tinker13`` model also gives you access to
its underlying analytical relations. For the most part, the ``tinker13`` model simply
inherits the methods of the ``leauthaud11`` model, which you can read about
in :ref:`leauthaud11_composite_model`. However, there are slight differences
due as ``tinker13`` also models quiescent designation.

>>> import numpy as np
>>> halo_mass = np.logspace(11, 15, 100)

Whereas in ``leauthaud11`` there was a ``mean_occupation_centrals`` method,
in ``tinker13`` there are instead methods for ``mean_occupation_active_centrals``
and ``mean_occupation_quiescent_centrals``.

>>> mean_ncen_q = model.mean_occupation_quiescent_centrals(prim_haloprop = halo_mass)
>>> mean_ncen_a = model.mean_occupation_active_centrals(prim_haloprop = halo_mass)

Similar comments apply to ``mean_stellar_mass`` and ``mean_log_halo_mass``
for centrals and satellites alike.

.. _tinker13_parameters:

Parameters of the Tinker et al. (2013) model
=================================================

For satellite galaxies, the ``tinker13`` model inherits all of
the parameters of the ``leauthaud11`` model twice: one set of parameters
for the star-forming satellites, a second set for the quiescent satellites.
Please refer to :ref:`leauthaud11_parameters` for details.
The same duplicate parameter inheritance applies for centrals.
Additionally, as described in :ref:`tinker13_model_features_overview`,
there are parameters specifying the quiescent fraction of
centrals at the set of control points determined by the
``quiescent_fraction_abscissa`` keyword passed to the constructor.












