.. _hearin15_composite_model:

*********************************************
Hearin et al. (2015) Composite Model
*********************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes the basic behavior of
the ``hearin15`` composite HOD model. To see how this composite
model is built by the `~halotools.empirical_models.PrebuiltHodModelFactory` class,
see `~halotools.empirical_models.hearin15_model_dictionary`.

Overview of the Hearin et al. (2015) Model Features
======================================================
This HOD-style model is based on Hearin et al. (2015), arXiv:1512.03050.
The behavior of this model is identical to Leauthaud et al. (2011),
except this model implements assemby bias using decorated HOD methods in the
`HeavisideAssembias` class.

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

For both centrals and satellites, the occupation statistics are decorated with assembly bias.

Building the Hearin et al. (2015) Model
=========================================
You can build an instance of this model using the
`~halotools.empirical_models.PrebuiltHodModelFactory` class as follows:

>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> model = PrebuiltHodModelFactory('hearin15')

Customizing the Hearin et al. (2015) Model
=============================================

There are numerous keyword arguments you can use to customize
the instance returned by the factory.

The ``threshold`` keyword argument and the ``redshift`` keyword
argument behave in the exact same way as they do in the ``leauthaud11`` model.
See :ref:`leauthaud11_composite_model` for further details.

The ``sec_haloprop_key`` keyword argument determines the secondary halo property
used to modulate the assembly bias. So, if you want halos at fixed mass with
above- or below-average concentration to have above- or below-average mean occupations,
you would set ``sec_haloprop_key`` to ``halo_nfw_conc``.

The ``central_assembias_strength`` keyword argument determines how strong the
assembly bias is in the occupation statistics of central galaxies.
For constant assembly bias strength at all masses,
set this variable to be a float between -1 to 1.
For assembly bias strength that has mass-dependence,
you should provide a list of control values, which will be interpreted as the
strength at each of the input ``central_assembias_strength_abscissa`` control points.

As described below in :ref:`hearin15_parameters`, the strength of assembly bias
can always be modulated after instantiation by changing the appropriate values
in ``param_dict``. However, the ``central_assembias_strength_abscissa`` cannot be
determined dynamically after instantiation. If you want to change the abscissa,
you must instantiate a new model.

Exactly analogous comments apply to the ``satellite_assembias_strength`` and
``satellite_assembias_strength_abscissa`` keyword arguments.

The ``split`` keyword argument determines how your halo population is divided into
two sub-populations at each mass. So, if you want halos in the upper 75th
percentile of concentration at each mass to have different occupation statistics than
halos in the lower 25th percentile, you would set ``split`` to 0.75.
You can study the impact of mass-dependence of subpopulation division by passing in
a list of control values for the ``split`` keyword, which will be interpreted as the
splitting fraction at each of the input ``split_abscissa`` control points.

Once you make a choice for ``split``, you cannot change this after instantiating a model.
If you want to study the effects of different choices for ``split``,
you must instantiate a new model.

Populating Mocks and Generating Hearin et al. (2015) Model Predictions
=========================================================================

As with any Halotools composite model, the model instance
can populate N-body simulations with mock galaxy catalogs.
In the following, we'll show how to do this
with fake simulation data via the ``halocat`` argument.

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> model = PrebuiltHodModelFactory('hearin15')
>>> model.populate_mock(halocat)  # doctest: +SKIP

See `ModelFactory.populate_mock` for information about how to
populate your model into different simulations.
See :ref:`galaxy_catalog_analysis_tutorial` for a sequence of worked examples
on how to use the `~halotools.mock_observables` sub-package
to study a wide range of astronomical statistics predicted by your model.

Studying the Hearin et al. (2015) Model Features
==================================================

In addition to populating mocks, the ``hearin15`` model also gives you access to
its underlying analytical relations. Firstly, the ``hearin15`` model naturally
inherits all of the methods of the ``leauthaud11`` model, which you can read about
in :ref:`leauthaud11_composite_model`. Here we give examples of the additional
methods that are unique to this model.

>>> import numpy as np
>>> halo_mass = np.logspace(11, 15, 100)

The ``mean_occupation`` methods of the ``hearin15`` model operate slightly
differently than they do in the ``leauthaud11`` model, because here
two halo properties govern the behavior of the model, not just one.
Suppose you wish to compute the mean occupation of centrals for halos
in the upper-percentile split. In this case you can force the
``mean_occupation_centrals`` method to interpret the halos as being
in the upper-percentile division via the ``sec_haloprop_percentile`` keyword:

>>> mean_ncen_upper = model.mean_occupation_centrals(prim_haloprop = halo_mass, sec_haloprop_percentile=1)

For halos in the lower-percentile split:

>>> mean_ncen_lower = model.mean_occupation_centrals(prim_haloprop = halo_mass, sec_haloprop_percentile=0)

If you have a mixed population, just pass in a second array storing the actual value values
of the secondary halo property via the ``sec_haloprop`` keyword:

>>> fake_sec_prop = np.random.random(len(halo_mass))
>>> mean_ncen = model.mean_occupation_centrals(prim_haloprop=halo_mass, sec_haloprop=fake_sec_prop)

To compute the strength of assembly bias as a function of halo mass:

>>> assembias_cens = model.assembias_strength_centrals(prim_haloprop=halo_mass)
>>> assembias_sats = model.assembias_strength_satellites(prim_haloprop=halo_mass)

.. _hearin15_parameters:

Parameters of the Hearin et al. (2015) Model
=================================================

The ``hearin15`` model naturally inherits all of
the parameters of the ``leauthaud11`` model, which you can read about
in :ref:`leauthaud11_parameters`. Here we only describe the parameters
that are unique to the ``hearin15`` model.

* param_dict['mean_occupation_centrals_assembias_param1'] - controls the strength of assembly bias in the centrals population as specified at the first control point. If you passed in a float to the ``centrals_assembias_strength`` keyword argument, there will only be one such parameter. If you passsed in a list, there will be one parameter per list element. Changing the values of this parameter modulates the strength of assembly bias. The only permissible values are between -1 to 1; values outside this range will be interpreted as endpoint values.




















