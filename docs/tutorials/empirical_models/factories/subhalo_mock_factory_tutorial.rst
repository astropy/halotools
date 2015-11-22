:orphan:

.. currentmodule:: halotools.empirical_models.factories

.. _subhalo_mock_factory_tutorial:

********************************************************************
Tutorial on Making Subhalo-based Mocks
********************************************************************

This section of the documentation provides detailed notes 
for how the `SubhaloMockFactory` populates subhalo catalogs with synthetic galaxy populations. 
The `SubhaloMockFactory` uses composite models built with the `SubhaloModelFactory`, which 
is documented in the :ref:`subhalo_model_factory_tutorial`. 


Outline 
========

We will start in :ref:`basic_syntax_subhalo_mocks` with a high-level overview of the functionality 
of the `SubhaloMockFactory` class. We provide detailed 
notes on the source code of the mock factory in :ref:`subhalo_mock_algorithm`. 


.. _basic_syntax_subhalo_mocks:

Basic syntax for making subhalo-based mocks
===============================================

The `SubhaloMockFactory` is responsible for one task: using a Halotools composite model 
to populate a simulation with mock galaxies. To fulfill this one task, there are just 
two required keyword arguments: ``model`` and ``halocat``. The model must be an instance 
of a `SubhaloModelFactory`, and the halocat must be an instance of a `~halotools.sim_manager.HaloCatalog`. 
For simplicity, in this tutorial we will assume that you are using the `SubhaloMockFactory`  
to populate the default halo catalog. For documentation on populating alternative catalogs, 
see :ref:`populating_mocks_with_alternate_sims_tutorial`. 

As a simple example, here is how to create an instance of the `SubhaloMockFactory` 
with a composite model based on the prebuilt 
`~halotools.empirical_models.composite_models.smhm_models.behroozi10_model_dictionary`:

.. code-block:: python

	behroozi10_model = SubhaloModelFactory('behroozi10')
	default_halocat = HaloCatalog()
	mock = SubhaloMockFactory(model = behroozi10_model, halocat = default_halocat)

Instantiating the `SubhaloMockFactory` triggers the pre-processing phase of mock population. 
Briefly, this phase does as many tasks in advance of actual mock population as possible 
to improve the efficiency of MCMCs (see below for details). 

By default, instantiating the factory also triggers 
the `SubhaloMockFactory.populate` method to be called. This is the method that actually creates 
the galaxy population. By calling the `SubhaloMockFactory.populate` method, 
a new ``galaxy_table`` attribute is created and bound to the instance. 
The ``galaxy_table`` attribute stores an Astropy `~astropy.table.Table` object with one row 
per mock galaxy and one column for every property assigned by the chosen composite model. 

An aside on the ``populate_mock`` convenience function 
---------------------------------------------------------

Probably the most common way in which you will actually interact with the `~SubhaloMockFactory` is 
by the `SubhaloModelFactory.populate_mock` method, which is just a convenience wrapper around the 
`SubhaloMockFactory.populate` method. Consider the following call to this function:

.. code-block:: python 

	behroozi10_model.populate_mock()

This is essentially equivalent to the three lines of code written above. The only difference is that 
in the above line will create a ``mock`` attribute that is bound to ``behroozi10_model``; this ``mock`` 
attribute is simply an instance of the `~SubhaloMockFactory`. 


.. _subhalo_mock_algorithm:

Algorithm for populating subhalo-based mocks 
================================================

.. _subhalo_mock_preprocessing_phase:

Pre-processing phase
----------------------

The pre-processing phase begins by calling the `__init__` constructor of the `~MockFactory` super-class. 
The main non-trivial task performed by the super-class constructor is to call 
the `~SubhaloMockFactory.build_additional_haloprops_list` method, 
which determines the set of halo catalog properties 
that will be included in the ``galaxy_table``. 

In the `~SubhaloMockFactory.preprocess_halo_catalog`, new columns are added to the ``halo_table`` 
according to any entries in the ``new_haloprop_func_dict``; any such columns will automatically be included 
in the ``galaxy_table``. See :ref:`new_haloprop_func_dict_mechanism` for further details. 

The pre-processing phase concludes with the call to the `~SubhaloMockFactory.precompute_galprops` method. 



.. _subhalo_mock_population_phase:

Mock-population phase
----------------------










