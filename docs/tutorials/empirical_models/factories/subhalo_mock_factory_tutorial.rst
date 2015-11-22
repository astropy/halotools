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
of the `SubhaloMockFactory` class. In :ref:`populate_subhalo_mock_convenience_method` we will 
describe the most common way in which the `SubhaloMockFactory` class is used: by calling the 
``populate_mock`` method bound to all Halotools composite models. We provide detailed 
notes on the source code of the mock factory in :ref:`subhalo_mock_algorithm`. 


.. _basic_syntax_subhalo_mocks:

Basic syntax for making subhalo-based mocks
===============================================

The `SubhaloMockFactory` is responsible for one task: using a Halotools composite model 
to populate a simulation with mock galaxies. To fulfill this one task, there are just 
two required keyword arguments: a ``model`` and a ``snapshot``. The model must be an instance 
of a `SubhaloModelFactory`, and the snapshot must be an instance of a `~halotools.sim_manager.HaloCatalog`. 
For simplicity, in this tutorial we will assume that you are using the `SubhaloMockFactory`  
to populate the default snapshot. For documentation on populating alternative catalogs, 
see :ref:`populating_mocks_with_alternate_sims_tutorial`. 

As a simple example, here is how to create an instance of the `SubhaloMockFactory` 
with a composite model based on the prebuilt 
`~halotools.empirical_models.composite_models.smhm_models.behroozi10_model_dictionary`:

.. code-block:: python

	behroozi10_model = SubhaloModelFactory('behroozi10')
	default_snapshot = HaloCatalog()
	mock = SubhaloMockFactory(model = behroozi10_model, snapshot = default_snapshot)

When you instantiate the `SubhaloMockFactory`, 



.. _populate_subhalo_mock_convenience_method:

The ``populate_mock`` convenience method
=====================================================


.. _subhalo_mock_algorithm:

Algorithm for populating subhalo-based mocks 
================================================














