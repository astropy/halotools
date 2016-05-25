:orphan:

.. currentmodule:: halotools.empirical_models

.. _hod_mock_factory_source_notes:

********************************************************************
Source code notes on `HodMockFactory` 
********************************************************************

This section of the documentation provides detailed notes 
for how the `HodMockFactory` populates halo catalogs with synthetic galaxy populations. 
The `HodMockFactory` uses composite models built with the `HodModelFactory`, which 
is documented in the :ref:`hod_model_factory_source_notes`. 


.. _basic_syntax_hod_mocks:

Basic syntax for making HOD-style mocks
===============================================

The `HodMockFactory` is responsible for one task: using a Halotools composite model 
to populate a simulation with mock galaxies. To fulfill this one task, there are just 
two required keyword arguments: ``model`` and ``halocat``. The model must be an instance 
of a `HodModelFactory`, and the halocat must be an instance of a `~halotools.sim_manager.CachedHaloCatalog` 
or `~halotools.sim_manager.UserSuppliedHaloCatalog`.  
For simplicity, in this tutorial we will assume that you are using the `HodMockFactory`  
to populate the default halo catalog. However, you can populate alternate halo catalogs 
using the same syntax works with any instance of either 
`~halotools.sim_manager.CachedHaloCatalog` or `~halotools.sim_manager.UserSuppliedHaloCatalog`.


As a simple example, here is how to create an instance of the `HodMockFactory` 
with a composite model based on the prebuilt 
`~halotools.empirical_models.zheng07_model_dictionary`:

.. code-block:: python

	zheng07_model = PrebuiltHodModelFactory('zheng07')
	default_halocat = CachedHaloCatalog()
	mock = HodMockFactory(model = zheng07_model, halocat = default_halocat)

Instantiating the `HodMockFactory` triggers the pre-processing phase of mock population. 
Briefly, this phase does as many tasks in advance of actual mock population as possible 
to improve the efficiency of MCMCs (see below for details). 

By default, instantiating the factory also triggers 
the `HodMockFactory.populate` method to be called. This is the method that actually creates 
the galaxy population. By calling the `HodMockFactory.populate` method, 
a new ``galaxy_table`` attribute is created and bound to the ``mock`` instance. 
The ``galaxy_table`` attribute stores an Astropy `~astropy.table.Table` object with one row 
per mock galaxy and one column for every property assigned by the chosen composite model. 

An aside on the ``populate_mock`` convenience function 
---------------------------------------------------------

Probably the most common way in which you will actually interact with the `~HodMockFactory` is 
by the `HodModelFactory.populate_mock` method, which is just a convenience wrapper around the 
`HodMockFactory.populate` method. Consider the following call to this function:

.. code-block:: python 

	zheng07_model.populate_mock(default_halocat)

This is essentially equivalent to the three lines of code written above. The only difference is that 
in the above line will create a ``mock`` attribute that is bound to ``zheng07_model``; this ``mock`` 
attribute is simply an instance of the `~HodMockFactory`. 


.. _hod_mock_algorithm:

Algorithm for populating HOD-style mocks 
================================================

.. _intro_to_np_repeat:

Basics of `numpy.repeat`: the core function in HOD mock-making
----------------------------------------------------------------

Before going into the details of the `HodMockFactory.populate` method, 
in this section we will first cover a basic introduction to the `numpy.repeat` function, 
which is the single-most important Numpy function used by Halotools to make mock catalogs. 

First let's demonstrate basic usage:

>>> import numpy as np 
>>> num_halos = 5
>>> halo_mass = np.logspace(11, 15, num_halos)
>>> halo_occupations = np.array([2, 0, 1, 0, 3])
>>> galaxy_host_halo_mass = np.repeat(halo_mass, halo_occupations)

The `numpy.repeat` function takes as input two arrays of equal length, 
in this case ``halo_mass`` and ``halo_occupations``. The second array is interpreted 
as the number of times the corresponding entry in the first array should be repeated. 
A visualization of how this function behaves is shown in the diagram below. 

.. image:: np_repeat_tutorial.png

This behavior is exactly what is needed to create a mock galaxy catalog with an HOD-style model. 
The core function of an HOD model is to specify how many galaxies reside in a given halo. 
The task of assigning galaxy occupations to halos is controlled by 
the `~halotools.empirical_models.OccupationComponent.mc_occupation` function 
of whatever sub-class of `~halotools.empirical_models.OccupationComponent` you select as your model. 
The `~halotools.empirical_models.HodMockFactory` does not need to know how occupation statistics 
are modeled - the only thing the factory needs to do is call 
the `~halotools.empirical_models.OccupationComponent.mc_occupation` function to fill the 
``occupations`` array in the diagram. Then the only thing the 
`~halotools.empirical_models.HodMockFactory` is to call `numpy.repeat`, 
passing in the halo catalog ``halo_mvir`` column for the first argument. 
This will create a length-*Ngals* array storing the host halo mass of every mock galaxy, 
which will then be included in the ``galaxy_table``. 
If you want to include additional halo properties as columns in your ``galaxy_table``, 
the `~halotools.empirical_models.HodMockFactory` only needs to pass in additional 
halo catalog columns to `numpy.repeat`. In this way, the factory does not need to know *anything* 
about how the ``occupations`` array comes into existence. The factory simply 
only needs to repeatedly call `numpy.repeat` with the appropriate inputs that are 
determined by the model. 


Pre-processing phase
----------------------

The preliminary tasks of HOD mock-making are carried out by the 
`~HodMockFactory.preprocess_halo_catalog` method. This first thing this function does 
is to throw out subhalos from the halo catalog, and to apply a halo mass completeness cut. 
You can control the completeness cut with the ``Num_ptcl_requirement`` keyword argument passed 
to the `HodMockFactory` constructor (``Num_ptcl_requirement`` is also an optional keyword argument 
that may be passed to the `HodModelFactory.populate_mock` method, which in turn passes this argument 
on to the `HodMockFactory` constructor.)

New columns are then added to the ``halo_table`` 
according to any entries in the ``new_haloprop_func_dict``; any such columns will automatically be included 
in the ``galaxy_table``. See :ref:`new_haloprop_func_dict_mechanism` for further details. 

Memory allocation phase 
---------------------------

After pre-processing the halo catalog, memory must be allocated to store the ``galaxy_table``. 
This is controlled by the `HodMockFactory.allocate_memory` method. 

Mapping galaxy properties to the ``galaxy_table``
------------------------------------------------------



.. _galprops_assigned_before_mc_occupation: 

Galaxy properties assigned prior to calling the occupation components 
========================================================================


.. _determining_the_gal_type_slice:

Determining the appropriate gal_type slice
========================================================================

This section of the tutorial is referenced by :ref:`hod_model_factory_source_notes` 
and explains the following mechanism. 

setattr(getattr(self, new_method_name), 'gal_type', gal_type) # line 4
setattr(getattr(self, new_method_name), 'feature_name', feature_name) # line 5
