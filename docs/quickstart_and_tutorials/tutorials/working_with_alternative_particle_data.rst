:orphan:

.. _working_with_alternative_particle_data:

**************************************************************
Instructions for Working with Alternative Particle Data
**************************************************************

This section of the documentation describes how to 
put your collection of particles into a standard form 
and optionally store the particle data in your cache. 
The :ref:`storing_user_supplied_ptcl_catalog_in_cache` section 
below describes how to cache your particles for convenient future use. 
If you do not want to use the Halotools cache system, 
instead refer to the :ref:`using_user_supplied_ptcl_catalog_without_the_cache` section. 

This tutorial is intended to be read together with the 
docstring of the `~halotools.sim_manager.UserSuppliedPtclCatalog` class. 
Please refer to the docstring as you follow the explanation below. 


.. _basic_usage_of_user_supplied_ptcl_catalog: 

Basic usage of the `~halotools.sim_manager.UserSuppliedPtclCatalog` class
============================================================================

In order to put your particle catalog into a standard form recognized by Halotools, 
the only thing you need to do is instantiate the 
`~halotools.sim_manager.UserSuppliedPtclCatalog` class 
by passing your data and metadata to the constructor. 
Once you have an instance of `~halotools.sim_manager.UserSuppliedPtclCatalog`, 
the particle data is bound to the ``ptcl_table`` attribute of the instance in the form 
of an Astropy `~astropy.table.Table`; 
see the `~astropy.table.Table`documentation  
for more information about this data structure. 
Additional metadata about your simulation is also stored as attributes 
of the `~halotools.sim_manager.UserSuppliedPtclCatalog` instance. 
The docstring of `~halotools.sim_manager.UserSuppliedPtclCatalog` 
gives a detailed description of the arguments required by the constructor. 


.. _storing_user_supplied_ptcl_catalog_in_cache:

Storing a `~halotools.sim_manager.UserSuppliedPtclCatalog` in cache 
=======================================================================

All instances of the `~halotools.sim_manager.UserSuppliedPtclCatalog` class 
have a `~halotools.sim_manager.UserSuppliedPtclCatalog.add_ptclcat_to_cache` 
method that can be used to store your catalog for convenient future use. 
If you call this method and no exceptions are raised, from then on 
you will be able to use this collection of dark matter particles together with 
the associated `~halotools.sim_manager.CachedHaloCatalog`. 
The docstring of the `~halotools.sim_manager.UserSuppliedPtclCatalog` class 
provides an explicit worked example of how to use your 
particle data together with the `~halotools.sim_manager.CachedHaloCatalog` class. 


.. _using_user_supplied_ptcl_catalog_without_the_cache:

Using your particle catalog without the cache
=======================================================================

None of the functionality of Halotools requires you to use the caching system. 
The constructor of the `~halotools.sim_manager.UserSuppliedHaloCatalog` class 
accepts a ``user_supplied_ptclcat`` argument. This argument accepts an 
instance of the `~halotools.sim_manager.UserSuppliedPtclCatalog` class, 
allowing you to completely bypass the Halotools cache system. 
The docstring of the `~halotools.sim_manager.UserSuppliedHaloCatalog` class 
provides an explicit worked example for this feature. 















