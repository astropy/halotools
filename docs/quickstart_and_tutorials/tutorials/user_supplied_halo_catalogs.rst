:orphan:

.. _user_supplied_halo_catalogs:

**************************************************************
Instructions for Working with an Alternative Halo Catalog
**************************************************************

This section of the documentation describes how to get started 
with simulation data besides the Halotools-provided catalogs. 
If you want to store a new Rockstar catalog in the Halotools cache, 
see :ref:`reducing_and_caching_a_new_rockstar_catalog`. Or if you 
just want to read Rockstar ASCII data and not cache the reduction, 
see the docstring of the `~halotools.sim_manager.TabularAsciiReader` class. 
Below we describe how to transform halo catalogs other than those 
identified with Rockstar into a standard form that will 
work with the mock-population factories of Halotools, with or without caching the halos. 

.. _basic_usage_of_user_supplied_halo_catalog: 

Basic usage of the `~halotools.sim_manager.UserSuppliedHaloCatalog` class
============================================================================

In order to put your halo catalog into a standard form recognized by Halotools, 
the only thing you need to do is instantiate the 
`~halotools.sim_manager.UserSuppliedHaloCatalog` class 
by passing your simulation data and metadata to the constructor. 
Once you have an instance of `~halotools.sim_manager.UserSuppliedHaloCatalog`, 
the halo data is bound to the ``halo_table`` attribute of the instance in the form 
of an Astropy `~astropy.table.Table`; 
see the `~astropy.table.Table`documentation  
for more information about this data structure. 
Additional metadata about your simulation is also stored as attributes 
of the `~halotools.sim_manager.UserSuppliedHaloCatalog` instance. 
The docstring of `~halotools.sim_manager.UserSuppliedHaloCatalog` 
gives a detailed description of the arguments required by the constructor. 

.. _properly_formatting_user_supplied_halo_catalog_columns:

Notes on properly formatting halo properties 
==============================================

Note that every column of a halo catalog must begin with the ``halo_`` substring. 
The justification for this requirement is easy to understand. 
The basic task performed during the generation of a mock galaxy population is 
to create a ``galaxy_table``, which is an Astropy `~astropy.table.Table` storing the galaxy data. 
The ``galaxy_table`` contains a combination of properties assigned by the model 
and properties inherited by the underlying halos. 
No Halotools model is permitted to define a name for a galaxy property that begins 
with ``halo_``, and all halo catalog properties must begin with ``halo_``. 
This provides a simple way to guarantee that there there will be no conflicts 
between the column names of halo properties bound to a mock and the 
column names created by some Halotools model during mock-population. 

In principle, if you are able to instantiate the 
`~halotools.sim_manager.UserSuppliedHaloCatalog` class 
without raising an exception then this automatically guarantees that 
the instance can be passed to the Halotools model factories to populate 
mock galaxies into the catalog. However, different Halotools models 
require different halo properties to be available. For example, 
HOD-style models typically only use *host* halos, and so those models 
must make some assumption for how to identify which elements of your halo catalog 
are host halos, and which are subhalos. Other models such as age matching 
require knowledge of some measure of halo formation time. If you are interested in 
using your simulation with a specific galaxy-halo model, be sure to check the 
documentation of that model so that you can properly format your halo catalog. 


.. _storing_user_supplied_halo_catalog_in_cache:

Storing a `~halotools.sim_manager.UserSuppliedHaloCatalog` in cache 
=======================================================================

All instances of the `~halotools.sim_manager.UserSuppliedHaloCatalog` class 
have a `~halotools.sim_manager.UserSuppliedHaloCatalog.add_halocat_to_cache` 
method that can be used to store your catalog for convenient future use. 
If you call this method and no exceptions are raised, from then on 
you will be able to load your halo catalog into memory with the 
`~halotools.sim_manager.CachedHaloCatalog` class using the 
exact same syntax as you would use to load one of the Halotools-provided catalogs. 















