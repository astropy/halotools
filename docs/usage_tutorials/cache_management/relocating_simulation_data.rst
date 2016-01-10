:orphan:

.. _relocating_simulation_data:

********************************************************
Relocating Simulation Data and Updating the Cache
********************************************************

This section of the documentation describes how you can 
update the Halotools cache log in the event that you 
move an already-cached halo catalog to a new location on disk. 
The :ref:`relocating_simulation_data_instructions` section 
tells you what to do, and the 
:ref:`relocating_simulation_data_explanation` section 
explains what is going on under the hood. 


.. _relocating_simulation_data_instructions:

Instructions for relocating simulation data
==============================================

The normal way to load a `~halotools.sim_manager.CachedHaloCatalog` is to 
pass a set of metadata to the constructor:

>>> halocat = CachedHaloCatalog(simname = simname, halo_finder = halo_finder, version_name = version_name, redshift = redshift) # doctest: +SKIP

However, if you have moved your halo catalog, then the cache log no longer 
points to the correct path. To load the relocated halo catalog and 
simultaneously update the cache log, just pass the absolute path of the 
relocated hdf5 file to the `~halotools.sim_manager.CachedHaloCatalog`, 
and set **update_cached_fname** to *True*:

>>> halocat = CachedHaloCatalog(fname = abs_path_to_hdf5_file, update_cached_fname = True) # doctest: +SKIP

For reasons described in :ref:`relocating_simulation_data_explanation`, this will only 
work with previously-cached halo catalogs that have been relocated. 


.. _relocating_simulation_data_explanation:

Explanation of the underlying source code 
================================================

All hdf5 files storing cached simulation data have 
metadata bound to them that helps protect 
against bookkeeping-related bugs. The `h5py <http://h5py.org/>`_ 
package manages all metadata with the following dictionary-like sytnax:

>>> import h5py # doctest: +SKIP
>>> fileobj = h5py.File(fname) # doctest: +SKIP
>>> list_of_metadata_keys = fileobj.attrs.keys() # doctest: +SKIP
>>> metadata_value = fileobj.attrs[metadata_key] # doctest: +SKIP

The hdf5 file of all cached halo catalogs 
have a **fname** metadata key. At the time 
each catalog is cached, the **fname** metadata of the hdf5 file 
is in agreement with the corresponding row and column of the 
Halotools cache log, which is stored as ASCII data in the following location:

	$HOME/.astropy/cache/halotools/halo_table_cache_log.txt

The path in the **fname** column of the cache log ASCII file 
is the location where `~halotools.sim_manager.HaloTableCache` class 
will go looking for the catalog. Whenever you load an instance 
of the `~halotools.sim_manager.CachedHaloCatalog` class by passing 
it metadata such as a **simname**, what happens is that 
the `~halotools.sim_manager.HaloTableCache` searches 
**halo_table_cache_log.txt** for a row with matching metadata. 
The **fname** column in the matching row is then treated as the 
absolute path to the hdf5 file where the halo data is stored. 
The `~halotools.sim_manager.CachedHaloCatalog` class then 
inspects the hdf5 file stored in that location, and if the 
**fname** metadata key of that file does not match the 
actual **fname**, the `~halotools.sim_manager.HaloTableCache` class 
raises an exception. Halotools intentionally makes it difficult 
to move simulation data around willy-nilly, as this is a common cause 
of buggy behavior in simulation analysis. 

There are nonetheless perfectly good reasons to relocate simulation data 
to new disk locations, and as described in the 
:ref:`relocating_simulation_data_instructions`, 
this can be accomplished with the **fname** keyword argument of the 
`~halotools.sim_manager.CachedHaloCatalog` class. When you use the 
**fname** keyword argument with **update_cached_fname** set to *True*, 
this triggers the following sequence of events:

	1. The h5py package is used to over-write the **fname** metadata of the hdf5 file. 

	2. The `~halotools.sim_manager.HaloTableCache` class deletes the appropriate row of **halo_table_cache_log.txt** and adds a new row with the new **fname**. 

From now on you can go back to loading this halo catalog into memory by 
passing in metadata to the `~halotools.sim_manager.CachedHaloCatalog` class constructor. 























