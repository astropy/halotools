
.. _using_halocat_binaries:

*************************************************
Tutorial on managing pre-processed halo catalogs
*************************************************

If you are following this tutorial, you should first have downloaded the
default Halotools catalog by following the instructions given in the
:ref:`first_steps` section of the documentation. Follow those
instructions now if you have not done so already.

Working with the default pre-processed snapshot
-----------------------------------------------

To get warmed up, let's see how to work with the default snapshot
provided by Halotools. After importing the
`~halotools.sim_manager` sub-package, you can load the default
snapshot into memory with a single line of code:

.. code:: python

    from halotools import sim_manager
    default_snapshot = sim_manager.ProcessedSnapshot()


.. parsed-literal::

    Loading halo catalog with the following absolute path: 
    /Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolshoi/rockstar/hlist_1.00030.list.halotools.official.version.hdf5
    


The `~halotools.sim_manager.ProcessedSnapshot` is the primary
class you will use when working with halo catalogs. When you instantiate
this class, as in the second line of code above, Halotools first
searches for the relevant halo catalog in your cache directory. Since
you called `~halotools.sim_manager.ProcessedSnapshot` with no
arguments, the default snapshot is chosen.

The halo catalog is attached to the snapshot object in the form of the
``halos`` attribute:

.. code:: python

    print(default_snapshot.halos[0:4])


.. parsed-literal::

    scale    haloid   scale_desc ... mvir_firstacc vmax_firstacc vmax_mpeak
    ------ ---------- ---------- ... ------------- ------------- ----------
    1.0003 3060299107        0.0 ...     1.643e+14        952.39     952.39
    1.0003 3060312953        0.0 ...     1.589e+14        823.11     823.11
    1.0003 3058440575        0.0 ...     1.144e+14        799.42     799.42
    1.0003 3058441456        0.0 ...     9.709e+13        679.37     679.37


The data structure behind the scenes of the ``halos`` attribute of
``default_snapshot`` is an Astropy Table. We'll give a few simple
examples illustrating how to manipulate Astropy Tables below, but for
more detailed information about this data structure, see
`astropy.table`.

The ``default_snapshot`` also has the halo catalog and simulation
metadata bound to it. Here are a few examples:

.. code:: python

    print("Simulation name = %s " % default_snapshot.simname)
    print("Halo-finder = %s " % default_snapshot.halo_finder)
    print("Snapshot redshift = %.1f " % default_snapshot.redshift)


.. parsed-literal::

    Simulation name = bolshoi 
    Halo-finder = rockstar 
    Snapshot redshift = -0.0 


There is also metadata describing details of how the catalog was
produced:

.. code:: python

    print("Source of original data = %s " % default_snapshot.original_data_source)
    print("Time of original reduction = %s " % default_snapshot.time_of_original_reduction)
    print("Description of applied cuts = \n%s " % default_snapshot.cuts_description)


.. parsed-literal::

    Source of original data = http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/ 
    Time of original reduction = 2015-05-08 17:49:14 
    Description of applied cuts = 
    The only cut on the original catalog made by  the default_halocat_cut method of RockstarReader is to throw out all (sub)halos with Mpeak < 300 particles 


This metadata is also bound to the hdf5 files themselves, so that both
the `~halotools.sim_manager.ProcessedSnapshot` and the binary file
itself are self-expressive regarding exactly how they were generated.

Downloading other pre-processed snapshots
=========================================

Up until now, you have been working with the default snapshot downloaded
by the startup script ``download_initial_halocat``. However, the
Halotools team also provides other pre-processed snapshots to choose
from. To see which ones, you need to use the Catalog Manager:

.. code:: python

    catman = sim_manager.CatalogManager()

First, let's take a look at which combinations and halo-finders are
supported by the package:

.. code:: python

    halocat_list = catman.available_halocats
    for simname, halo_finder in halocat_list:
        print(simname, halo_finder)
        

.. parsed-literal::

    ('bolshoi', 'rockstar')
    ('bolshoipl', 'rockstar')
    ('bolshoi', 'bdm')
    ('multidark', 'rockstar')
    ('consuelo', 'rockstar')


Each simulation/halo-finder combination is actually composed of a
collection of many, many publicly available snapshots. To see which
snapshots have been pre-processed, we'll use the
`~halotools.sim_manager.CatalogManager.available_redshifts` method
of the `~halotools.sim_manager.CatalogManager`:

.. code:: python

    location = 'web'
    catalog_type = 'halos'
    simname = 'bolshoi'
    halo_finder = 'rockstar'
    redshift_list = catman.available_redshifts(location, catalog_type, simname, halo_finder)
    for z in redshift_list:
        print("z = %.2f " % z)
        

.. parsed-literal::

    z = 2.03 
    z = 0.98 
    z = 0.49 
    z = -0.00 


So for this combination of simulation/halo-finder, we have four options
to choose from for our pre-processed snapshot. To download the z=2
snapshot:

.. code:: python

    desired_redshift = 2.03
    catman.check_for_existing_halocat('cache', 'halos', simname, halo_finder, 
                                      redshift=desired_redshift)




.. parsed-literal::

    u'/Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolshoi/rockstar/hlist_0.33030.list.halotools.official.version.hdf5'



.. code:: python

    catman.download_preprocessed_halo_catalog(simname, halo_finder, desired_redshift)


.. parsed-literal::

    The following filename already exists in your cache directory: 
    
    /Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolshoi/rockstar/hlist_0.33030.list.halotools.official.version.hdf5
    
    If you really want to overwrite the file, 
    you must call the same function again 
    with the keyword argument `overwrite` set to `True`


In this case, Halotools detected that the pre-processed halo catalog was
actually already stored in my cache directory, so there was no need to
download the catalog. If you are following this tutorial for the first
time, the download would proceed.

Now that your z=2 catalog is in cache, you can load it into memory just
as before by using the `~halotools.sim_manager.ProcessedSnapshot`
method:

.. code:: python

    z2_snapshot = sim_manager.ProcessedSnapshot(simname, halo_finder, desired_redshift)


.. parsed-literal::

    Loading halo catalog with the following absolute path: 
    /Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolshoi/rockstar/hlist_0.33030.list.halotools.official.version.hdf5
    


Concluding notes
----------------

If you anticipate studying how your science targets depend on redshift,
cosmology, or halo-finding, you may find it useful to use the
`~halotools.sim_manager.CatalogManager.download_all_default_catalogs`
method. This will download the following pre-processed snapshots to your
cache directory:

::

    * Rockstar-based Bolshoi halos at z = 0, 0.5, 1, and 2
    * BDM-based Bolshoi halos at z=0
    * Rockstar-based Bolshoi-Planck halos at z=0

These catalogs will occupy a total of ~3Gb of disk space on your
machine.

There are two other convenience methods that are worthy of special
mention here. The first is
`~halotools.sim_manager.CatalogManager.check_for_existing_halocat`,
which checks your cache for an existing catalog. And the second is
`~halotools.sim_manager.CatalogManager.all_halocats_in_cache`,
which lists all cached catalogs of a given type. Refer to the
:ref:`sim_manager_api` section of the documentation to see how to call
these and other methods of the
`~halotools.sim_manager.CatalogManager`.

