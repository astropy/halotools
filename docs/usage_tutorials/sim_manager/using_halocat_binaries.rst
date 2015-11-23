:orphan:

.. _using_halocat_binaries:

*************************************************
Tutorial on managing pre-processed halo catalogs
*************************************************

If you are following this tutorial, you should first have downloaded the
default Halotools catalog by following the instructions given in the
:ref:`first_steps` section of the documentation. Follow those
instructions now if you have not done so already.

Working with the default pre-processed snapshot
===============================================

To get warmed up, let's see how to work with the default snapshot
provided by Halotools. After importing the
`~halotools.sim_manager` sub-package, you can load the default
snapshot into memory with a single line of code:

.. code:: python

    from halotools import sim_manager
    default_snapshot = sim_manager.HaloCatalog()


.. parsed-literal::

    Loading halo catalog with the following absolute path: 
    /Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolshoi/rockstar/hlist_1.00030.list.halotools.official.version.hdf5
    


The `~halotools.sim_manager.HaloCatalog` is the primary class you
will use when working with halo catalogs. When you instantiate this
class, as in the second line of code above, Halotools first searches for
the relevant halo catalog in your cache directory. Since you called
`~halotools.sim_manager.HaloCatalog` with no arguments, the
default snapshot is chosen.

The halo catalog is attached to the snapshot object in the form of the
``halo_table`` attribute:

.. code:: python

    print(default_snapshot.halo_table[0:4])


.. parsed-literal::

    scale    haloid   scale_desc ... mvir_firstacc vmax_firstacc vmax_mpeak
    ------ ---------- ---------- ... ------------- ------------- ----------
    1.0003 3060299107        0.0 ...     1.643e+14        952.39     952.39
    1.0003 3060312953        0.0 ...     1.589e+14        823.11     823.11
    1.0003 3058440575        0.0 ...     1.144e+14        799.42     799.42
    1.0003 3058441456        0.0 ...     9.709e+13        679.37     679.37


The data structure behind the scenes of the ``halo_table`` attribute of
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
    print("Description of applied cuts = \n%s " % default_snapshot.cuts_description)


.. parsed-literal::

    Simulation name = bolshoi 
    Halo-finder = rockstar 
    Snapshot redshift = -0.0 


This metadata is also bound to the hdf5 files themselves, so that both
the `~halotools.sim_manager.HaloCatalog` and the binary file
itself are self-expressive regarding exactly how they were generated.

The default snapshot also comes with a randomly selected downsampling of
~1e6 dark matter particles, which you can access via the ``ptcl_table``
attribute:

.. code:: python

    print(default_snapshot.ptcl_table[0:4])

Downloading other halo catalogs
===============================

Up until now, you have been working with the default snapshot downloaded
by the startup script ``download_initial_halocat.py``. However, the
Halotools team also provides other pre-processed snapshots to choose
from. To see which ones, you need to use the Catalog Manager:

.. code:: python

    catman = sim_manager.CatalogManager()

To see which snapshots have been pre-processed for a given simulation:

.. code:: python

    catlist = catman.processed_halo_tables_available_for_download(simname='bolshoi', halo_finder='rockstar')    
    for fname in catlist:
        print fname
        

.. parsed-literal::

    z = 2.03 
    z = 0.98 
    z = 0.49 
    z = -0.00 


Halotools keeps the same filenames for each processed catalog to
maintain consistency with the original data sources; the convention is
that the scale factor of the snapshot is part of the ``hlist_``
filename.

So for this combination of simulation/halo-finder, we have four options
to choose from for our pre-processed snapshot. To download the z=2
snapshot:

.. code:: python

    catman.download_processed_halo_table(simname='bolshoi', halo_finder='rockstar', desired_redshift=2)


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
as before by using the `~halotools.sim_manager.HaloCatalog`
method:

.. code:: python

    z2_snapshot = sim_manager.HaloCatalog(simname='bolshoi', halo_finder='rockstar', desired_redshift=2)


.. parsed-literal::

    Loading halo catalog with the following absolute path: 
    /Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolshoi/rockstar/hlist_0.33030.list.halotools.official.version.hdf5
    

