
.. _raw_halocats_tutorial:

*************************************************
Tutorial on managing raw ASCII halo catalog data
*************************************************

The goal of this tutorial is to teach you how to use Halotools to start
from scratch by downloading raw ASCII halo catalog data and producing
your own reduced binary, appropriately processed for your science
target. Before following this tutorial, you will probably find it
helpful to first read the :ref:`using_halocat_binaries`, so that you
have a sense of what the end product will be like to work with.

Downloading the raw halo catalog
--------------------------------

The first thing we'll do is to load the Catalog Manager, which takes
charge of handling all the bookkeeping of simulation files and
processing.

.. code:: python

    from halotools import sim_manager
    catman = sim_manager.CatalogManager()

In this section, we'll download some raw ascii data from one of the
publicly available data sources on the web. Let's use the
`~halotools.sim_manager.CatalogManager.available_halocats` method
to take a look at what options we have for the catalogs:

.. code:: python

    for simname, halo_finder in catman.available_halocats:
        print(simname, halo_finder)

.. parsed-literal::

    ('bolshoi', 'rockstar')
    ('bolshoipl', 'rockstar')
    ('bolshoi', 'bdm')
    ('multidark', 'rockstar')
    ('consuelo', 'rockstar')


Ok, so let's suppose we're interested in downloading the
highest-available redshift of Rockstar halos from the Multidark
simulation. We need to tell the
`~halotools.sim_manager.CatalogManager` the specific redshift we
want to download, so let's use the
`~halotools.sim_manager.CatalogManager.available_redshifts` method
to see which specific snapshots are available for Multidark:

.. code:: python

    simname, halo_finder = 'multidark', 'rockstar'
    location = 'web'
    catalog_type = 'raw_halos' # This specifies that we want the original halo catalogs, not a pre-processed binary
    
    redshift_list = catman.available_redshifts(location, catalog_type, simname, halo_finder)
    desired_redshift = max(redshift_list)
    print("desired redshift = %.2f " % desired_redshift)


.. parsed-literal::

    desired redshift = 10.34 


Now that we know what redshift we want, we can use the
`~halotools.sim_manager.CatalogManager.download_raw_halocat`
method to find the catalog on the web and download it to our cache
directory:

.. code:: python

    downloaded_fname = catman.download_raw_halocat(simname, halo_finder, desired_redshift)


.. parsed-literal::

    The following filename already exists in your cache directory: 
    
    /Users/aphearin/.astropy/cache/halotools/raw_halo_catalogs/multidark/rockstar/hlist_0.08820.list.gz
    
    If you really want to overwrite the file, 
    you must call the same function again 
    with the keyword argument `overwrite` set to `True`


The highest-redshift Rockstar catalog for Multidark is now in your cache
directory. You can verify this using the
`~halotools.sim_manager.CatalogManager.check_for_existing_halocat`
method. This method returns ``False`` if no catalog is detected; if a
matching catalog is detected, the filename (including absolute path) is
returned.

.. code:: python

    check_location = 'cache'
    test_already_exists = catman.check_for_existing_halocat('cache', catalog_type, 
                                      simname=simname, halo_finder=halo_finder, 
                                      redshift=desired_redshift)
    print(test_already_exists)
    downloaded_fname = test_already_exists

.. parsed-literal::

    /Users/aphearin/.astropy/cache/halotools/raw_halo_catalogs/multidark/rockstar/hlist_0.08820.list.gz


Success! We're now in business with a newly downloaded halo catalog.

Processing the raw halo catalog into a reduced binary
-----------------------------------------------------

ASCII data is a relatively slow file format to load into memory,
particularly for large files such as halo catalogs. So in this section
we'll describe how to convert a raw halo catalong into a fast-loading
HDF5 file, and store it in your cache directory for future use.

.. code:: python

    result = catman.process_raw_halocat(downloaded_fname, simname, halo_finder, 
                                        store_result=True, overwrite=True, 
                                        version_name='dummy', cuts_funcobj='nocut')


.. parsed-literal::

    ...uncompressing ASCII data
    
    ...Processing ASCII data of file: 
    /Users/aphearin/.astropy/cache/halotools/raw_halo_catalogs/multidark/rockstar/hlist_0.08820.list
     
     Total number of rows in file = 90
     Number of rows in detected header = 57 
    
    Reading catalog in a single chunk of size 90
    
    Total runtime to read in ASCII = 0.0 seconds
    
    ...re-compressing ASCII data
    Storing reduced halo catalog in the following location:
    /Users/aphearin/.astropy/cache/halotools/halo_catalogs/multidark/rockstar/hlist_0.08820.list.dummy.hdf5


