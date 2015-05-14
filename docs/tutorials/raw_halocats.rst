
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

The primary method of the `~halotools.sim_manager.CatalogManager`
class that you will use is
`~halotools.sim_manager.CatalogManager.process_raw_halocat`. This
method does three things: 1. Reads the raw halo catalog ASCII either
from the cache or an alternative location 2. Optionally makes
customizable cuts on the rows of the halo catalog, returning a numpy
structured array 3. Optionally stores the cut catalog into cache, or
another directory location of your choosing

Let's use the Multidark file we just downloaded to see how
`~halotools.sim_manager.CatalogManager.process_raw_halocat` works.

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


Although this particular file processes almost instantly, this is not
the case for much larger catalogs, and so Halotools issues messages
describing the status of the reduction along the way.

In the above call to
`~halotools.sim_manager.CatalogManager.process_raw_halocat`, there
were three required positional arguments. The first is simply the
filename (including absolute path) that the method should use to look
for the ASCII data. The second two arguments, ``simname`` and
``halo_finder``, tell Halotools how to interpret the columns of data in
the file.

Under the hood, the ``simname`` and ``halo_finder`` trigger Halotools to
look for a `~halotools.sim_manager.HaloCat` object with matching
``simname`` and ``halo_finder``. If you want to use
`~halotools.sim_manager.CatalogManager` to process your halo
catalogs, you must either choose one of the supported combinations of
simulation/halo-finder, or write your own
`~halotools.sim_manager.HaloCat` object. This latter option is
quite straightforward, as the class pattern can be simply matched
against the existing `~halotools.sim_manager.HaloCat` objects; the
main component of the work in using your own simulation is simply
writing a ``dtype`` that specifies the keyname and data type for each
column in your ASCII data.

Processing options
------------------

Now let's unpack the remaining arguments to get a sense of what options
you have for how your ASCII data is processed.

1. Storing the processed catalog in cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting ``store_result`` to ``True`` triggers Halotools to create an
HDF5 file for the processed halo catalog and place it in your cache
directory. If you choose this option, you must also specify a
``version_name`` that will be used to create a unique filename for the
hdf5 file.

If a matching halo catalog with the same version name already exists in
the cache directory, then Halotools will not overwrite the existing
catalog unless you explicitly set the optional ``overwrite`` keyword
argument to ``True``.

If you set ``store_result`` to ``False``, or simply omit this keyword
argument, Halotools will not create an hdf5 file. In either case, the
`~halotools.sim_manager.CatalogManager.process_raw_halocat` method
will return two things:

::

    1. A structured numpy array containing the processed halo catalog
    2. The instance of the ~halotools.sim_manager.RockstarReader object used to read the catalog.

2. Specifying your catalog cuts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``cuts_funcobj`` keyword argument gives you the option to make any
cuts you like when reducing the raw halo catalog. Whatever cuts you
choose to make, as described below Halotools provides you with a
bookkeeping device to automatically keep track of the exact cuts you
used when creating a reduced binary file.

Whatever cuts you choose, the
`~halotools.sim_manager.CatalogManager.process_raw_halocat` method
applies the cuts as the raw ASCII is being read in a series of chunks.
This way, you do not need to have enough memory on your machine to load
the entire uncut catalog - all you need is enough memory to store the
post-processed catalog.

::

    Option 1: **Default cut**. If you do not pass the ``cuts_funcobj`` keyword argument to the ~halotools.sim_manager.CatalogManager.process_raw_halocat method, default cuts will be chosen for you. These default cuts are specified by the ~halotools.sim_manager.RockstarReader method of the ~halotools.sim_manager.RockstarReader. The current default cut is to throw out any halo or subhalo that never had more than 300 particles at any point in its past history. 

::

    Option 2: **No cut**. If you set the ``cuts_funcobj`` keyword argument to the string ``nocut``, then the ~halotools.sim_manager.CatalogManager.process_raw_halocat method will keep all rows. 

Note that for most science applications, the default 300-particle cut is
reasonably conservative. For many science targets, more stringent
completeness requirements are appropriate, in which case the additional
cuts can be applied post-processing with a boolean mask. However, this
simple cut alone dramatically reduces the size of the resulting binary
file, and so it is not recommended that you use the ``nocut`` option
unless you are confident that relaxing the 300-particle cut is a
necessity.

::

    Option 3: **Custom cut**. By passing a python function object to ``cuts_funcobj``, you have the freedom to make any cuts you like. We'll give an example of this usage below. The only requirements on the function object are as follows:
        i) The input is a numpy structured array with the same column names as the halo catalog, or fewer. 
        ii) The output is a boolean array of the same length as the input array.
        iii) The function is a callable object from the namespace in which ~halotools.sim_manager.CatalogManager.process_raw_halocat is called
        iv) The function is stand-alone, and not a bound instance method of some other object.

.. code:: python

    def example_custom_cut(x):
        return x['vmax'] > 200
.. code:: python

    custom_cut_halos, reader_obj = catman.process_raw_halocat(downloaded_fname, simname, halo_finder, 
                                        store_result=False, cuts_funcobj=example_custom_cut)


.. parsed-literal::

    ...uncompressing ASCII data
    
    ...Processing ASCII data of file: 
    /Users/aphearin/.astropy/cache/halotools/raw_halo_catalogs/multidark/rockstar/hlist_0.08820.list
     
     Total number of rows in file = 90
     Number of rows in detected header = 57 
    
    Reading catalog in a single chunk of size 90
    
    Total runtime to read in ASCII = 0.0 seconds
    
    ...re-compressing ASCII data


