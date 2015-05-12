
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

.. code:: python

    from halotools import sim_manager
    catman = sim_manager.CatalogManager()
.. code:: python

    simname = 'multidark'
    halo_finder = 'rockstar'
    location = 'web'
    catalog_type = 'raw_halos'
    redshift_list = catman.available_redshifts(location, catalog_type, simname, halo_finder)
    desired_redshift = max(redshift_list)
    print("desired redshift = %.2f " % desired_redshift)


.. parsed-literal::

    desired redshift = 10.34 


.. code:: python

    catman.check_for_existing_halocat('cache', catalog_type, 
                                      simname=simname, halo_finder=halo_finder, 
                                      redshift=desired_redshift)


.. parsed-literal::

    No raw multidark halo catalog has 
    a redshift within 0.10 of the input_redshift = 10.34.
     The closest redshift for these catalogs is 8.83




.. parsed-literal::

    False



.. code:: python

    downloaded_fname = catman.download_raw_halocat(simname, halo_finder, desired_redshift)


.. parsed-literal::

    
    ... Downloading data from the following location: 
    http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/hlist_0.08820.list.gz
    
     ... Saving the data with the following filename: 
    /Users/aphearin/.astropy/cache/halotools/raw_halo_catalogs/multidark/rockstar/hlist_0.08820.list.gz
    
     100.0% of 8051 bytes

The highest-redshift Rockstar catalog for Multidark is now in your cache
directory. You can verify this as follows:

.. code:: python

    check_location = 'cache'
    catman.check_for_existing_halocat('cache', catalog_type, 
                                      simname=simname, halo_finder=halo_finder, 
                                      redshift=desired_redshift)




.. parsed-literal::

    u'/Users/aphearin/.astropy/cache/halotools/raw_halo_catalogs/multidark/rockstar/hlist_0.08820.list.gz'



