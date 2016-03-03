.. _working_with_halotools_provided_catalogs:

********************************************************
Downloading and caching Halotools-provided catalogs
********************************************************

This section of the documentation describes how to get up and running 
with the halo and particle catalogs provided by Halotools. To see 
if Halotools provides the catalogs you need, see the 
:ref:`supported_sim_list` page.   
If you want to use your own catalog and/or use Halotools to process 
an alternative catalog, see the :ref:`working_with_alternative_catalogs` page. 

Halotools provides a handful of homogeneously processed 
halo catalogs and associated particle data. These catalogs 
have been prepared into a standard form, and so 
once they are downloaded they will be directly added to your cache 
and can immediately be used for your science application. 

The class responsible for downloading and caching these 
catalogs is `~halotools.sim_manager.DownloadManager`. 

>>> from halotools.sim_manager import DownloadManager
>>> dman = DownloadManager()

Below appears a summary table of the snapshots available for download. 
See the :ref:`supported_sim_list` for further information about the snapshots. 
The information in the table below can be used to select the appropriate 
arguments to pass to the `~halotools.sim_manager.DownloadManager`. 

============  ===========================  ======================
simname          available halo-finders     available redshifts 
============  ===========================  ======================
bolshoi       rockstar, bdm                0.0, 0.5, 0.84, 2.00
bolplanck     rockstar                     0.0, 0.5, 1.00, 2.00
multidark     rockstar                     0.0, 0.5, 1.00, 2.15
consuelo      rockstar                     0.0, 0.5, 1.00, 2.00
============  ===========================  ======================

Using the convenience script 
===============================

There is also a convenience script providing 
command-line wrapper behavior around this class: 
scripts/download_additional_halocat.py. Whether you use 
the script or the `~halotools.sim_manager.DownloadManager` class, 
when you download the Halotools-provided catalogs, 
the download location will be stored in 
the cache log so that Halotools will remember where 
you stored them. For example, if you downloaded Multidark 
rockstar halos at *z = 1*, you can load this catalog into memory 
using the `~halotools.sim_manager.CachedHaloCatalog` class:

>>> from halotools.sim_manager import CachedHaloCatalog
>>> halocat = CachedHaloCatalog(simname = 'multidark', halo_finder = 'rockstar', redshift = 1) # doctest: +SKIP

The rest of this section of the documentation decribes how to use 
the `~halotools.sim_manager.DownloadManager` class. If you prefer 
to use the convenience script, run it from the command line and throw the 
help flag to see its calling sequence:

	python scripts/download_additional_halocat.py --help


.. _download_manager_usage_tutorial:

Usage tutorial for the `~halotools.sim_manager.DownloadManager` class
=========================================================================

The primary functionality of the `~halotools.sim_manager.DownloadManager` 
class lies in just two very similar methods: 
`~halotools.sim_manager.DownloadManager.download_processed_halo_table` and 
`~halotools.sim_manager.DownloadManager.download_ptcl_table`.  
The `~halotools.sim_manager.DownloadManager.download_processed_halo_table` method 
accepts three positional arguments: simname, halo_finder and redshift. 

>>> from halotools.sim_manager import DownloadManager
>>> dman = DownloadManager()
>>> dman.download_processed_halo_table('bolplanck', 'rockstar', 0.5) # doctest: +SKIP

The optional argument ``version_name`` can be used to specify which version of 
the catalogs you download, with default value set 
in the `~halotools.sim_manager.sim_defaults` module. The purpose of the ``version_name`` 
is to differentiate between the same simulation data processed in different ways. 
Processing differences could either occur at the level of halo-finding, or simply in 
the cuts placed on the original halo catalog. 

You can the download location with the ``download_dirname`` argument. By default, 
your halo catalogs will be downloaded to the Halotools cache directory:

$HOME/.astropy/cache/halotools/

You are free to store the halo catalogs in any location on disk that you like, 
though it will generally help ensure reproducibility if you 
do not move your halo catalogs around.  
For more information about managing the disk locations of your halo catalogs, 
see :ref:`relocating_simulation_data`. 

The `~halotools.sim_manager.DownloadManager.download_ptcl_table` method 
has the same API, where the only difference is that you do not specify 
a halo-finder when you want a particle table, as the particles were 
selected randomly without regard to halo membership. 













