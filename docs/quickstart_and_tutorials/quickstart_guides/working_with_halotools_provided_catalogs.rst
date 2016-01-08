.. _working_with_halotools_provided_catalogs:

********************************************************
Downloading and Caching Halotools-Provided Catalogs
********************************************************

Halotools provides a handful of homogeneously processed 
halo catalogs and associated particle data. These catalogs 
have been prepared into a standard form, and so 
once they are downloaded they will be directly added to your cache 
and can immediately be used for your science application. 

The class responsible for downloading and caching these 
catalogs is `~halotools.sim_manager.DownloadManager`. 
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





