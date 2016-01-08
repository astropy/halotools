.. _working_with_halotools_provided_catalogs:

********************************************************
Downloading and Caching Halotools-Provided Catalogs
********************************************************

Halotools provides a handful of homogeneously processed 
halo catalogs and associated particle data. These catalogs 
have been prepared into a standard form, and so they can 
be directly added to your cache and loaded into memory 
as soon as they are downloaded. 

The class responsible for downloading and caching these 
catalogs is `~halotools.sim_manager.DownloadManager`. 