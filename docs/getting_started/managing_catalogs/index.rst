.. _cat_manage:

**********************************************************************
Managing simulation data (`halotools.sim_manager`)
**********************************************************************

**Overview**
======================

One of the most tedious tasks of simulation analysis 
is simply file management. Halo catalogs are very large, 
and they typically contain vast amounts of information that is 
irrelevant for your science target. Simulated data 
is commonly provided in ASCII format; this ensures that 
anyone can read the data, and also guarantees that 
the initial I/O will be a painful and slow process. 
It is thus widespread practice to read the "raw" ASCII data, 
apply some cuts, and then store the cut catalog as a binary file. 

The ``sim_manager`` sub-package of Halotools 
provides end-to-end bookeeping tools for this painstaking process. 
The features supported by the ``sim_manager`` include:

	* Downloading snapshots of unprocessed ASCII data from a range of publicly available sources to the Halotools cache directory. 

	* Reading these ASCII-formatted files into a python data structure, including on-the-fly cuts to minimize memory requirements. 

	* Storing the cut catalogs to the cache directory as fast-loading HDF5 binary files, automatically keeping track of which cuts were applied to which file. 

	* Loading the reduced catalogs into memory with a single, simple line of python code. 

**Beginner's instructions**
============================================

To see simple examples of how to manipulate the data stored in halo catalogs, 
see the Examples section of the `~halotools.sim_manager.HaloCatalog` API documentation. 

For beginner's instructions in using the ``sim_manager`` sub-package to download new catalogs 
and manage your cache, see :ref:`sim_manager_step_by_step`. 

**List of supported simulations**
============================================
To see what catalogs Halotools is pre-configured to work with, see :ref:`supported_sim_list`. If you have your own catalog you would like to work with instead, see the :ref:`using_your_own_catalog`. 


Using ``sim_manager``
======================

.. toctree::
   :maxdepth: 1

   sim_manager_step_by_step.rst
   alternate_catalogs.rst





