:orphan:

.. _getting_started:

******************************
Getting Started with Halotools
******************************

Importing Halotools
===================

After installing halotools (see :ref:`step_by_step_install` for detailed instructions), 
you can open up a python terminal and load the entire package by::

    >>> import halotools

However, most of the functionality of halotools is divvied into 
sub-packages, and so it is better to load only the sub-package 
that you need for your application. You can do this with the following syntax::

    >>> from halotools import some_subpackage  # doctest: +SKIP

.. _first_steps:

First steps with Halotools
================================

Running the test suite
------------------------
After installing the code, you should navigate to the root directory of the package and run the full test suite to make sure your copy of Halotools is science-ready:

	python setup.py test

Depending on how you have configured your copy of the gcc compiler, Mac users may need to instead run 

	CC=clang python setup.py test 

This will trigger Cython compilation of various package components, 
and the execution of every function in the test suite of every sub-package, 
generating a large number of compiler warnings that you can ignore. 
Runtime for test suite execution typically takes less than a minute. 
At the end, you will see a short summary of the outcome of the test suite. 

.. _download_default_halos:

Downloading the default halo catalog
-------------------------------------

Once you have installed Halotools and verified that you can import it,
likely the first thing you will want to do is to download the default 
halo catalog so that you can quickly get up and running. You can accomplish 
this by navigating to the root directory of the package and running the initial 
download script::

	python scripts/download_initial_halocat.py

Running this script will set up the Halotools cache directory system on your local machine, 
and then download the default halo catalog to the cache (Bolshoi z=0 rockstar halos), 
storing the catalog as a pre-processed hdf5 file. The default catalog is ~400Mb. 
Because this script automatically updates the Halotools cache with the catalog, 
you can now load these halos into memory using the `~halotools.sim_manager.CachedHaloCatalog` class: 

>>> from halotools import sim_manager 
>>> default_halocat = sim_manager.CachedHaloCatalog() # doctest: +SKIP
>>> print(default_halocat.halo_table[0:9]) # doctest: +SKIP

To see simple examples of how to manipulate the data stored in halo catalogs, 
see the Examples section of the `~halotools.sim_manager.CachedHaloCatalog` documentation. 

If you wish to download alternate snapshots, you can either use the 
`~halotools.sim_manager.DownloadManager`, or use the download_additional_halocat.py convenience script, which should be called with four arguments: simname, halo_finder, version_name and redshift. For example::

	python scripts/download_additional_halocat.py multidark rockstar most_recent 0.5

Choosing ``most_recent`` as the version_name automatically selects the most up-to-date version of the Halotools-provided catalogs. You can read about your download options by executing the script and throwing the help flag::

	python scripts/download_alternate_halocats.py --help


Getting started with subpackages
================================

Although the different sub-packages of Halotools are woven together for the science aims of the package (see :ref:`halotools_science_overview`), individually the sub-packages have very different functionality. 

Downloading and processing simulations
---------------------------------------

The Halotools ``sim_manager`` sub-package  
makes it easy to download halo catalogs, process them into fast-loading binaries, 
store them in a cache directory of your choosing, and swap back and forth between 
different simulations using a single line of code. 

	>>> from halotools import sim_manager


Building models and making mocks
------------------------------------

To get started with building models and making mocks, you can import the empirical modeling sub-package::

>>> from halotools import empirical_models 

For an outline of how to generate mock galaxy catalogs, see :ref:`mock_making_quickstart`. 

To learn how to build a customized model that is tailored to your interests, see :ref:`model_building`. 

Making mock observations 
-------------------------

The ``mock_observables`` sub-package provides a virtual observatory for your synthetic galaxy population::

>>> from halotools import mock_observables # doctest: +SKIP





