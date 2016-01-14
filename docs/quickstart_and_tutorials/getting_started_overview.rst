:orphan:

.. _getting_started:

******************************
Getting Started with Halotools
******************************

This 10-minute guide gives an overview of the functionality of Halotools 
and each of its sub-packages. You can find links to more detailed information in 
each of the subsections below. This getting-started guide assumes you have 
already followed the :ref:`step_by_step_install` section of the documentation to get the package 
and its dependencies set up on your machine. 

Importing Halotools
===================

After installing Halotools you can open up a python terminal and load the entire package by:

    >>> import halotools

However, most of the functionality of halotools is divvied into 
sub-packages, and so it is better to load only the sub-package 
that you need for your application. You can do this with the following syntax:

    >>> from halotools import some_subpackage  # doctest: +SKIP

We will cover each of the main sub-packages in the documentation below, but first 
we'll show how to ensure that your install was successful and how to 
get up and running with the default halo catalog. 

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

Although the different sub-packages of Halotools are woven together for the specialized science aims of the package (see :ref:`halotools_science_overview`), individually the sub-packages have very different functionality. The sections below give a broad-brush overview of the functionality of each sub-package as well as links to quickstart guides and tutorials containing more detailed instructions. 

Working with simulation data
------------------------------------------------------

The Halotools ``sim_manager`` sub-package  
makes it easy to download Halotools-provided halo catalogs, 
process them into fast-loading binaries with self-expressive metadata, 
create a persistent memory of each catalog's disk location, and swap back and forth between 
different simulations. 

    >>> from halotools import sim_manager

See the :ref:`supported_sim_list` section of the documentation for information about the catalogs that come with the package. 

The full functionality of Halotools is available for use with halos in any N-body simulation identified with any halo-finder. For example, the `~halotools.sim_manager.RockstarHlistReader` class in the ``sim_manager`` sub-package  provides a memory-efficient tool for reading any Rockstar-produced ASCII data and storing the processed halo catalog in cache.

>>> from halotools.sim_manager import RockstarHlistReader

See :ref:`reducing_and_caching_a_new_rockstar_catalog` for more information. 

If you want to work with halo catalog ASCII data produced by a different halo finder, and/or if you want to reduce some N-body ASCII data but do not wish to use the Halotools cache system, you can use the stand-alone `~halotools.sim_manager.TabularAsciiReader` class instead. For more information about using Halotools with your own simulation data, see :ref:`working_with_alternative_catalogs`. 

For information about how to get started using Halotools to analyze N-body simulations and halo catalogs, see :ref:`halo_catalog_analysis_quickstart`. 


Building models and making mocks
------------------------------------

Pre-built models provided by Halotools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``empirical_models`` sub-package implements many pre-built models of the galaxy-halo connection. These models have been methodically tested against the publication upon which they are based and can be used out-of-the-box to make mock catalogs and generate observational predictions. 

Consider the HOD-style model used in `Zheng et al 2007 <http://arxiv.org/abs/astro-ph/0703457/>`_ to fit the clustering of DEEP2 and SDSS galaxies:

>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> zheng07_model = PrebuiltHodModelFactory('zheng07', threshold = -19.5, redshift = 0.5)
>>> zheng07_model.populate_mock(simname = 'bolshoi', redshift = 0.5) # doctest: +SKIP
>>> r, xi_gg = zheng07_model.compute_average_galaxy_clustering() # doctest: +SKIP

As an additional example, consider the abundance matching-style model introduced in `Behroozi et al 2010 <http://arxiv.org/abs/1001.0015/>`_:

>>> from halotools.empirical_models import PrebuiltSubhaloModelFactory
>>> behroozi_model = PrebuiltSubhaloModelFactory('behroozi10', redshift = 0)
>>> r, xi_gm = behroozi_model.compute_average_galaxy_matter_cross_clustering() # doctest: +SKIP

For a comprehensive list of pre-built models provided by Halotools, see :ref:`preloaded_models_overview`. For an overview of how to get started with mock galaxy catalogs, see :ref:`mock_making_quickstart`. 

Designing your own galaxy-halo model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To learn how to build a customized model that is tailored to your interests, see :ref:`model_building`. 

Making mock observations 
-------------------------

The ``mock_observables`` sub-package provides a large collection of functions you can use both to study halo catalogs and generate predictions of Halotools models that can be directly compared to observational data: 

>>> from halotools import mock_observables # doctest: +SKIP





