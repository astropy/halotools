:orphan:

.. _getting_started:

******************************
Getting started with Halotools
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

After installing the code and its dependencies, navigate to some new working directory and execute the test suite. (This only needs to be done once per installed version.) For halotools versions v0.6 and later, you can use the `test_installation` feature that runs a few select tests scattered throughout the repository. For versions v0.5 and earlier, you will need to run the full test suite, which is much more CPU- and memory-intensive.

.. code:: python

    import halotools
    halotools.installation_test()  #  v0.6 and later
    halotools.test()  #  v0.5 and earlier

The full test suite is memory intensive and takes several minutes to run. It will generate a few small, temporary dummy files that you can delete or just ignore.

See :ref:`verifying_your_installation` for details about the message that prints after you run the test suite.

.. _download_default_halos:

Downloading the default halo catalog
-------------------------------------

Once you have installed Halotools and verified that you can import it,
likely the first thing you will want to do is to download the default
halo catalog so that you can quickly get up and running. You can accomplish
this with the ``download_initial_halocat.py`` script that is packaged as part
of your global install. To use it, navigate to any working directory
on your machine and run the script::

    $ download_initial_halocat.py -h

Throwing the ``-h`` flag will tell you the script's options. If you omit the ``-h`` flag,
this will download the default halo catalog to the default location on disk, the
Halotools cache location::

    $HOME/.astropy/cache/halotools/halo_catalogs

Alternatively, you can control the disk location of the download with the ``-dirname`` flag.

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
`~halotools.sim_manager.DownloadManager`, or use the **download_additional_halocat.py** convenience script, which should be called with four positional arguments: *simname, halo_finder, version_name* and *redshift.* For example, navigate to any working directory and execute::

    $ download_additional_halocat.py multidark rockstar most_recent 0.5

Choosing ``most_recent`` as the version_name automatically selects the most up-to-date version of the Halotools-provided catalogs. You can read about your download options by executing the script and throwing the help flag::

    $ download_alternate_halocats.py --help


Getting started with galaxy/halo analysis
===========================================

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
>>> from halotools.sim_manager import CachedHaloCatalog
>>> halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 0.5) # doctest: +SKIP
>>> zheng07_model.populate_mock(halocat) # doctest: +SKIP
>>> r, xi_gg = zheng07_model.compute_average_galaxy_clustering() # doctest: +SKIP

The `~halotools.empirical_models.ModelFactory.compute_average_galaxy_clustering` of any model repeatedly populates a halo catalog with mock galaxies and returns the average clustering signal in each separation bin. As described in the docstring, this function has many optional keyword arguments. In the following example call, we'll show how to calculate the auto-clustering of centrals and satellites, as well as the cross-correlation between the two, using the maximum number of cores available on your machine.

>>> r, xi_cc, xi_cs, xi_ss = zheng07_model.compute_average_galaxy_clustering(gal_type = 'centrals', include_crosscorr = True, num_iterations = 3, num_threads = 'max') # doctest: +SKIP

For a comprehensive list of pre-built models provided by Halotools, see :ref:`preloaded_models_overview`. For a sequence of worked examples showing how to use Halotools to analyze mock galaxy catalogs, see :ref:`galaxy_catalog_analysis_tutorial`.

Designing your own galaxy-halo model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Halotools has special factories that allow you to build your own model of the galaxy-halo connection. The foundation of this factory pattern is the modular design of the ``empirical_models`` sub-package.

Galaxy-halo models are broken down into a set of independently-defined *component models*. For example, the `~halotools.empirical_models.NFWProfile` class is a component model governing the spatial distribution of satellite galaxies within their halos, and the `~halotools.empirical_models.Tinker13Cens` class is a component model controlling the stellar-to-halo mass relation of quenched and star-forming central galaxies. To build your own model, you choose a collection of component models and compose them together into a *composite model* using the appropriate Halotools factory class: `~halotools.empirical_models.HodModelFactory` for HOD-style models and `~halotools.empirical_models.SubhaloModelFactory` for abundance matching-style models.

Composing together different collections of components gives you a large amount of flexibility to construct highly complex models of galaxy evolution. There are no limits on the number of component models you can use, nor on the number or kind of galaxy population(s) that make up the universe in your composite model.

In choosing component models, you are not restricted to choose from the set of features that ship with the Halotools package. You are welcome to write your own component models and use the Halotools factories to build the composite, to write just one new component model and include it in a collection of Halotools-provided components, or anywhere in between. This way, if you are mostly interested in a specific feature of the galaxy population, you can focus exclusively on developing code for that one feature, and use existing Halotools components to model the remaining features.

For a step-by-step guide and many worked examples of how to build a customized model that is tailored to your interests, see :ref:`model_building`.

Making mock observations
-------------------------

The ``mock_observables`` sub-package provides a large collection of heavily optimized functions for calculating commonly encountered astronomical statistics.

>>> from halotools import mock_observables # doctest: +SKIP

To list a few examples of functions you can use the ``mock_observables`` sub-package to calculate:

    1.  the projected correlation function, `~halotools.mock_observables.wp`,

    2. the pairwise line-of-sight velocity dispersion, `~halotools.mock_observables.los_pvd_vs_rp`,

    3. marked correlation functions with highly customizable weights, `~halotools.mock_observables.marked_tpcf`,

    4. galaxy-galaxy lensing, `~halotools.mock_observables.mean_delta_sigma`,

    5. friends-of-friends group identification, `~halotools.mock_observables.FoFGroups`.

These functions take simple point data as input. This means that the ``mock_observables`` sub-package not only works with Halotools models and catalogs, but also equally well with hydrodynamical simulation outputs or mocks based on semi-analytic models that have no connection to Halotools. See `~halotools.mock_observables` for a comprehensive list of functions you can choose from, and :ref:`galaxy_catalog_analysis_tutorial` for example usages with mock galaxy catalogs.








