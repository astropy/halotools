=========
Halotools
=========

Halotools is a specialized python package for building and testing models of the galaxy-halo connection, and analyzing catalogs of dark matter halos.
The core feature of Halotools is a modular platform for creating mock universes of galaxies starting from a catalog of dark matter halos obtained from a cosmological simulation. Functionality of the package includes:

* Fast generation of synthetic galaxy populations using HODs, abundance matching, and related methods
* Efficient algorithms for calculating galaxy clustering, lensing, z-space distortions, and other astronomical statistics
* A modular, object-oriented framework for designing your own galaxy evolution model
* End-to-end support for downloading publicly-available halo catalogs and reducing them to fast-loading hdf5 files

The code is publicly available at https://github.com/astropy/halotools.

Installation
-------------
The simplest way to install the latest release of the code is with conda-forge::

    conda install -c conda-forge halotools

Or alternatively, you can install using pip::

    		pip install halotools

You can find detailed installation instructions
in the **Package Installation** section of http://halotools.readthedocs.io. After installing the package, you should navigate to the *Quickstart Guides and Tutorials* section and follow the *Getting started with Halotools* 10-minute tutorial. This will get you set up with the default halo catalog so that you can quickly get started with creating mock galaxy populations.


Documentation
-------------
The latest build of the documentation can be found at http://halotools.readthedocs.io. The documentation includes installation instructions, quickstart guides and step-by-step tutorials. The *Basic features* section below gives an overview of the primary functionality of the package.


Basic features
--------------
Once you have installed the code and downloaded the default halo catalog (see the Getting Started guide in the documentation), you can use Halotools models to populate mock galaxy populations.

.. code-block:: python

    # Select a model
    from halotools.empirical_models import PrebuiltHodModelFactory
    model = PrebuiltHodModelFactory('zheng07')

    # Select a halo catalog
    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname='bolshoi', redshift=0, halo_finder='rockstar')

    # populate the catalog with the model
    model.populate_mock(halocat)

After calling *populate_mock*, your model will have a *mock* attribute storing your synthetic galaxy population. All Halotools models have a *populate_mock* method that works in this way, regardless of the features of the model. There are no restrictions on the simulation or halo-finder with which you can use the populate_mock method.

Creating alternate mocks
------------------------

All Halotools models have a **param_dict** that controls the behavior of the model. By changing the parameters in this dictionary, you can create alternative versions of your mock universe by re-populating the halo catalog as follows.

.. code-block:: python

    model.param_dict['logMmin'] = 12.1
    model.mock.populate()

Note how much faster the call to **mock.populate** is relative to **model.populate_mock(halocat)**. This is due to a large amount of one-time-only pre-processing that is carried out upon creation of the first mock universe. The process of varying *param_dict* values and repeatedly calling *model.mock.populate()* is part of a typical workflow in an MCMC-type analysis conducted with Halotools.


Modeling the galaxy-halo connection
-----------------------------------

The pre-built model factories give you a wide range of models to choose from, each based on an existing publication. Alternatively, you can use the Halotools factories to design a customized model of your own creation, such as models for stellar mass, color, size, morphology, or any property of your choosing. The modular design of the **empirical_models** sub-package allows you to mix-and-match an arbitrary number or kind of features to create your own composite model of the full galaxy population. You can choose from component models provided by Halotools, components exclusively written by you, or anywhere in between. Whatever science features you choose, any Halotools model can populate any Halotools-formatted halo catalog with the same syntax shown above.

Making mock observations
------------------------

The **mock_observables** sub-package contains a wide variety of optimized functions that you can use to study your mock galaxy population. For example, you can calculate projected clustering via the **wp** function, identify friends-of-friends groups with **FoFGroups**, or compute galaxy-galaxy lensing with **mean_delta_sigma**.

.. code-block:: python

    from halotools.mock_observables import wp
    from halotools.mock_observables import FoFGroups
    from halotools.mock_observables import mean_delta_sigma


There are many other functions provided by the **mock_observables** package, such as RSD multipoles, pairwise velocities, generalized marked correlation functions, customizable isolation criteria, void statistics, and more.

Managing simulation data
------------------------

Halotools provides end-to-end support for downloading simulation data, reducing it to a fast-loading hdf5 file with metadata to help with the bookkeeping, and creating a persistent memory of where your data is stored on disk. This functionality is handled by the **sim_manager** sub-package:

.. code-block:: python

    from halotools import sim_manager

The **sim_manager** package comes with a memory-efficient **TabularAsciiReader** designed to handle the very large file sizes that are typical of contemporary cosmological simulations. There are 20 halo catalogs available for download from the Halotools website using the **download_additional_halocat script.py**, including simulations run with different volumes, resolutions and cosmologies, and also catalogs identified using different halo-finders and at different redshift. Any simulation you store in cache can be loaded into memory in the same way, and all such catalogs have a **halo_table** attribute storing the actual data.

.. code-block:: python

    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname=any_simname, redshift=any_redshift, halo_finder=any_halo_finder)
    print(halocat.halo_table[0:10])

You are not limited to use the halo catalogs pre-processed by Halotools. The **UserSuppliedHaloCatalog** allows you to use your own simulation data and transform it into a Halotools-formatted catalog in a simple way.

.. code-block:: python

    from halotools.sim_manager import UserSuppliedHaloCatalog

Although the **sim_manager** provides an object-oriented framework for creating a persistent memory of where you store your halo catalogs, your cache is stored in a simple, human-readable ASCII log in the following location:

**$HOME/.astropy/cache/halotools/halo_table_cache_log.txt**


Project status
--------------

Halotools is a fully open-source package with contributing scientists spread across many universities. The latest stable release of the package, v0.8, is now available on pip and conda-forge. You can also install the development version of the package by cloning the master branch on GitHub and locally building the source code, as described in the installation instructions.


## Asking questions and staying up-to-date

You can contact Andrew Hearin directly by email at ahearin-at-anl-dot-gov, or by tagging @aphearin on GitHub.


Citing Halotools
----------------
If you use Halotools modules to support your science publication, please cite `Hearin et al. (2017) <https://arxiv.org/abs/1606.04106>`_, ideally taking note of the version of the code you used, e.g., v0.8::

    @ARTICLE{halotools,
           author = {{Hearin}, Andrew P. and {Campbell}, Duncan and {Tollerud}, Erik and {Behroozi}, Peter and {Diemer}, Benedikt and {Goldbaum}, Nathan J. and {Jennings}, Elise and {Leauthaud}, Alexie and {Mao}, Yao-Yuan and {More}, Surhud and {Parejko}, John and {Sinha}, Manodeep and {Sip{\"o}cz}, Brigitta and {Zentner}, Andrew},
            title = "{Forward Modeling of Large-scale Structure: An Open-source Approach with Halotools}",
          journal = {The Astronomical Journal},
         keywords = {cosmology: theory, galaxies: halos, galaxies: statistics, large-scale structure of universe, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
             year = 2017,
            month = nov,
           volume = {154},
           number = {5},
              eid = {190},
            pages = {190},
              doi = {10.3847/1538-3881/aa859f},
    archivePrefix = {arXiv},
           eprint = {1606.04106},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2017AJ....154..190H},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

License
-------

Halotools is licensed under a 3-clause BSD style license - see the licenses/LICENSE.rst file.
