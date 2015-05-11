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

First steps with Halotools
================================

Running the test suite
------------------------
The simplest way to verify that the code is up-to-date and science-ready 
is to navigate to the root directory of the package and run the test suite::

	python setup.py test

This will trigger Cython compilation of various package components, as well as 
soup-to-nuts testing of all sub-packages. This typically takes less than a minute. 
At the end, you will see a short summary of the outcome of the test suite. 


Downloading the default halo catalog
-------------------------------------

Once you have installed Halotools and verified that you can import it,
likely the first thing you will want to do is to download the default 
halo catalog so that you can quickly get up and running. You can accomplish 
this by navigating to the root directory of the package and running the initial 
download script::

	python scripts/download_initial_halocat

Running this script will set up the Halotools cache directory system on your local machine, 
and then download the default halo catalog to the cache, 
storing the catalog as a pre-processed binary file. The default catalog is ~400Mb, and if 
you are on a fast university connection the download time should be less than a minute. 

If you want to start playing with this catalog right away:

>>> from halotools import sim_manager
>>> default_snapshot = sim_manager.ProcessedSnapshot() # doctest: +SKIP
>>> print(default_snapshot.halos[0:9]) # doctest: +SKIP

Getting started with subpackages
================================

Although the different sub-packages of Halotools are woven together for the science aims of the package (see :ref:`halotools_overview`), individually the sub-packages have very different functionality. 

Downloading and processing simulations
---------------------------------------

The Halotools ``sim_manager`` sub-package  
makes it easy to download halo catalogs, process them into fast-loading binaries, 
store them in a cache directory of your choosing, and swap back and forth between 
different simulations using a single line of code. 

	>>> from halotools import sim_manager

You can find an overview of this sub-package at :ref:`cat_manage`. 


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





