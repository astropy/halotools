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

For example, if you were interested in reading/processing a N-body simulation, 
you'd want to import the Simulation Manager sub-package:

    >>> from halotools import sim_manager

Note that for clarity, and to avoid any issues, we recommend that you **never**
import any Halotools functionality using ``*``, for example::

    >>> from halotools.sim_manager import *  # NOT recommended

Getting started with subpackages
================================

Although the different sub-packages of Halotools are woven together for the science aims of the package (see :ref:`halotools_overview` for a sketch of the primary science targets), individually the sub-packages have very different functionality. You can learn about how to work with the package as a whole from the tutorials that appear throughout the :ref:`user-docs`. 

To get started with building models and making mocks, you can import the empirical modeling sub-package::

>>> from halotools import empirical_models 

A complete reference to all the classes and functions in this sub-package can be found at `~halotools.empirical_models`. For an outline of how to generate mock galaxy catalogs, see :ref:`mock_making_quickstart`. To learn how to use the empirical modeling sub-package to build a customized structure formation model that includes the features you are interested in studying, see :ref:`model_building`. 

Halotools comes with a halo/merger tree catalog management tool that 
makes it easy to swap back and forth between simulations, 
and link halos to their assembly histories. 

	>>> from halotools import sim_manager

The complete Reference/API of the simulation manager sub-package is documented at `~halotools.sim_manager`. You can find an overview of this sub-package at :ref:`sim_analysis`. 



