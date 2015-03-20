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

Although the different sub-packages of Halotools are woven together for the science aims of the package (see :ref:`halotools_overview` for a sketch of the primary science targets), individually the sub-packages have very different functionality. 

For instructions on how to get started with making mocks, 
see :ref:`mock_making_quickstart`. 
To learn how to build a customized model that includes the 
features you are interested in studying, see :ref:`model_building`. 

Halotools comes with a halo/merger tree catalog management tool that 
makes it easy to swap back and forth between simulations, 
and link halos to their assembly histories. 

	>>> from halotools import sim_manager

The simulation manager sub-package is 
documented in the :ref:`sim_analysis`. 



