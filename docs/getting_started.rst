******************************
Getting Started with Halotools
******************************

Importing Halotools
===================

After installing halotools (see :ref:`step_by_step_install` for detailed instructions), 
you can open up a python terminal and load the entire package by:

    >>> import halotools

However, most of the functionality of halotools is divvied into 
sub-packages, and so it is better to load only the sub-package 
that you need for your application. For example, if you were interested in 
building a mock galaxy distribution based on some model, you'd want to 
import the model-building sub-package:

    >>> from halotools import empirical_models

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



