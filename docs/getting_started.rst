******************************
Getting Started with Halotools
******************************

Importing Halotools
===================

After installing halotools (see :ref:`step_by_step_install` for detailed instructions), 
you can open up a python terminal 
and load the entire package by:

    >>> import halotools

However, most of the functionality of halotools is divvied into 
sub-packages, and so it is better to load only the sub-package 
that you need for your application:

    >>> from halotools import make_mocks

For more detailed information about how to make mocks, see :ref:`mock_making_quickstart:`

    >>> from halotools import model_builder
    >>> from halotools import sim_analysis

