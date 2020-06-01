:orphan:

.. _installing_halotools_with_virtualenv:

****************************************************
Installing Halotools using a virtual environment
****************************************************

If you use `conda <https://www.continuum.io/downloads>`_ to manage
your python distribution and package dependencies, it is easy to
create a virtual environment that will automatically have compatible versions of the necessary dependencies required by Halotools.
By installing into a virtual environment, you will not change any of the
packages that are already installed system-wide on your machine. In the example below, we will use conda to create a virtual environment with all the dependencies handled automatically::

	conda create -n halotools_env python=3.7 astropy numpy scipy h5py requests beautifulsoup4 cython

In order to activate this environment::

	source activate halotools_env

Then install halotools into this environment::

	pip install halotools

Or, alternatively, you can install the latest master branch by following the :ref:`install_from_source` instructions.

Any additional packages you install into the ``halotools_env`` virtual environment will not impact your system-wide environment. Whenever you want to do science involving Halotools,
just activate the environment and import the code. When you are done
and wish to return to your normal system environment::

	source deactivate



