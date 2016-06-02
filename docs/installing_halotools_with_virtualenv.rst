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

	conda create -n halotools_env astropy numpy scipy h5py requests beautifulsoup4 cython python=2.7.11

In order to activate this environment::

	source activate halotools_env

You may wish to add additional packages into this environment, depending on what else you want to use when working with Halotools. This can be done by tacking on additional package names when you create the environment, and/or by running *conda install pkg_name* after activating the environment. For example::

	conda create -n halotools_env astropy numpy scipy h5py requests beautifulsoup4 cython python=2.7.11 ipython matplotlib

Within the *halotools_env* environment, you can install Halotools using pip 
and you should encounter no problems with package dependencies. 
Then whenever you want to do science involving Halotools, 
just activate the environment and import the code. When you are done 
and wish to return to your normal system environment::

	source deactivate 



