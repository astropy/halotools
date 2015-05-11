************************
Package Installation
************************

Dependencies
============

Core Dependencies
---------------------

Halotools is built upon the following core dependencies:

- `Python <http://www.python.org/>`_: 2.7, 3.3, or 3.4

- `Numpy <http://www.numpy.org/>`_: 1.9 or later

- `Scipy <http://www.scipy.org/>`_: 0.15 or later

- `Astropy`_: 1.0 or later

All of the above come pre-installed with a modern python distribution such as Anaconda. In addition to the above, Halotools also requires the h5py package for fast I/O of large simulated datasets.

- `h5py <http://h5py.org/>`_: 2.5 or later

Use pip to install h5py and the other core packages, as necessary. 


Optional Dependencies
---------------------

Halotools also depends on other packages for optional features and enhanced performance:

- `mpi4py <http://mpi4py.scipy.org/>`_: For parallelizing MCMCs and various expensive simulation analyses.

- `numba <http://numba.pydata.org/>`_: For speeding up calculations using vectorization and just-in-time compiling. 

- `BeautifulSoup <http://www.crummy.com/software/BeautifulSoup/>`_: For crawling the web for halo catalogs. 

For each item in the list above, you only need to install the package if you wish to use the associated feature/enhancement. Halotools will import even if these dependencies are not installed. All optional and core packages can be installed with pip. 

.. _step_by_step_install:

Installing Halotools
====================

Using pip
-------------

Installing Halotools can be accomplished with `pip <http://www.pip-installer.org/en/latest/>`_ with a single line of code executed at terminal:

	coming soon!





