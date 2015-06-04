************************
Package Installation
************************

.. _step_by_step_install:

Installing Halotools
====================

There are two simple options for how to install Halotools on your machine. In both cases, once you install the package you will be able to import the Halotools package from a python terminal running from any location on your machine.

Using pip
-------------

The pip install option is not yet available - coming soon!

Building from source 
--------------------------

The other option for installing Halotools is to clone the source code from github and call the setup file::

	git clone https://github.com/astropy/halotools.git
	cd halotools
	python setup.py build
	python setup.py install

Dependencies
============

Whether you installed the code with pip or using setup.py, all of the package dependencies 
will be automatically handled for you. However, if you did not install the code with 
either of the above two methods, then you will need to be aware of the following dependencies.

Core Dependencies
---------------------

Halotools is built upon the following core dependencies:

- `Python <http://www.python.org/>`_: 2.7, 3.3, or 3.4

- `Numpy <http://www.numpy.org/>`_: 1.9 or later

- `Scipy <http://www.scipy.org/>`_: 0.15 or later

- `Astropy`_: 1.0 or later

- `BeautifulSoup <http://www.crummy.com/software/BeautifulSoup/>`_: For crawling the web for halo catalogs. 

- `Requests <http://docs.python-requests.org/en/latest/>`_: Also for crawling the web for halo catalogs. 

All of the above come pre-installed with a modern python distribution such as Anaconda. In addition to the above, Halotools also requires the h5py package for fast I/O of large simulated datasets.

- `h5py <http://h5py.org/>`_: 2.5 or later

Use pip to install h5py and the other core packages, as necessary. 


Optional Dependencies
---------------------

Halotools also depends on other packages for optional features and enhanced performance:

- `mpi4py <http://mpi4py.scipy.org/>`_: For parallelizing MCMCs and various expensive simulation analyses.

- `numba <http://numba.pydata.org/>`_: For speeding up calculations using vectorization and just-in-time compiling. 

For each item in the list above, you only need to install the package if you wish to use the associated feature/enhancement. Halotools will import even if these dependencies are not installed. All optional and core dependencies can be installed with pip. 







