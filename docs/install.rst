.. _step_by_step_install:

************************
Package Installation
************************

To install Halotools, you can either use pip or clone the repo from GitHub and build the source code. 

Using pip
====================

The simplest way to install the latest release of the code is with pip. 

The pip install option will become available very soon. 

Building from source 
====================

The other option for installing Halotools is to clone the source code from github and call the setup file. *Before* doing this, be sure you have already installed the package dependencies described in the following section::

	git clone https://github.com/astropy/halotools.git
	cd halotools
	python setup.py install

Note that installation from source compiles the code's Cython-based back-end, which will generate a large number of compiler warnings that you can ignore.

Dependencies
============

If you installed v0.1 using pip, then most of your dependencies will be handled for you automatically. The only additional dependency you may need is:

- `h5py <http://h5py.org/>`_: 2.5 or later

The h5py package is used for fast I/O of large simulated datasets. 

If you did not use pip, then you should be aware of the following strict requirements:

- `Python <http://www.python.org/>`_: 2.7.x

- `Numpy <http://www.numpy.org/>`_: 1.9 or later

- `Scipy <http://www.scipy.org/>`_: 0.15 or later

- `Cython <http://www.cython.org/>`_: 0.23 or later

- `Astropy`_: 1.0 or later

- `BeautifulSoup <http://www.crummy.com/software/BeautifulSoup/>`_: For crawling the web for halo catalogs. 

- `Requests <http://docs.python-requests.org/en/latest/>`_: Also for crawling the web for halo catalogs. 

- `h5py <http://h5py.org/>`_: 2.5 or later

Any of the above can be installed with either pip or conda. 

.. _verifying_your_installation:

Verifying your installation 
==============================

After installing the code and its dependencies, navigate to some new working directory and execute the test suite. 

.. code:: python 

	import halotools
	halotools.test()

The full test suite is memory intensive and takes several minutes to run. It will generate a few small, temporary dummy files that you can delete or just ignore. 

Whether you installed the master branch or a release branch, the message that concludes the execution of the test suite should not indicate that there were any errors or failures. A typical acceptable test suite report will read something like "445 passed, 45 skipped in 383.2 seconds". If you installed the master branch, your message may read something like "475 passed, 4 xfailed in 374.3 seconds". The *xfail* marker is shorthand for "expected failure"; tests marked by *xfail* do not indicate a bug or installation problem; instead, this indicates that there is a new feature that has only been partially implemented. If you encounter problems when running the test suite, please be sure you have installed the package dependencies first before raising a Github Issue and/or contacting the Halotools developers.  

Once you have installed the package, see :ref:`getting_started` for instructions on how to get up and running. 





