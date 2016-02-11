************************
Package Installation
************************

.. _step_by_step_install:

Installing Halotools
====================

To install Halotools, you can either use pip or clone the repo from GitHub and build the source code. Whichever method you choose, note that installation compiles the code's Cython-based back-end, which will generate a large number of compiler warnings that you can ignore.

Using pip
-------------

The pip install option will become available upon beta release.

Building from source 
--------------------------

The other option for installing Halotools is to clone the source code from github and call the setup file::

	git clone https://github.com/astropy/halotools.git
	cd halotools
	python setup.py install

Dependencies
============

Halotools has the following strict requirements:

- `Python <http://www.python.org/>`_: 2.7.x

- `Numpy <http://www.numpy.org/>`_: 1.9 or later

- `Scipy <http://www.scipy.org/>`_: 0.15 or later

- `Cython <http://www.cython.org/>`_: 0.23 or later

- `Astropy`_: 1.0 or later

- `BeautifulSoup <http://www.crummy.com/software/BeautifulSoup/>`_: For crawling the web for halo catalogs. 

- `Requests <http://docs.python-requests.org/en/latest/>`_: Also for crawling the web for halo catalogs. 

All of the above come pre-installed with a modern python distribution such as Anaconda. If any of the above packages is not already installed on your machine, you can install it using pip or conda. In addition to the above, Halotools also requires the h5py package for fast I/O of large simulated datasets:

- `h5py <http://h5py.org/>`_: 2.5 or later


Verifying your installation 
==============================


After installing the code and its dependencies, you should navigate to the root directory of the package and run the full test suite to make sure your copy of Halotools is science-ready. This will also generate a large number of compiler messages that you can ignore, and takes about a minute to run.

	python setup.py test 

Depending on how you have configured your copy of the gcc compiler, Mac users may need to instead run

	CC=clang python setup.py test 

If you are working from the master branch of Halotools, the message that concludes test suite execution should not indicate that there were any errors or failures. A typical acceptable test suite report will read something like "245 passed in 83.2 seconds", 
or "275 passed, 4 xfailed in 74.3 seconds". The "xfail" marker is shorthand for "expected failure"; tests marked by xfail do not indicate a bug or installation problem; instead, this indicates that there is a new feature that has only been partially implemented. If you encounter problems when running the test suite, please be sure you have installed the package dependencies first before raising a Github Issue and/or contacting the Halotools developers.  

Once you have installed, see :ref:`getting_started` for instructions on how to get up and running. 





