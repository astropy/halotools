************************
Package Installation
************************

.. _step_by_step_install:

Installing Halotools
====================

There are two simple options for how to install Halotools on your machine. In both cases, once you install the package you will be able to import the Halotools package from a python interpreter running from any location on your machine.

Using pip
-------------

The pip install option will become available upon the package beta-release - coming soon!

Building from source 
--------------------------

The other option for installing Halotools is to clone the source code from github and call the setup file::

	git clone https://github.com/astropy/halotools.git
	cd halotools
	python setup.py install

Verifying your installation 
-----------------------------

After installing the code, you should navigate to the root directory of the package and run the full test suite to make sure your copy of Halotools is science-ready. This will generate a large number of 
compiler messages that you can ignore, and takes about a minute to run. 

	python setup.py test 

Depending on how you have configured your copy of the gcc compiler, Mac users may need to instead run 

	CC=clang python setup.py test 

If you are working from the master branch of Halotools, the message that concludes test-suite execution should not indicate that there were any errors or failures. If you encounter problems when running the test suite, please be sure you have installed the package dependencies first before raising a Github Issue and/or contacting the Halotools developers.  

Dependencies
============

If you installed the code with pip then all of the package dependencies 
will be automatically handled for you. However, if you installed using setup.py after 
cloning the main repository (currently the only available method), 
then you will need to be aware of the following dependencies.

Core Dependencies
---------------------

Halotools is built upon the following core dependencies:

- `Python <http://www.python.org/>`_: 2.7.x

- `Numpy <http://www.numpy.org/>`_: 1.9 or later

- `Scipy <http://www.scipy.org/>`_: 0.15 or later

- `Cython <http://www.cython.org/>`_: 0.23 or later

- `Astropy`_: 1.0 or later

All of the above come pre-installed with a modern python distribution such as Anaconda. In addition to the above, Halotools also requires the h5py package for fast I/O of large simulated datasets:

- `h5py <http://h5py.org/>`_: 2.5 or later

Optional Dependencies 
-----------------------

Some of the features in Halotools are enhanced through the use of additional packages. Although most of the  
Halotools package will run perfectly fine without them, you will need to install the following software 
to benefit from the full functionality of the code. 

- `BeautifulSoup <http://www.crummy.com/software/BeautifulSoup/>`_: For crawling the web for halo catalogs. 

- `Requests <http://docs.python-requests.org/en/latest/>`_: Also for crawling the web for halo catalogs. 






