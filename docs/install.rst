.. _step_by_step_install:

************************
Package Installation
************************

To install Halotools, you can either use pip or clone the repo from GitHub and build the source code. 
Either way, be sure to read the :ref:`halotools_dependencies` section prior to installation. 

Using pip
====================

The simplest way to install the latest release of the code is with pip. Before installation, be sure you have installed the package dependencies described in the :ref:`halotools_dependencies` section. If you will be :ref:`installing_halotools_with_virtualenv`, activate the environment before installing with pip::

	pip install halotools

This will install the latest official release of the code. 
If you want the latest master branch, 
you will need to build the code from source following the instructions in the next section. 

.. note:: 

	Consider installing Halotools into a virtual environment. 
	Setting this up is completely straightforward and takes less than a minute, 
	even if this is your first time using a virtual environment. 
	Using a virtual environment simplifies not just the current installation 
	but also package upgrades and your subsequent workflow. 
	If you use `conda <https://www.continuum.io/downloads>`_ 
	to manage your python distribution, you can find explicit instructions 
	in the :ref:`installing_halotools_with_virtualenv` 
	section of the documentation. 

Building from source 
====================

If you don't install the latest release using pip, 
you can instead clone the cource code and call the setup file. 
Before installation, be sure you have installed the package dependencies 
described in the :ref:`halotools_dependencies` section. 
If you will be :ref:`installing_halotools_with_virtualenv`, 
activate the environment before following the instructions below. 
The first step is to clone the halotools repository::

	git clone https://github.com/astropy/halotools.git
	cd halotools

Installing one of the official releases
------------------------------------------

All official releases of the code are tagged with their version name, e.g., v0.2. 
To install a particular release::

	git checkout v0.2
	python setup.py install

This will install the v0.2 release of the code. Other official release versions (e.g., v0.1) can be installed similarly. 

Installing the most recent master branch
------------------------------------------

If you prefer to use the most recent version of the code::

	git checkout master
	python setup.py install

This will install the master branch of the code that is currently under development. While the features in the official releases have a stable API, new features being developed in the master branch may not. However, the master branch may have new features and/or performance enhancements that you may wish to use for your science application. A concerted effort is made to ensure that only thoroughly tested and documented code appears in the public master branch, though Halotools users should be aware of the distinction between the bleeding edge version in master and the official release version available through pip. 

.. note::

	Whichever version of the code you choose, installation automatically compiles the Cython-based back-end, which will generate a large number of compiler warnings that you can ignore. 

.. _halotools_dependencies: 

Dependencies
============

If you install halotools using pip, then most of your dependencies will be handled for you automatically. The only additional dependency you may need is:

- `h5py <http://h5py.org/>`_: 2.5 or later

The h5py package is used for fast I/O of large simulated datasets. 

If you did not use pip, then you should be aware of the following strict requirements:

- `Python <http://www.python.org/>`_: 2.7.x or 3.x

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

After installing the code and its dependencies, navigate to some new working directory and execute the test suite. If you installed Halotools into a virtual environment as described in the :ref:`installing_halotools_with_virtualenv` section of the documentation, activate the environment before spawning a python session and executing the code below. 

.. code:: python 

	import halotools
	halotools.test()

The full test suite is memory intensive and takes several minutes to run. It will generate a few small, temporary dummy files that you can delete or just ignore. 

Whether you installed the master branch or a release branch, the message that concludes the execution of the test suite should not indicate that there were any errors or failures. A typical acceptable test suite report will read something like "445 passed, 45 skipped in 383.2 seconds". If you installed the master branch, your message may read something like "475 passed, 4 xfailed in 374.3 seconds". The *xfail* marker is shorthand for "expected failure"; tests marked by *xfail* do not indicate a bug or installation problem; instead, this indicates that there is a new feature that has only been partially implemented. If you encounter problems when running the test suite, please be sure you have installed the package dependencies first before raising a Github Issue and/or contacting the Halotools developers.  

Once you have installed the package, see :ref:`getting_started` for instructions on how to get up and running. 





