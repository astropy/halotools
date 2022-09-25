.. _step_by_step_install:

************************
Package Installation
************************

To install Halotools, you can use conda-forge, pip, or clone the repo from GitHub and build the source code.
Either way, be sure to read the :ref:`halotools_dependencies` section prior to installation.

Using conda-forge and pip
=========================

The simplest way to install the latest release of the code is with conda-forge.
If you will be :ref:`installing_halotools_with_virtualenv`, activate the environment before installing::

    conda install -c conda-forge halotools

Alternatively, you can install using pip::

		pip install halotools

Either pip or conda will install the latest official release of the code.

If instead you want the latest master branch,
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

.. _install_from_source:

Building from source
====================

If you don't install the latest release using conda or pip,
you can instead clone the cource code and call the setup file.
This is the most common way to install Halotools if you want versions of the
code that have been updated since the latest official release. In this case,
after installation it is particularly important that you follow the instructions
in the :ref:`verifying_your_installation` section below.

Before installation, be sure you have installed the package dependencies
described in the :ref:`halotools_dependencies` section.
If you will be :ref:`installing_halotools_with_virtualenv`,
activate the environment before following the instructions below.
The first step is to clone the halotools repository::

	git clone https://github.com/astropy/halotools.git
	cd halotools

Installing one of the official releases
------------------------------------------

All official releases of the code are tagged with their version name, e.g., v0.7.
To install a particular release::

	git checkout v0.7
	pip install .

This will install the v0.7 release of the code. Other official release versions (e.g., v0.5) can be installed similarly.

Installing the most recent master branch
------------------------------------------

If you prefer to use the most recent version of the code::

	git checkout master
	pip install .

This will install the master branch of the code that is currently under development. While the features in the official releases have a stable API, new features being developed in the master branch may not. However, the master branch may have new features and/or performance enhancements that you may wish to use for your science application. A concerted effort is made to ensure that only thoroughly tested and documented code appears in the public master branch, though Halotools users should be aware of the distinction between the bleeding edge version in master and the official release version available through pip.

.. note::

	**Optional:** If you need to fine-tune the optimization of an especially
	performance-critical science application,
	we recommend that you install the package from source.
	This will give you the opportunity to manually
	throw your own compiler flags that are enabled by
	your version of gcc. For example, throwing the
	``-Ofast`` and ``-march=native`` flags
	can improve performance of the `~halotools.mock_observables`
	functions by 10-40% (with zero impact on the performance
	of the mock-making algorithm implemented in `~halotools.empirical_models`).
	To compile Halotools with these flags thrown,
	simply add two new elements to the
	``extra_compiler_args`` list in every source code file
	named ``setup_package.py``: the string ``'-Ofast'`` and
	the string ``'-march=native'``.
	When you've made these modifications to the code,
	install Halotools by following the *Building fom source* instructions above
	using with your locally modified source code.
	Alternatively, if you have an older version of gcc that
	does not support the default choice for these flags made by Halotools,
	you may need to *remove* the flag causing the installation problem.

.. _halotools_dependencies:

Dependencies
============

If you install halotools using conda or pip, then most of your dependencies will be handled for you automatically. The only additional dependency you may need is:

- `h5py <http://h5py.org/>`_: 3.7 or later

The h5py package is used for fast I/O of large simulated datasets.

If you did not use conda or pip, then you should be aware of the following strict requirements:

- `Python <http://www.python.org/>`_: 3.9.x

- `Numpy <http://www.numpy.org/>`_: 1.9 or later

- `Scipy <http://www.scipy.org/>`_: 0.15 or later

- `Cython <http://www.cython.org/>`_: 0.29.32 or later

- `Astropy`_: 5.0 or later

- `BeautifulSoup <http://www.crummy.com/software/BeautifulSoup/>`_: For crawling the web for halo catalogs.

- `Requests <http://docs.python-requests.org/en/latest/>`_: Also for crawling the web for halo catalogs.

Any of the above can be installed with either pip or conda.

.. _verifying_your_installation:

Verifying your installation
==============================

After installing the code and its dependencies, start up a Python interpreter and
check that the version number matches what you expect:

.. code:: python

	import halotools
	print(halotools.__version__)

If the version number is not what it should be, this likely means you have a previous
installation that is superseding the version you tried to install. This *should* be accomplished by doing `conda remove halotools` before your new installation, but you may need to uninstall the previous build "manually". Like all python packages, you can find the installation location as follows:

.. code:: python

	import halotools
	print(halotools.__file__)

This will show where your active version is located on your machine. You can manually delete this copy of Halotools prior to your new installation to avoid version conflicts. (There may be multiple copies of Halotools in this location, depending on how may times you have previously installed the code - all such copies may be deleted prior to reinstallation).

Once you have installed the package, see :ref:`getting_started` for instructions on how to get up and running.

Testing your installation
=========================

To verify that your Halotools installation runs properly, navigate to some new working directory and execute the test suite. If you installed Halotools into a virtual environment as described in the :ref:`installing_halotools_with_virtualenv` section of the documentation, activate the environment before spawning a python session and executing the code below.

For halotools versions v0.6 and later, there is a `test_installation` feature that runs a few simple tests scattered throughout the code base:

.. code:: python

	import halotools
	halotools.test_installation()  #  v0.6 and later

For earlier versions, you will need to run the full test suite, which is more memory intensive and takes several minutes to run:

.. code:: python

	halotools.test()  #  v0.5 and earlier


Whether you installed the master branch or a release branch, the message that concludes the execution of the test suite should not indicate that there were any errors or failures. A typical acceptable test suite report will read something like "445 passed, 45 skipped in 383.2 seconds". If you installed the master branch, your message may read something like "475 passed, 4 xfailed in 374.3 seconds". The *xfail* marker is shorthand for "expected failure"; tests marked by *xfail* do not indicate a bug or installation problem; instead, this indicates that there is a new feature that has only been partially implemented. If you encounter problems when running the test suite, please be sure you have installed the package dependencies first before raising a Github Issue and/or contacting the Halotools developers.


Troubleshooting
==================
See :ref:`installation_troubleshooting` for solutions to known installation-related problems.
