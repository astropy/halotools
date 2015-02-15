# Welcome to halotools!

Halotools is a python package designed to study large-scale structure,
cosmology, and galaxy evolution using N-body simulations and halo
models. This code is publicly available at
https://github.com/astropy/halotools. Most of the python code
associated with halotools-related science is in the
directory halotools/halotools. 

### Package dependencies

1. Halotools is being written to be an affiliated package of astropy, and so astropy will need to be installed on your machine to use this package. If you are using the Anaconda python distribution, astropy comes pre-installed. Otherwise, see  github.com/astropy for installation instructions.

2. Several subpackages of halotools require the use of MPI to parallelize various expensive calculations. MPI is not necessary to make the mocks themselves, but MPI makes it possible, for example, to hook the make_mocks subpackage into an MCMC engine so that model posteriors can be computed in a reasonable amount of time. To make use of this parallelization, you will need to install the mpi4py package.

---

## Documentation

The latest build of the documentation can be found at http://halotools.readthedocs.org. If you would like to build your own copy of the docs on your local machine, see below.

### Dependencies

1. Building the documentation requires Sphinx, so Halotools automatically 
installs it installs it for you upon setup if you do not already have it. 

2. Sphinx uses the graphviz package to build simple class inheritance diagrams that help 
make the structure of the code visually apparent. In order to view these diagrams, you will need to install the graphviz package onto your machine, so Halotools will install this package 
upon setup. 

### Building the docs

When run from the root directory of Halotools, 
the command `python setup.py build_sphinx`  will build the documentation into docs/_build/html. Throwing the -o flag with this command will automatically open the html version of the documentation in your default web browser. 

---

## Contributing

Contributions to the halotools package are warmly welcomed! 
If you are interested in collaborating, and/or would like to know more 
about the intended scope of the project, please contact Andrew Hearin 
at andrew-dot-hearin-at-yale-dot-edu.

---

## Running the test suite

The halotools package includes a test suite designed around py.test. 
To run the test suite, navigate to the root directory of the package, and run 
the command `python setup.py test`. 










