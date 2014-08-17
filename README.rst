# Welcome to halotools!

Halotools is a python package designed  
to study large scale structure, cosmology, and galaxy evolution using 
N-body simulations and halo models. This code is publicly available at 
``https://github.com/halotools``. Most of the python code 
associated with halotools-related science is in the
directory halotools/halotools. 

### Package dependencies

1. Halotools is being written to be an affiliated package of astropy, and so astropy will need to be installed on your machine to use this package. If you are using the Anaconda python distribution, astropy comes pre-installed. Otherwise, see  github.com/astropy for installation instructions.

2. Several subpackages of halotools require the use of MPI to parallelize various expensive calculations. MPI is not necessary to make the mocks themselves, but MPI makes it possible, for example, to hook the make_mocks subpackage into an MCMC engine so that model posteriors can be computed in a reasonable amount of time. To make use of this parallelization, you will need to install the mpi4py package.

---

## Documentation

### Dependencies

1. To read the documentation, you will need to have Sphinx installed. If you are using the anaconda python distribution, "conda install sphinx" will install the package in a directory that is already in your PYTHONPATH. Otherwise, you can use a package manager like brew or macports, making sure that the installation location of your package manager is in your PYTHONPATH.

2. Sphinx builds simple class inheritance diagrams that help 
make the structure of the code visually apparent. In order to view these diagrams, you will need to install the graphviz package onto your machine. Graphviz is readily available and maintained by common package managers such as brew and macports.

### Building the docs
The command "python setup.py build_sphinx"  will build the documentation into docs/_build/html. Throwing the -o flag with this command will automatically open the html version of the documentation in your default web browser. 


