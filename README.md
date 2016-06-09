# Halotools

Halotools is a specialized python package for building and testing models of the galaxy-halo connection, and analyzing catalogs of dark matter halos. 
The core feature of Halotools is a modular platform for creating mock universes of galaxies starting from a catalog of dark matter halos obtained from a cosmological simulation. Functionality of the package includes:

* Fast generation of synthetic galaxy populations using HODs, abundance matching, and related methods
* Efficient algorithms for calculating galaxy clustering, lensing, z-space distortions, and other astronomical statistics 
* A modular, object-oriented framework for designing your own galaxy evolution model
* End-to-end support for downloading publicly-available halo catalogs and reducing them to fast-loading hdf5 files

The code is publicly available at https://github.com/astropy/halotools. 

---

## Documentation

The latest build of the documentation can be found at http://halotools.readthedocs.io. The documentation includes installation instructions, quickstart guides and step-by-step tutorials. The *Basic features* section below gives an overview of the primary functionality of the package. 

---

## Installation 

You can find detailed installation instructions 
in the *Package Installation* section of http://halotools.readthedocs.io. After installing the package, you should navigate to the *Quickstart Guides and Tutorials* section and follow the *Getting started with Halotools* 10-minute tutorial. This will get you set up with the default halo catalog so that you can quickly get started with creating mock galaxy populations. 

---


## Basic features

Once you have installed the code and downloaded the default halo catalog (see the Getting Started guide in the documentation), you can use Halotools models to populate mock galaxy populations. 

``` 
# Select a model 
from halotools.empirical_models import PrebuiltHodModelFactory 
model = PrebuiltHodModelFactory('zheng07')

# Select a halo catlaog
from halotools.sim_manager import CachedHaloCatalog
halocat = CachedHaloCatalog(simname='bolshoi', redshift=0, halo_finder='rockstar')

# populate the catalog with the model
model.populate_mock(halocat)
```
After calling *populate_mock*, your model will have a *mock* attribute storing your synthetic galaxy population. All Halotools models have a populate_mock method that works in this way, regardless of the features of the model. There are no restrictions on the simulation or halo-finder with which you can use the populate_mock method. 

### Creating alternate mocks

All Halotools models have a *param_dict* that controls the behavior of the model. By changing the parameters in this dictionary, you can create alternative versions of your mock universe by re-populating the halo catalog as follows. 

```
model.param_dict['logMmin'] = 12.1
model.mock.populate()
```
Note how much faster the call to *mock.populate* is relative to *model.populate_mock(halocat)*. This is due to a large amount of one-time-only pre-processing that is carried out upon creation of the first mock universe. The process of varying *param_dict* values and repeatedly calling *model.mock.populate()* is part of a typical workflow in an MCMC-type analysis conducted with Halotools.


### Modeling the galaxy-halo connection

The pre-built model factories give you a wide range of models to choose from, each based on an existing publication. Alternatively, you can use the Halotools factories to design a customized model of your own creation, such as models for stellar mass, color, size, morphology, or any property of your choosing. The modular design of the *empirical_models* sub-package allows you to mix-and-match an arbitrary number or kind of features to create your own composite model of the full galaxy population. You can choose from component models provided by Halotools, components exclusively written by you, or anywhere in between. Whatever science features you choose, any Halotools model can populate any Halotools-formatted halo catalog with the same syntax shown above. 

### Making mock observations

The *mock_observables* sub-package contains a wide variety of optimized functions that you can use to study your mock galaxy population. For example, you can calculate projected clustering via the *wp* function, identify friends-of-friends groups with *FoFGroups*, or compute galaxy-galaxy lensing with *delta_sigma*. 

```
from halotools.mock_observables import wp
from halotools.mock_observables import FoFGroups
from halotools.mock_observables import delta_sigma
```

There are many other functions provided by the *mock_observables* package, such as RSD multipoles, pairwise velocities, generalized marked correlation functions, customizable isolation criteria, void statistics, and more. 

### Managing simulation data

Halotools provides end-to-end support for downloading simulation data, reducing it to a fast-loading hdf5 file with metadata to help with the bookkeeping, and creating a persistent memory of where your data is stored on disk. This functionality is handled by the *sim_manager* sub-package:

```
from halotools import sim_manager
```

The *sim_manager* package comes with a memory-efficient *TabularAsciiReader* designed to handle the very large file sizes that are typical of contemporary cosmological simulations. There are 20 halo catalogs available for download from the Halotools website using the *halotools/scripts/download_additional_halocat.py* script, including simulations run with different volumes, resolutions and cosmologies, and also catalogs identified using different halo-finders and at different redshift. Any simulation you store in cache can be loaded into memory in the same way, and all such catalogs have a *halo_table* attribute storing the actual data. 

```
from halotools.sim_manager import CachedHaloCatalog
halocat = CachedHaloCatalog(simname = any_simname, redshift = any_redshift, halo_finder = any_halo_finder)
print(halocat.halo_table[0:10])
```

You are not limited to use the halo catalogs pre-processed by Halotools. The *UserSuppliedHaloCatalog* allows you to use your own simulation data and transform it into a Halotools-formatted catalog in a simple way. 

```
from halotools.sim_manager import UserSuppliedHaloCatalog
```
Although the *sim_manager* provides an object-oriented framework for creating a persistent memory of where you store your halo catalogs, your cache is stored in a simple, human-readable ASCII log in the following location:

**$HOME/.astropy/cache/halotools/halo_table_cache_log.txt**

---

## Project status

[![Coverage Status](https://coveralls.io/repos/astropy/halotools/badge.svg?branch=master&service=github)](https://coveralls.io/github/astropy/halotools?branch=master)

Halotools is a fully open-source package with contributing scientists spread across many universities. The first stable release of the package, v0.2, is now available on pip. Halotools is and will remain an evolving software package, but the API of the classes, methods and functions in v0.2 is stable. 

---

## Asking questions and staying up-to-date

If you would like to receive notifications of new code releases, sign up for the google group 

https://groups.google.com/forum/#!forum/halotools

The google group also serves as a public forum for asking questions about the code. You are also welcome to contact Andrew Hearin directly by email or by tagging @aphearin on GitHub. 

---

## Citing Halotools

The v0.2 release will be accompanied with a code-release paper, currently being housed and developed in the repository stored here: https://github.com/aphearin/halotools_release_paper.

If you use Halotools modules to support your science publication, we ask that you cite this paper in your work. A link to the source of the paper will be added to this section of the documentation as soon as the release-paper is submitted. See the *License and Citation Information* section of http://halotools.readthedocs.io for further information about how to properly cite Halotools. 


---

## License 

Halotools is licensed under a 3-clause BSD style license - see the licenses/LICENSE.rst file.

